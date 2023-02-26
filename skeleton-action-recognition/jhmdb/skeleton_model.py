#HYPERPARAMETERS:
LEARNING_RATE = 0.001
EPOCHS = 2000
BATCH_SIZE = 128
MAX_FRAMES = 39
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import time
import random

RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
parser.add_argument('--split', default=1)
parser.add_argument('--save_all_models', default=False)
args = parser.parse_args()

BASE_DIR = args.drive_dir
VERSION = args.version
SPLIT = args.split
SAVE_ALL_MODELS = args.save_all_models

MODEL_SAVE_DIR = f"{BASE_DIR}/models"

PROCESSED_JOINT_DATA_FOLDER = f"{BASE_DIR}/processed_joints" 
SPLITS_FOLDER = f"{BASE_DIR}/splits/splits"

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

categories = os.listdir(PROCESSED_JOINT_DATA_FOLDER)

print(f"\nCategories: {categories}")

train_split = []
test_split = []

for c in categories:

    splits_file = f"{SPLITS_FOLDER}/{c}_test_split{SPLIT}.txt"
    
    with open(splits_file, 'r') as f:
        splits_lines = f.readlines()
    
    for l in splits_lines:

        split_l = l.split(' ')
        
        if split_l[1][0] == '1':
            train_split.append(split_l[0][:-4])
        elif split_l[1][0] == '2':
            test_split.append(split_l[0][:-4])
        else:
            print("SOMETHING WRONG")
            print(split_l)

print("\nSplits")
print(f"Train: {len(train_split)}")
print(f"Test: {len(test_split)}")

X_train = []
y_train = []

X_test = []
y_test = []

print("\nLoading data...")

for c in categories:

    for i in os.listdir(f"{PROCESSED_JOINT_DATA_FOLDER}/{c}"):

        with open(f"{PROCESSED_JOINT_DATA_FOLDER}/{c}/{i}", 'r') as f:
            data = np.asarray(json.load(f))

        channel_first_data = []
        for j in range(data.shape[2]):
            channel_first_data.append(data[:,:,j])
        
        data = np.asarray(channel_first_data)
        data = np.pad(data, [(0,0), (0,0), (0,MAX_FRAMES-data.shape[2])])

        new_data = []
        for j in range(0,data.shape[0],8):
            new_data.append(data[j])
        data = np.asarray(new_data)

        if i[:-5] in train_split:
            X_train.append(data)
            y_train.append(categories.index(c))

            #For appending the inverse
            X_train.append(data * -1)
            y_train.append(categories.index(c))

        elif i[:-5] in test_split:
            X_test.append(data)
            y_test.append(categories.index(c))
        else:
            print("SOMETHING WRONG")
            print(i[:-5])

print("Done loading!\n")

X_test, y_test = shuffle(X_test, y_test, random_state=RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(5, 128, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.vertical_convolutions = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(10,3)),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*37,1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, len(categories)),
            nn.Softmax(dim=1)
        )

    def forward(self, i):
        
        x = self.conv_block_1(i)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.vertical_convolutions(x)
        x = self.fc(x)

        return x

cnn_net = CNN()
if device != cpu:
    cnn_net = nn.DataParallel(cnn_net)

checkpoint = args.load_checkpoint
if not checkpoint is None:
    cnn_net.load_state_dict(torch.load(f"{MODEL_SAVE_DIR}/m_{VERSION}/{checkpoint}")['model_state_dict'])
else:
    checkpoint = 0

cnn_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    cnn_net.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

train_accuracies = []
val_accuracies = []

for e in range(int(checkpoint), EPOCHS):

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    #shuffle dataset
    X_train, y_train = shuffle(X_train, y_train, random_state=RANDOM_STATE)

    losses = []
    train_correct = 0
    train_total = 0

    batch_input = []
    batch_actual = []

    train_predicted = []
    train_actual = []
    val_predicted = []
    val_actual = []

    pbar = tqdm(total=len(X_train))
    optimizer.zero_grad()

    for X, y in zip(X_train, y_train):

        pbar.update(1)

        batch_input.append(X)
        batch_actual.append(y)

        if len(batch_input) == BATCH_SIZE:
            input_tensor = torch.from_numpy(np.asarray(batch_input)).float().to(device)
            batch_predicted = cnn_net(input_tensor)

            loss = criterion(
                batch_predicted,
                torch.tensor(batch_actual).to(device).long())
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            for output, label in zip(batch_predicted.argmax(dim=1).cpu().detach().numpy(), batch_actual):
                if output == label:
                    train_correct += 1
                train_total += 1

                train_predicted.append(output)
                train_actual.append(label)
            
            pbar.set_description(f"{(train_correct / train_total) * 100}% Correct :)")

            batch_input = []
            batch_actual = []

            optimizer.zero_grad()

    scheduler.step()

    if len(batch_input) != 0:
        input_tensor = torch.from_numpy(np.asarray(batch_input)).float().to(device)
        batch_predicted = cnn_net(input_tensor)

        loss = criterion(
            batch_predicted,
            torch.tensor(batch_actual).to(device).long())
        loss.backward()
        optimizer.step()

        losses.append(loss.item()) 

        for output, label in zip(batch_predicted.argmax(dim=1).cpu().detach().numpy(), batch_actual):
            if output == label:
                train_correct += 1
            train_total += 1

            train_predicted.append(output)
            train_actual.append(label)
        
        pbar.set_description(f"{(train_correct / train_total) * 100}% Training Correct :)")

    pbar.close()

    train_accuracies.append(train_correct / train_total)
    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    with torch.no_grad():

        val_correct = 0

        pbar = tqdm(total=len(X_test))
        count = 0

        val_loss = 0

        for X, y in zip(X_test, y_test):

            pbar.update(1)

            input_tensor = torch.from_numpy(np.asarray([X])).float()

            out = cnn_net(input_tensor)
            pred = out.argmax(dim=1).item()

            val_loss += criterion(out, torch.tensor([y]).to(device).long())

            if pred == y:
                val_correct += 1
            
            val_predicted.append(pred)
            val_actual.append(y)
            
            count += 1
            pbar.set_description(f"{(val_correct / count) * 100}% Validation Correct :)")

        pbar.close()
        time.sleep(1)

        val_accuracies.append(val_correct / len(y_test))
        print(f"Epoch {e} Validation Accuracy: {val_correct / len(y_test)}")


    print(confusion_matrix(val_actual, val_predicted))

    print("---------------------------------------------------------------")

    if SAVE_ALL_MODELS:
        torch.save({
            'model_state_dict': cnn_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{MODEL_SAVE_DIR}/m_{VERSION}/m_{e}")
    else:
        torch.save({
            'model_state_dict': cnn_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{MODEL_SAVE_DIR}/m_{VERSION}/model")

    torch.save({
        'epoch': e,
        'loss': losses,
        'train_predicted': train_predicted,
        'train_actual': train_actual,
        'val_predicted': val_predicted,
        'val_actual': val_actual,
        'categories': categories
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

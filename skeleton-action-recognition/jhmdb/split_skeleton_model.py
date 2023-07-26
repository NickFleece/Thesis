# hyperparameters that can be overwritten
EPOCHS = 2000

MAX_FRAMES = 39
import os

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
parser.add_argument('--learning_rate', default=0.01)
parser.add_argument('--batch_size', default=16)
parser.add_argument('--num_filters', default=512)
parser.add_argument('--weight_decay', default=0.005)
parser.add_argument('--gpu', default="1")
parser.add_argument('--verbose', default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

BASE_DIR = args.drive_dir
VERSION = args.version
SPLIT = args.split
SAVE_ALL_MODELS = args.save_all_models
VERBOSE = int(args.verbose)

LEARNING_RATE = float(args.learning_rate)
BATCH_SIZE = float(args.batch_size)
NUM_FILTERS = int(args.num_filters)
WEIGHT_DECAY = float(args.weight_decay)

MODEL_SAVE_DIR = f"{BASE_DIR}/models"

ANGLE_DATA_FOLDER = f"{BASE_DIR}/processed_joints_only_angle"
ANGLE_CHANGE_DATA_FOLDER = f"{BASE_DIR}/processed_joints_only_change"
SPLITS_FOLDER = f"{BASE_DIR}/splits/splits"

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

categories = os.listdir(ANGLE_DATA_FOLDER)

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

    for i in os.listdir(f"{ANGLE_DATA_FOLDER}/{c}"):

        with open(f"{ANGLE_DATA_FOLDER}/{c}/{i}", 'r') as f:
            angle_data = np.asarray(json.load(f))

        angle_data = np.pad(angle_data, [(0,0), (0,0), (0,MAX_FRAMES-angle_data.shape[2])])

        with open(f"{ANGLE_CHANGE_DATA_FOLDER}/{c}/{i}", 'r') as f:
            change_data = np.asarray(json.load(f))

        change_data = np.pad(change_data, [(0,0), (0,0), (0,MAX_FRAMES-change_data.shape[2])])

        data = np.asarray([angle_data, change_data])

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

        self.angle_block = nn.Sequential(
            # Conv 1
            nn.Conv2d(14, NUM_FILTERS, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Conv 2
            nn.Dropout(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS*2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Conv 3
            nn.Dropout(),
            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*4, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS*4, NUM_FILTERS*4, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Pool, first fc
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(NUM_FILTERS*4, NUM_FILTERS*4),
        )

        self.change_block = nn.Sequential(
            # Conv 1
            nn.Conv2d(14, NUM_FILTERS, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Conv 2
            nn.Dropout(),
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS*2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*2, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*2),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Conv 3
            nn.Dropout(),
            nn.Conv2d(NUM_FILTERS*2, NUM_FILTERS*4, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(),
            nn.Conv2d(NUM_FILTERS*4, NUM_FILTERS*4, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(NUM_FILTERS*4),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            # Pool, first fc
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(NUM_FILTERS*4, NUM_FILTERS*4),
        )

        self.fc = nn.Sequential(
            nn.Linear(NUM_FILTERS*8, NUM_FILTERS*4),
            nn.Linear(NUM_FILTERS*4, len(categories)),
            nn.Softmax(dim=1)
        )

    def forward(self, i):
       
        x_angle = self.angle_block(i[:,0])
        x_change = self.change_block(i[:,1])

        x = torch.cat([x_angle, x_change], dim=1)

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
    momentum=0.9,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

train_accuracies = []
val_accuracies = []

start_time = time.time()

for e in range(int(checkpoint), EPOCHS):

    if VERBOSE == 1:
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

    if VERBOSE == 1:
        pbar = tqdm(total=len(X_train))
    optimizer.zero_grad()

    for X, y in zip(X_train, y_train):

        if VERBOSE == 1:
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
            
            if VERBOSE == 1:
                pbar.set_description(f"{(train_correct / train_total) * 100}% Correct :)")

            batch_input = []
            batch_actual = []

            optimizer.zero_grad()

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
        
        if VERBOSE == 1:
            pbar.set_description(f"{(train_correct / train_total) * 100}% Training Correct :)")

    if VERBOSE == 1:
        pbar.close()

    train_accuracies.append(train_correct / train_total)

    if VERBOSE == 1:
        print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    with torch.no_grad():

        val_correct = 0

        if VERBOSE == 1:
            pbar = tqdm(total=len(X_test))
        count = 0

        val_loss = 0

        for X, y in zip(X_test, y_test):
            
            if VERBOSE == 1:
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

            if VERBOSE == 1:
                pbar.set_description(f"{(val_correct / count) * 100}% Validation Correct :)")

        if VERBOSE == 1:
            pbar.close()
        time.sleep(1)

        val_accuracies.append(val_correct / len(y_test))
        if VERBOSE == 1:
            print(f"Epoch {e} Validation Accuracy: {val_correct / len(y_test)}")

    scheduler.step()

    if VERBOSE == 1:
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

    if e != 0:
        time_diff = time.time() - start_time
        time_left = (time_diff / e) * (EPOCHS - e)
    else:
        time_left = 0

    time_left_sec = time_left % 60
    time_left_min = time_left // 60

    if VERBOSE == 2:
        print(f"Model: {VERSION} Epoch: {e} Train: {round((train_correct / train_total) * 100, 3)} Val: {round((val_correct / len(y_test)) * 100, 3)} {time_left_min} minutes {time_left_sec} seconds estimated left")

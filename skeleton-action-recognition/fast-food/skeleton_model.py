#HYPERPARAMETERS:
LEARNING_RATE = 1e-2
EPOCHS = 1000
BATCH_SIZE = 5
MAX_FRAMES = 881
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

RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir
VERSION = args.version

MODEL_SAVE_DIR = f"{BASE_DIR}/models"

data_summary = pd.read_csv(f"{BASE_DIR}/clean_data_summary_v2.csv")
data_summary = data_summary.fillna("None")

print(data_summary)

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

categories = list(data_summary['category'].unique())
categories.remove(' ')
category_counts = data_summary['category'].value_counts()

print(categories)
print(category_counts)

x = []
y = []

for _, d in data_summary.iterrows():

    if d['category'] == ' ': continue

    y.append(
        categories.index(d['category'])
    )

    x.append(
        f"{d['folder']}/processed_extracted_pose_new/{d['category']}~{d['instance_id']}~{d['person_id']}.json"
    )

x_data = []
for x_path in x:
    with open(f"{BASE_DIR}/{x_path}", 'r') as f:
        skeleton_data = np.asarray(json.load(f))

    skeleton_data = np.pad(skeleton_data, [(0,0), (0,MAX_FRAMES-skeleton_data.shape[1]), (0,0)])

    # channel last to channel first
    channel_first_final_data = []
    for i in range(skeleton_data.shape[2]):
        channel_first_final_data.append(skeleton_data[:,:,i])

    x_data.append(channel_first_final_data)

#sample data
new_x_data = []
new_y = []

for x_1, y_1 in zip(x_data, y):

    new_x_data.append(x_1)
    new_y.append(y_1)

    if y_1 == categories.index("picking_up"): continue
    
    new_x_data.append(np.asarray(x_1)*-1)
    new_y.append(y_1)

x_data = new_x_data
y = new_y

X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(10, 128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, len(categories)),
            nn.Softmax(dim=1)
        )

    def forward(self, i):
        
        x = self.conv_block_1(i)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3)

train_accuracies = []
val_accuracies = []

for e in range(int(checkpoint), EPOCHS):

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

        scheduler.step(val_loss/len(X_test))

        pbar.close()
        time.sleep(1)

        val_accuracies.append(val_correct / len(y_test))
        print(f"Epoch {e} Validation Accuracy: {val_correct / len(y_test)}")


    print(confusion_matrix(val_actual, val_predicted))

    print("---------------------------------------------------------------")

    torch.save({
        'epoch': e,
        'model_state_dict': cnn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
        'train_predicted': train_predicted,
        'train_actual': train_actual,
        'val_predicted': val_predicted,
        'val_actual': val_actual,
        'categories': categories
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

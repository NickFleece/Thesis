import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
from ntu_skeleton_config import THREE_PERSON_SKELETON_FILES

#random state for consistency
RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir
VERSION = args.version

PROCESSED_SKELETON_FILES_DIR = f"{BASE_DIR}/ntu_processed_skeleton_data"
MODEL_SAVE_DIR = f"{BASE_DIR}/models"

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

# maximum amount of frames in a single video
MAX_FRAMES = 299

def load_data(json_path):
    with open(f"{PROCESSED_SKELETON_FILES_DIR}/{json_path}", 'rb') as f:
        skeleton_json = json.load(f)

    final_skeleton_json = []
    for person in skeleton_json.keys():
        
        person_skeleton_json = np.asarray(skeleton_json[person])
        person_skeleton_json = np.pad(person_skeleton_json, [(0,0), (0,MAX_FRAMES-person_skeleton_json.shape[1]), (0,0)])

        # channel last to channel first
        new_person_skeleton_json = []
        for i in range(person_skeleton_json.shape[2]):
            new_person_skeleton_json.append(person_skeleton_json[:,:,i])
        new_person_skeleton_json = np.asarray(new_person_skeleton_json)
       
        final_skeleton_json.append(new_person_skeleton_json.tolist())

    # 1 person, add second person padding
    if len(final_skeleton_json) == 1:
        final_skeleton_json.append(np.zeros(np.asarray(final_skeleton_json).shape[1:]).tolist())

    # reshape from (2,2,24,299) to (299, 96)
    joint_combined_json = []
    for person in final_skeleton_json:
       for channel in person:
           for joint in channel:
               joint_combined_json.append(joint)
    joint_combined_json = np.asarray(joint_combined_json)

    final_json = []
    for i in range(joint_combined_json.shape[1]): final_json.append(joint_combined_json[:,i])

    return final_json

data = []
classes = []

all_data_files = os.listdir(PROCESSED_SKELETON_FILES_DIR)
for json_path in all_data_files:
    if json_path in THREE_PERSON_SKELETON_FILES: continue

    json_class = json_path[:-5].split("A")[1].strip("0")

    data.append(json_path)
    classes.append(json_class)

X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, random_state=RANDOM_STATE, stratify=classes)

# HYPERPARAMETERS:
LEARNING_RATE = 0.1
EPOCHS = 100
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.rnn = nn.RNN(
           96,
           1000,
           #num_layers=2,
           batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(299000,1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000, 60),
            nn.Softmax(dim=1)
        )

    def forward(self, i):
        x, _ = self.rnn(i)
        x = self.flatten(x)
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
optimizer = optim.Adam(
    cnn_net.parameters(),
    lr=LEARNING_RATE,
    #momentum=0.9
)

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

    pbar = tqdm(total=len(X_train))
    optimizer.zero_grad()

    for X, y in zip(X_train, y_train):

        pbar.update(1)

        batch_input.append(load_data(X))
        batch_actual.append(int(y))

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
        
        pbar.set_description(f"{(train_correct / train_total) * 100}% Training Correct :)")

    pbar.close()

    train_accuracies.append(train_correct / train_total)
    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    with torch.no_grad():

        val_correct = 0

        pbar = tqdm(total=len(X_test))
        count = 0

        for X, y in zip(X_test, y_test):

            pbar.update(1)

            input_tensor = torch.from_numpy(np.asarray([load_data(X)])).float()

            pred = cnn_net(input_tensor).argmax(dim=1).item()

            if pred == int(y):
                val_correct += 1
            
            count += 1
            pbar.set_description(f"{(val_correct / count) * 100}% Validation Correct :)")

        pbar.close()
        time.sleep(1)

        val_accuracies.append(val_correct / len(y_test))
        print(f"Epoch {e} Validation Accuracy: {val_correct / len(y_test)}")

    print("---------------------------------------------------------------")

    torch.save({
        'epoch': e,
        'model_state_dict': cnn_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

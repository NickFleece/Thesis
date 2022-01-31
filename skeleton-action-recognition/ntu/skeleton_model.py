import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sklearn
import argparse
from sklearn.model_selection import train_test_split

#random state for consistency
RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir

PROCESSED_SKELETON_FILES_DIR = f"{BASE_DIR}/ntu_processed_skeleton_data"
MODEL_SAVE_DIR = f"{BASE_DIR}/models"

# maximum amount of frames in a single video
MAX_FRAMES = 299

data = []
classes = []

print("Loading data into memory...")

for json_path in os.listdir(PROCESSED_SKELETON_FILES_DIR)[:1000]:
    with open(f"{PROCESSED_SKELETON_FILES_DIR}/{json_path}", 'rb') as f:
        skeleton_json = json.load(f)

    json_class = json_path[:-5].split("A")[1].strip("0")

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

    data.append(final_skeleton_json)
    classes.append(json_class)

X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, random_state=RANDOM_STATE, stratify=classes)

# HYPERPARAMETERS:
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 100),
            #nn.Softmax(dim=1)
        )

        self.rnn = nn.RNN(100, 200)

        self.final_fc = nn.Sequential(
            nn.Linear(200, 60),
            nn.Softmax(dim=2)
        )

    def forward(self, i):

        split = torch.split(i, 1, dim=1)

        hn = torch.zeros((1,1,200))

        for person in split:
            person = torch.squeeze(person, dim=0)

            #convolutions
            x = self.conv_block_1(person)
            x = self.conv_block_2(x)
            x = self.conv_block_3(x)

            #final flatten & fc layer
            person_cnn_output = self.fc(x)

            rnn_input = torch.unsqueeze(person_cnn_output, dim=0)

            rnn_out, hn = self.rnn(rnn_input, hn)

        x = self.final_fc(rnn_out)

        return torch.squeeze(x, dim=0)

cnn_net = CNN()
if device != cpu:
    cnn_net = nn.DataParallel(cnn_net)

checkpoint = args.load_checkpoint
if not checkpoint is None:
    cnn_net.load_state_dict(torch.load(f"{MODEL_SAVE_DIR}/m_{VERSION}/{checkpoint}"))
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

for e in range(checkpoint, EPOCHS):

    losses = []
    train_correct = 0
    train_total = 0

    batch_predicted = []
    batch_actual = []

    for X, y in zip(X_train, y_train):

        input_tensor = torch.from_numpy(np.asarray([X])).float()

        cnn_output = cnn_net(input_tensor)

        batch_actual.append(int(y))
        batch_predicted.append(cnn_output)

        if len(batch_predicted) == BATCH_SIZE:
            batch_predicted = torch.cat(batch_predicted)

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
            
            print(f"{train_correct / train_total}% Correct :)")

            batch_predicted = []
            batch_actual = []

    print("---------------------------------------------------------------")
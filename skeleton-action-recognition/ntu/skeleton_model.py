# HYPERPARAMETERS:
LEARNING_RATE = 0.0001
EPOCHS = 200
BATCH_SIZE = 512
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
from ntu_skeleton_config import THREE_PERSON_SKELETON_FILES, TWO_PERSON_SKELETON_FILES

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

    person_combined_skeleton_json = []
    for person in skeleton_json.keys():
        
        person_skeleton_json = np.asarray(skeleton_json[person])
        person_skeleton_json = np.pad(person_skeleton_json, [(0,0), (0,MAX_FRAMES-person_skeleton_json.shape[1]), (0,0)])
        
        person_combined_skeleton_json.append(person_skeleton_json)

    if len(person_combined_skeleton_json) != 1:
        print("Something has gone wrong? More than one skeleton.")
        return
    final_json = person_combined_skeleton_json[0]

    # channel last to channel first
    channel_first_final_json = []
    for i in range(final_json.shape[2]):
        channel_first_final_json.append(final_json[:,:,i] / final_json[:,:,i].max())

    return channel_first_final_json

data = []
classes = []

all_data_files = os.listdir(PROCESSED_SKELETON_FILES_DIR)
for json_path in all_data_files:
    if json_path in TWO_PERSON_SKELETON_FILES: continue
    # if json_path in THREE_PERSON_SKELETON_FILES: continue

    json_class = int(json_path[:-5].split("A")[1])
    if int(json_class) > 49: continue

    data.append(json_path)
    classes.append(json_class - 1)

#find number of classes
unique_classes = []
for c in classes:
    if c not in unique_classes:
        unique_classes.append(c)
NUM_CLASSES = len(unique_classes)

X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.2, random_state=RANDOM_STATE, stratify=classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Conv2d(128, 128, kernel_size=(3, 3)),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.Dropout(0.8),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=(3,3)),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Dropout(0.8),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=(3,3)),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.Dropout(0.8),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),
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

    train_predicted = []
    train_actual = []
    val_predicted = []
    val_actual = []

    pbar = tqdm(total=len(X_train))
    optimizer.zero_grad()

    for X, y in zip(X_train, y_train):

        pbar.update(1)

        batch_input.append(load_data(X))
        batch_actual.append(unique_classes.index(int(y)))

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

        for X, y in zip(X_test, y_test):

            pbar.update(1)

            input_tensor = torch.from_numpy(np.asarray([load_data(X)])).float()

            pred = cnn_net(input_tensor).argmax(dim=1).item()

            if pred == unique_classes.index(int(y)):
                val_correct += 1
            
            val_predicted.append(pred)
            val_actual.append(unique_classes.index(int(y)))
            
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
        'train_predicted': train_predicted,
        'train_actual': train_actual,
        'val_predicted': val_predicted,
        'val_actual': val_actual,
        'unique_classes': unique_classes
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

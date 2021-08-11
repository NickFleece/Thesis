import os
# SELECT WHAT GPU TO USE
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

# Constants/File Paths/Config things
DATA_PATH = "../JHMDB"
MODEL_SAVE_DIR = "models/2_test_model"
HIST_SAVE_DIR = "models/2_test_model_hist.pickle"
EPOCHS = 0
SLICE_INDEX = 1
BATCH_SIZE = 2
RANDOM_SEED = 123
NUM_WORKERS = BATCH_SIZE * 2
MAX_CACHE = BATCH_SIZE * 10
THREAD_WAIT_TIME = 1 # in seconds
LEARNING_RATE = 0.05

#Imports
import time
import numpy as np
import pandas as pd
import queue
import threading
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from scipy import ndimage

from appendix import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)

data = []
for c in classes:

    for i in range(1, 4):

        lines = open(f"{DATA_PATH}/splits/{c}_test_split{i}.txt").read().splitlines()

        for L in lines:
            f, train_or_test = L.split(' ')

            if train_or_test == '1':
                split = "train"

            else:
                split = "test"

            data.append({
                "file": f,
                "class": c,
                "split": split,
                "ind": i
            })

data = pd.DataFrame(data)

train_data = data.loc[data['ind'] == SLICE_INDEX].loc[data['split'] == 'train'][["file", "class"]]
val_data = data.loc[data['ind'] == SLICE_INDEX].loc[data['split'] == 'test'][["file", "class"]]

result_queue = queue.Queue()

def process_data(data):
    c = data['class']
    f = data['file']

    openpose_heatmaps_dir = f"{DATA_PATH}/OpenPose_Heatmaps/{c}/{f[:-4]}"

    all_heatmaps = []
    image_index = 0
    image_path = f"{openpose_heatmaps_dir}/{f[:-4]}_{str(image_index).zfill(12)}_pose_heatmaps.png"
    while os.path.exists(image_path):
        all_heatmaps.append(
            np.asarray(Image.open(image_path))
        )

        image_index += 1
        image_path = f"{openpose_heatmaps_dir}/{f[:-4]}_{str(image_index).zfill(12)}_pose_heatmaps.png"
    
    all_heatmaps = np.asarray(all_heatmaps)
    slice_size = int(all_heatmaps.shape[2] / 25)

    all_keypoint_speeds = [[], []]
    for i in range(len(parts)):

        keypoints = []
        for heatmap in all_heatmaps:
            part_map = heatmap[:, slice_size * i:slice_size * i + slice_size]

            keypoint = ndimage.measurements.center_of_mass(part_map)

            keypoints.append(keypoint)

        #move the channel to the 2nd shape input, get the speeds of the keypoints
        keypoints = np.asarray(keypoints)
        keypoints = np.asarray([keypoints[:, 0], keypoints[:, 1]])
        keypoint_speeds = np.diff(keypoints, axis=1)

        #have to clean up some nan values
        keypoint_speeds = np.nan_to_num(keypoint_speeds)

        all_keypoint_speeds[0].append(keypoint_speeds[0])
        all_keypoint_speeds[1].append(keypoint_speeds[1])
        # all_keypoint_speeds.append(keypoint_speeds)

    #pad the input to 39 len, normalize it
    all_keypoint_speeds = np.asarray(all_keypoint_speeds)
    all_keypoint_speeds = all_keypoint_speeds / all_keypoint_speeds.max()
    while all_keypoint_speeds.shape[2] != 39:
        all_keypoint_speeds = np.pad(all_keypoint_speeds, ((0,0),(0,0),(0,1)))

    target = classes.index(c)

    result_queue.put([all_keypoint_speeds, target])
    return [all_keypoint_speeds, target]

def load_train_data(data_df):
    data_df = data_df.sample(frac=1)

    threads = []
    batch_index = 0
    while True:

        for _, i in data_df.iloc[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE].iterrows():
            
            while len(threads) == NUM_WORKERS or (result_queue.qsize() + len(threads)) >= MAX_CACHE:
                time.sleep(THREAD_WAIT_TIME)
                threads = [t for t in threads if t.is_alive()]           
 
            t = threading.Thread(target=process_data, args=(i,))
            t.start()
            threads.append(t)
        
        batch_index += 1
        
        if (batch_index + 1) * BATCH_SIZE > len(data_df):
            break

def load_val_data(data_df):
    data_df = data_df.sample(frac=1)
    
    threads = []
    for _, i in data_df.iterrows():
        
        while len(threads) == NUM_WORKERS or (result_queue.qsize() + len(threads)) >= MAX_CACHE:
            threads = [t for t in threads if t.is_alive()]

        t = threading.Thread(target=process_data, args=(i,))
        t.start()
        threads.append(t)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=(25,3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(3,), stride=(1,)),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Conv1d(256, 512, kernel_size=(3,), stride=(1,)),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d((1,)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, len(classes)),
        )

    def forward(self, i):

        # convolutions
        x = self.conv_block_1(i)
        x = torch.squeeze(x, dim=2)
        x = self.conv_block_2(x)

        # final flatten & fc layer
        x = self.fc(x)
        return x

cnn_net = CNN()
if device != cpu:
    cnn_net = nn.DataParallel(cnn_net)
# cnn_net.double()
cnn_net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    cnn_net.parameters(),
    lr=LEARNING_RATE,
    #momentum=0.9
)

train_accuracies = []
val_accuracies = []
for e in range(EPOCHS):

    # start the background train load thread
    t = threading.Thread(target=load_train_data, args=(train_data,))
    t.start()

    # iter through batches
    losses = []
    train_correct = 0
    train_total = 0
    for i in tqdm(range(len(train_data) // BATCH_SIZE)):
        optimizer.zero_grad()

        cnn_outputs = []
        actual_labels = []
        batch = []

        for _ in range(BATCH_SIZE):
            d = result_queue.get()

            single_input = d[0]
            label = d[1]

            actual_labels.append(label)
            batch.append(single_input)

        input_tensor = torch.from_numpy(np.asarray(batch)).float()

        cnn_outputs = cnn_net(input_tensor)

        del input_tensor

        loss = criterion(cnn_outputs, torch.tensor(actual_labels).to(device).long())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        for output, label in zip(cnn_outputs.argmax(dim=1).cpu().detach().numpy(), actual_labels):
            if output == label:
                train_correct += 1
            train_total += 1

        print(train_correct / train_total)

        del cnn_outputs
        del actual_labels

    train_accuracies.append(train_correct / train_total)
    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    if not result_queue.empty():
        print("Something went wrong, result queue not empty... emptying...")
        while not result_queue.empty():
            result_queue.get()

    #Validation
    with torch.no_grad():

        val_correct = 0
        t = threading.Thread(target=load_val_data, args=(val_data,))
        t.start()

        for _ in tqdm(range(len(val_data))):
            processed_d, label = result_queue.get()
            processed_d = torch.from_numpy(np.asarray([processed_d])).float()
            pred = cnn_net(processed_d).argmax(dim=1).item()

            if pred == label:
                val_correct += 1

            del processed_d

        val_accuracies.append(val_correct / len(val_data))
        print(f"Epoch {e} Validation Accuract: {val_correct / len(val_data)}")

    print("---------------------------------------------------------------")

torch.save(cnn_net, MODEL_SAVE_DIR)
with open(HIST_SAVE_DIR, 'wb') as f:
    pickle.dump({"Train":train_accuracies, "Val":val_accuracies}, f)

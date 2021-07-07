import torch
import torch.nn as nn
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import time
import threading
import queue
import torch.optim as optim
import pickle

from appendix import *

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(device)

# Constants/File Paths
DATA_PATH = "../../../../comm_dat/nfleece/JHMDB"
MODEL_SAVE_DIR = "models/pa3d_torch_model"
HIST_SAVE_DIR = "models/pa3d_torch_model_hist.pickle"
EPOCHS = 150
SLICE_INDEX = 1
BATCH_SIZE = 8
RANDOM_SEED = 123
VIDEO_PADDED_LEN = 40
NUM_WORKERS = 16
FILTERS_1D = 6
LEARNING_RATE = 0.001
MAX_CACHE = BATCH_SIZE * 3

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_3d = nn.Conv3d(1, FILTERS_1D, kernel_size=(VIDEO_PADDED_LEN, 1, 1))

        self.conv_1_block = nn.Sequential(
            nn.Conv2d(FILTERS_1D * 25, 128, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_2_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.conv_3_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(2,2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.final_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 21),
            nn.Softmax(dim=1),
        )

    def forward(self, x):

        x = self.conv_3d(x)

        x = torch.split(x, x.shape[4] // 25, dim=4)
        x = torch.cat(x, dim=1)
        x = torch.squeeze(x, dim=2)

        x = self.conv_1_block(x)
        x = self.conv_2_block(x)
        x = self.conv_3_block(x)

        x = self.final_block(x)

        return x

cnn_net = CNN()
cnn_net = nn.DataParallel(cnn_net)
cnn_net.to(device)

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
    for i in range(VIDEO_PADDED_LEN):
        if not os.path.exists(f"{openpose_heatmaps_dir}/{f[:-4]}_{str(i).zfill(12)}_pose_heatmaps.png"):
            continue

        all_heatmaps.append(
            np.asarray(Image.open(f"{openpose_heatmaps_dir}/{f[:-4]}_{str(i).zfill(12)}_pose_heatmaps.png"))
        )

    img_shape = all_heatmaps[0].shape
    while len(all_heatmaps) != VIDEO_PADDED_LEN:
        all_heatmaps.append(np.zeros(img_shape))

    images = np.expand_dims(np.asarray(all_heatmaps), axis=0)

    target = classes.index(c)

    result_queue.put([images, target])

def load_train_data(data_df):
    data_df = data_df.sample(frac=1)

    threads = []
    batch_index = 0
    while True:

        for _, i in data_df.iloc[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE].iterrows():

            while len(threads) == NUM_WORKERS or result_queue.qsize() >= MAX_CACHE:
                threads = [t for t in threads if t.is_alive()]

            t = threading.Thread(target=process_data, args=(i,))
            t.start()
            threads.append(t)

        batch_index += 1

        if (batch_index + 1) * BATCH_SIZE > len(data_df):
            break

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    cnn_net.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)

train_accuracies = []
val_accuracies = []
for e in range(EPOCHS):

    t = threading.Thread(target=load_train_data, args=(train_data,))
    t.start()

    #iterate through batches
    losses = []
    train_correct = 0
    train_total = 1
    for i in tqdm(range(len(train_data) // BATCH_SIZE)):
        optimizer.zero_grad()

        cnn_outputs = []
        actual_labels = []
        batch = []

        for _ in range(BATCH_SIZE):
            d = result_queue.get()

            input = d[0]
            label = d[1]
            actual_labels.append(label)
            batch.append(input)

        
        input_tensor = torch.from_numpy(np.asarray(batch)).float()

        cnn_outputs = cnn_net(input_tensor)

        del input_tensor

        loss = criterion(cnn_outputs.to(cpu), torch.tensor(actual_labels).to(cpu).long())
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        for output in cnn_outputs.argmax(dim=1).cpu().detach().numpy():
            if output == label:
                train_correct += 1
            train_total += 1

        del cnn_outputs
        del actual_labels
        torch.cuda.empty_cache()

    train_accuracies.append(train_correct / train_total)
    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    if not result_queue.empty():
        print("Something went wrong, result queue not empty... emptying...")
        while not result_queue.empty():
            result_queue.get()

    #Test validation
    with torch.no_grad():

        val_data = val_data.sample(frac=1)
        thread_count = 0
        val_correct = 0
        for _, d in val_data.iterrows():

            if thread_count == NUM_WORKERS:
                processed_d, label = result_queue.get()
                processed_d = torch.from_numpy(np.asarray([processed_d])).to(device).float()
                pred = cnn_net(processed_d).argmax(dim=1).item()

                if pred == label:
                    val_correct += 1

                del processed_d

                thread_count -= 1


            t = threading.Thread(target=process_data, args=(d,))
            t.start()
            thread_count += 1

        while not thread_count == 0:
            processed_d, label = result_queue.get()
            processed_d = torch.from_numpy(np.asarray([processed_d])).to(device).float()
            pred = cnn_net(processed_d).argmax(dim=1).item()

            if pred == label:
                val_correct += 1

            del processed_d

            thread_count -= 1

        val_accuracies.append(val_correct / len(val_data))
        print(f"Epoch {e} Validation Accuracy: {val_correct / len(val_data)}")

torch.save(cnn_net, MODEL_SAVE_DIR)
with open(HIST_SAVE_DIR, 'wb') as f:
    pickle.dump({"Train":train_accuracies, "Val":val_accuracies}, f)

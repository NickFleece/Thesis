# Constants/File Paths/Config things
DATA_PATH = "../JHMDB"
MODEL_SAVE_DIR = "models/2_test_model"
HIST_SAVE_DIR = "models/2_test_model_hist.pickle"
EPOCHS = 200
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
import matplotlib.pyplot as plt
import pandas as pd
import queue
import threading
import pickle
import random
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
import os
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

for _, i in train_data.iterrows():
    #print(i)

    c = i['class']
    f = i['file']

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

    all_keypoint_speeds = [[],[]]
    for i in range(len(parts)):
        
        keypoints = []
        for heatmap in all_heatmaps:
            part_map = heatmap[:, slice_size * i:slice_size * i + slice_size]
            
            keypoint = ndimage.measurements.center_of_mass(part_map)
            keypoints.append(keypoint)

        keypoints = np.asarray(keypoints)
        keypoints = np.asarray([keypoints[:,0], keypoints[:,1]])
        keypoint_speeds = np.diff(keypoints, axis=1)

        all_keypoint_speeds[0].append(keypoint_speeds[0])
        all_keypoint_speeds[1].append(keypoint_speeds[1])
        # all_keypoint_speeds.append(keypoint_speeds)

    all_keypoint_speeds = np.asarray(all_keypoint_speeds)
    all_keypoint_speeds = all_keypoint_speeds / all_keypoint_speeds.max()
    while (all_keypoint_speeds.shape[2] != 39):
        all_keypoint_speeds = np.pad(all_keypoint_speeds, ((0,0),(0,0),(0,1)))
    break

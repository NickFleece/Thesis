#HYPERPARAMETERS:
LEARNING_RATE = 0.1
EPOCHS = 500
BATCH_SIZE = 16
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

PROCESSED_JOINT_DATA_FOLDER = f"{BASE_DIR}/processed_joints" 
SPLITS_FOLDER = f"{BASE_DIR}"

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

categories = os.listdir(PROCESSED_JOINT_DATA_FOLDER)

print(f"categories: {categories}")

train_split = []
test_split = []

for i 

x_data = []
y_data = []

for c in categories:

    print(c)

    for i in os.listdir(f"{PROCESSED_JOINT_DATA_FOLDER}/{c}"):

        with open(f"{PROCESSED_JOINT_DATA_FOLDER}/{c}/{i}", 'r') as f:
            data = json.load(f)
            x_data.append(np.asarray(data))
        y_data.append(categories.index(c))


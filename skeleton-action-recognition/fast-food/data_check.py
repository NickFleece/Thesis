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
import time

RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir

data_summary = pd.read_csv(f"{BASE_DIR}/data_summary.csv")
data_summary = data_summary.fillna("None")

max_len = 0

for _, d in data_summary.iterrows():

    with open(f"{BASE_DIR}/{d['folder']}/processed_extracted_pose/{d['category']}~{d['instance_id']}~{d['person_id']}.json", 'r') as f:
        skeleton_data = np.asarray(json.load(f))

    max_shape = max(max_len, skeleton_data.shape[1])

print(max_shape)
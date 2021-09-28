import json
import threading
import argparse
import os
import pandas as pd
import queue
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env')
args = parser.parse_args()

if args.env == 'dev':
    JSON_DIR = "/media/nick/External Boi/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/media/nick/External Boi/"
    MAX_THREADS = 4
elif args.env == 'pc':
    JSON_DIR = "D:/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "D:/"
    MAX_THREADS = 5
else:
    JSON_DIR = "/comm_dat/nfleece/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/comm_dat/nfleece/"
    MAX_THREADS = 15

TRAIN_TEST_SPLIT = 0.8

all_files = pd.read_csv(f"{BASE_DIR}file_joint_data.csv")
file_counts = all_files.loc[all_files['has_joints'] == True][['class', 'has_joints']].groupby('class', as_index=False).agg('count')

print("Classes and counts:")
classes = []
for _, d in file_counts.iterrows():
    if d['has_joints'] > 50:
        classes.append(d['class'])
        print(f"{d['has_joints']} - {d['class']}")
print("--- END ---")

all_joint_files = all_files.loc[all_files['class'].isin(classes)].loc[all_files['has_joints'] == True][['file', 'class']]

print("Classes not in the dataset used:")
for c in all_files['class'].value_counts().keys():
    if c not in classes:
        print(c)
print("--- END ---\n\n")

train = []
test = []
for c in classes:
    class_files = all_joint_files.loc[all_joint_files['class'] == c]
    train_df = class_files.sample(frac=TRAIN_TEST_SPLIT)
    test_df = class_files.drop(train_df.index)

    train.append(train_df)
    test.append(test_df)

train = pd.concat(train)
test = pd.concat(test)

train.to_csv(f"{BASE_DIR}train_data.csv")
test.to_csv(f"{BASE_DIR}test_data.csv")

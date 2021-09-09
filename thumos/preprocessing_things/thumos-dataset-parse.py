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

all_files = []
all_classes = []
for file_name in os.listdir(JSON_DIR):
    all_files.append(file_name)
    all_classes.append(file_name.split('_')[1])

data = pd.DataFrame({"file":all_files, "class":all_classes})

classes = data['class'].value_counts().keys()

data_queue = queue.Queue()

def process_data_row (row):
    with open(f"{JSON_DIR}/{row['file']}", 'r') as jsonfile:
        jsondata = np.asarray(json.load(jsonfile))
    has_joints = True
    if jsondata.shape == (16,0):
        has_joints = False

    data_queue.put({
        "file": row['file'],
        "class": row['class'],
        "has_joints": has_joints
    })

def load_all_train_data ():
    threads = []
    count = 0
    for _, d in tqdm(data.iterrows(), total=len(data)):

        while len(threads) == MAX_THREADS:
            threads = [t for t in threads if t.is_alive()]

        t = threading.Thread(target=process_data_row, args=(d,))
        t.start()
        threads.append(t)

    while len(threads) != 0:
        threads = [t for t in threads if t.is_alive()]
    data_queue.put(None)

    all_processed_data = []
    while True:
        processed_data = data_queue.get()
        if processed_data is None:
            break
        all_processed_data.append(processed_data)

    all_processed_data = pd.DataFrame(all_processed_data)

    print(all_processed_data)
    print(all_processed_data.keys())

    # sns.countplot(data=all_processed_data, x='class', hue='has_joints')
    # plt.xticks(rotation=90)
    # plt.show()

    all_processed_data.to_csv(f"{BASE_DIR}file_joint_data.csv")

load_all_train_data()

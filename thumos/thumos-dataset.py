import json
import threading
import argparse
import os
import pandas as pd
import queue
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env')
args = parser.parse_args()

if args.env == 'dev':
    JSON_DIR = "/media/nick/External Boi/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 4
else:
    JSON_DIR = "/comm_dat/nfleece/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 15

RANDOM_SEED = 123
TRAIN_FRAC = 0.8

all_files = []
all_classes = []
for file_name in os.listdir(JSON_DIR):
    all_files.append(file_name)
    all_classes.append(file_name.split('_')[1])

data = pd.DataFrame({"file":all_files, "class":all_classes})

classes = data['class'].value_counts().keys()

train_data = pd.DataFrame({"file":[], "class":[]})
val_data = pd.DataFrame({"file":[], "class":[]})
for c in classes:
    c_data = data.loc[data['class'] == c]
    train = c_data.sample(frac=TRAIN_FRAC, random_state=RANDOM_SEED)
    test = c_data.drop(train.index)

    train_data = train_data.append(train)
    val_data = val_data.append(test)

train_data_queue = queue.Queue()
val_data_queue = queue.Queue()

def process_data_row (row):
    with open(f"{JSON_DIR}/{row['file']}", 'r') as jsonfile:
        jsondata = np.asarray(json.load(jsonfile))
    if jsondata.shape == (16,0):
        return 1
    elif 'flute' in row['class']:
        print(row['file'])
    return 0

def load_all_train_data ():
    threads = []
    count = 0
    for _, d in data.iterrows():
        
        count += process_data_row(d)
        continue

        while len(threads) == MAX_THREADS:
            threads = [t for t in threads if t.is_alive()]

        t = threading.Thread(target=process_data_row, args=(d,))
        t.start()
        threads.append(t)
    print(count / len(train_data))

load_all_train_data()

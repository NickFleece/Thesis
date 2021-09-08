import pandas as pd
import argparse
import threading
import queue
import numpy as np
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--test')
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

MAX_PADDED_LEN = 900

all_files = pd.read_csv(f"{BASE_DIR}file_joint_data.csv")
file_counts = all_files.loc[all_files['has_joints'] == True][['class', 'has_joints']].groupby('class', as_index=False).agg('count')

classes = []
for _, d in file_counts.iterrows():
    if d['has_joints'] > 50:
        classes.append(d['class'])

all_joint_files = all_files.loc[all_files['class'].isin(classes)].loc[all_files['has_joints'] == True][['file', 'class']]

data_queue = queue.Queue()

def process_data_row (row):
    with open(f"{JSON_DIR}/{row['file']}", 'r') as jsonfile:
        jsondata = np.asarray(json.load(jsonfile))

    #pad to 900 frames (longest in dataset)
    jsondata = np.pad(jsondata, [(0,0), (0,900-jsondata.shape[1]), (0,0)])

    # reshape from channel last to channel first
    newjsondata = []
    for i in range(jsondata.shape[2]):
        newjsondata.append(jsondata[:,:,i])
    jsondata = np.asarray(newjsondata)

    data_queue.put([jsondata, classes.index(row['class'])])

def load_data (data):
    threads = []
    for _, d in data.iterrows():

        while len(threads) == MAX_THREADS:
            threads = [t for t in threads if t.is_alive()]

        t = threading.Thread(target=process_data_row, args=(d,))
        t.start()
        threads.append(t)

    while len(threads) != 0:
        threads = [t for t in threads if t.is_alive()]
    data_queue.put(None)


if args.test == 'y':
    load_data(all_joint_files)
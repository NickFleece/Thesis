import pandas as pd
import argparse
import threading
import queue
import numpy as np
import json
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument('--env')
parser.add_argument('--test')
parser.add_argument('--load_checkpoint')
parser.add_argument('--version')
args = parser.parse_args()

if args.env == 'dev':
    JSON_DIR = "/media/nick/External Boi/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/media/nick/External Boi/"
    MAX_THREADS = 4
elif args.env == 'pc':
    JSON_DIR = "D:/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "D:/"
    MAX_THREADS = 5
elif args.env == 'razor':
    JSON_DIR = "/comm_dat/DATA/nfleece/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/comm_dat/DATA/nfleece/"
    MAX_THREADS = 15
elif args.env == 'dev_usb':
    JSON_DIR = "/media/nick/MY USB/ThesisData/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/media/nick/MY USB/ThesisData/"
    MAX_THREADS = 4
else:
    JSON_DIR = "/comm_dat/nfleece/THUMOS_JOINT_ROTATIONS"
    BASE_DIR = "/comm_dat/nfleece/"
    MAX_THREADS = 16

MODEL_SAVE_DIR = f"{BASE_DIR}models"

MAX_PADDED_LEN = 900

train = pd.read_csv(f"{BASE_DIR}train_data.csv")
test = pd.read_csv(f"{BASE_DIR}test_data.csv")

classes = train['class'].value_counts().keys().tolist()

data_queue = queue.Queue()

def process_data_row (row):
    
    transform_random = random.random()

    with open(f"{JSON_DIR}/{row['file']}", 'r') as jsonfile:
        jsondata = np.asarray(json.load(jsonfile))

    #pad to 900 frames (longest in dataset)
    if jsondata.shape == (0,):
        print(row)
    jsondata = np.pad(jsondata, [(0,0), (0,900-jsondata.shape[1]), (0,0)])

    # reshape from channel last to channel first
    newjsondata = []
    for i in range(jsondata.shape[2]):
        newjsondata.append(jsondata[:,:,i])
    jsondata = np.asarray(newjsondata)

    if jsondata.shape != (2,18,900):
        print(f"Wrong shape: {row['file']}, {jsondata.shape}")

    # 50% chance of flipping image, swapping right and left joints
    if transform_random > 0.5: 
        jsondata[0] = jsondata[0] * -1

    data_queue.put([jsondata, classes.index(row['class']), row['file']])

def load_data (data):
    #randomize :)
    data = data.sample(frac=1)

    if args.test == 'y': pbar = tqdm(total=len(data))

    threads = []
    for _, d in data.iterrows():

        while len(threads) == MAX_THREADS:
            threads = [t for t in threads if t.is_alive()]

        t = threading.Thread(target=process_data_row, args=(d,))
        t.start()
        threads.append(t)

        if args.test == 'y': pbar.update(1)

    while len(threads) != 0:
        threads = [t for t in threads if t.is_alive()]
    data_queue.put(None)

if args.test == 'y':
    load_data(train)
    load_data(test)

import os
import json
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir

PROCESSED_SKELETON_FILES_DIR = f"{BASE_DIR}/ntu_processed_skeleton_data"

# maximum amount of frames in a single video
MAX_FRAMES = 299
data = []

for json_path in tqdm(os.listdir(PROCESSED_SKELETON_FILES_DIR)):
    with open(f"{PROCESSED_SKELETON_FILES_DIR}/{json_path}", 'rb') as f:
        skeleton_json = json.load(f)
    for person in skeleton_json.keys():
        person_skeleton_json = np.asarray(skeleton_json[person])
        data.append(person_skeleton_json)
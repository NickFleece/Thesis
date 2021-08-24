import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import threading
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--env')
args = parser.parse_args()

if args.env == 'dev':
    BASE_DIR = "/media/nick/External Boi/THUMOS14_skeletons/training"
    JSON_EXPORT_DIR = "/media/nick/External Boi/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 1
else:
    BASE_DIR = "/comm_dat/THUMOS14_skeletons/training"
    JSON_EXPORT_DIR = "/comm_dat/nfleece/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 30

skeleton = [
    ["ankler", "heelr"],
    ["ankler", "toer"],
    ["anklel", "heell"],
    ["anklel", "toel"],
    ["kneer", "ankler"],
    ["kneel", "anklel"],
    ["hipr", "kneer"],
    ["hipl", "kneel"],
    ["hipr", "shoulderr"],
    ["hipl", "shoulderl"],
    ["shoulderr", "elbowr"],
    ["shoulderl", "elbowl"],
    ["elbowr", "wristr"],
    ["elbowl", "wristl"],
    ["shoulderr", "earr"],
    ["shoulderl", "earl"]
]

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_data (f):

    with open(f"{BASE_DIR}/{f}", 'rb') as infile:
        j = json.load(infile)

    #f_skeleton_indices = []
    #for i in skeleton:
    #    f_skeleton_indices.append([
    #        j['categories'][0]['keypoints'].index(i[0]),
    #        j['categories'][0]['keypoints'].index(i[1])
    #    ])
    f_skeleton_indices = j['categories'][0]['skeleton']

    main_person = []
    for x in j['annotation_sequences']:
        if len(x['annotation_ids']) > len(main_person):
            main_person = x['annotation_ids']

    vector_keypoints = []
    for c in range(len(j['annotations'])):
        x = j['annotations'][c]
        
        #Not main person
        if not x['id'] in main_person:
            continue
        
        keypoints = list(divide_chunks(x['keypoints'], 3))

        vector_keypoints_one_frame = []
        for s in f_skeleton_indices:
            if keypoints[s[0] - 1][2] == 0 or keypoints[s[1] - 1][2] == 0:
                vector_keypoints_one_frame.append([None, None])
                continue

            skeleton_keypoints = [keypoints[s[0] - 1][:2], keypoints[s[1] - 1][:2]]
            vector_keypoint = [skeleton_keypoints[1][0] - skeleton_keypoints[0][0],
                               (240 - skeleton_keypoints[1][1]) - (240 - skeleton_keypoints[0][1])]
            vector_keypoints_one_frame.append(vector_keypoint)

        vector_keypoints.append(vector_keypoints_one_frame)
    
    vector_keypoints = np.asarray(vector_keypoints)

    all_vector_movements = []
    for c in range(len(f_skeleton_indices)):
        vector_movements_one_joint = []
        for v in range(len(vector_keypoints)):
            if v == 0:
                continue

            prev_index = v - 1
            prev_vector = None
            while prev_index > 0:
                if vector_keypoints[:, c][prev_index][0] is not None:
                    prev_vector = vector_keypoints[:, c][prev_index]
                    break
                prev_index -= 1

            if prev_vector is None or vector_keypoints[:, c][v][0] is None:
                vector_movements_one_joint.append([0.0, 0.0])
                continue

            vec_1 = vector_keypoints[:, c][prev_index]
            vec_2 = vector_keypoints[:, c][v]

            unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
            unit_vec_2 = vec_2 / np.linalg.norm(vec_2)

            ang1 = np.arctan2(*unit_vec_1[::-1])
            ang2 = np.arctan2(*unit_vec_2[::-1])
            angle = (ang2 - ang1) % (2 * np.pi)
            if angle > np.pi:
                angle = angle - (2 * np.pi)
            angle = angle / np.pi

            magnitude_vec_1 = np.linalg.norm(vec_1)
            magnitude_vec_2 = np.linalg.norm(vec_2)

            magnitude_change = (magnitude_vec_2 / magnitude_vec_1) - 1

            vector_movements_one_joint.append([angle, magnitude_change])

        all_vector_movements.append(vector_movements_one_joint)

    all_vector_movements = np.asarray(all_vector_movements)

    with open(f"{JSON_EXPORT_DIR}/{f}", 'w') as outfile:
        json.dump(all_vector_movements.tolist(), outfile)

    return all_vector_movements

files = os.listdir(BASE_DIR)
classes = []
for f in files:
    classes.append(f.split('_')[1])
data = pd.DataFrame({"file":files, "class":classes})

threads = []
for _, d in tqdm(data.iterrows(), total=len(data)):
    t = threading.Thread(target=process_data, args=(d['file'],))
    t.start()
    threads.append(t)

    while len(threads) == MAX_THREADS:
        threads = [t for t in threads if t.is_alive()]

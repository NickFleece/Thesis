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
elif args.env == 'pc':
    BASE_DIR = "D:/THUMOS14_skeletons/training"
    JSON_EXPORT_DIR = "D:/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 5
elif args.env == 'alpha':
    BASE_DIR = "/comm_dat/nfleece/THUMOS14_skeletons/training"
    JSON_EXPORT_DIR = "/comm_dat/nfleece/THUMOS_JOINT_ROTATIONS"
    MAX_THREADS = 64

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
    ["hipl", "hipr"],
    ["shoulderl", "shoulderr"],
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

    f_skeleton_indices = []
    for i in skeleton:
       f_skeleton_indices.append([
           j['categories'][0]['keypoints'].index(i[0]) + 1,
           j['categories'][0]['keypoints'].index(i[1]) + 1
       ])
    # idk why i was using this directly, causes issues with other keypoint formats
    # f_skeleton_indices = j['categories'][0]['skeleton']

    #main_person = []
    #for x in j['annotation_sequences']:
    #    if len(x['annotation_ids']) > len(main_person):
    #        main_person = x['annotation_ids']

    people = []
    for x in j['annotation_sequences']:
        people.append(x['annotation_ids'])

    if people == []: people = [[]]

    people_movements = []
    for person in people:
        vector_keypoints = []
        for c in range(len(j['annotations'])):
            x = j['annotations'][c]
            
            #Not the person we want
            if not x['id'] in person:
                continue
            
            keypoints = list(divide_chunks(x['keypoints'], 3))

            vector_keypoints_one_frame = []
            plt.figure()
            plt.xlim(320)
            plt.ylim(240)
            for s in f_skeleton_indices:
                if keypoints[s[0] - 1][2] == 0 or keypoints[s[1] - 1][2] == 0:
                    vector_keypoints_one_frame.append([None, None])
                    continue

                skeleton_keypoints = [keypoints[s[0] - 1][:2], keypoints[s[1] - 1][:2]]
                plt.plot([skeleton_keypoints[0][0], skeleton_keypoints[1][0]], [skeleton_keypoints[0][1], skeleton_keypoints[1][1]])
                vector_keypoint = [skeleton_keypoints[1][0] - skeleton_keypoints[0][0],
                                   (240 - skeleton_keypoints[1][1]) - (240 - skeleton_keypoints[0][1])]
                vector_keypoints_one_frame.append(vector_keypoint)
            plt.savefig("TEST.png")

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

        people_movements.append(np.asarray(all_vector_movements))

    all_vector_movements = np.array([])
    #people_movements_sum = np.sum(people_movements, axis=1)
    for person_movement in people_movements:
        if np.sum(np.absolute(person_movement)) > np.sum(np.absolute(all_vector_movements)):
            all_vector_movements = person_movement

    #all_vector_movements = people_movements[main_person_index]

    #this is if something goes wrong
    #if all_vector_movements.shape == (0,):
    #    print(f"File: {f} has no skeleton data, exiting")
    #    print(people_movements)
    #    return

    with open(f"{JSON_EXPORT_DIR}/{f}", 'w') as outfile:
        json.dump(all_vector_movements.tolist(), outfile)

    # print(all_vector_movements.shape)

    return all_vector_movements

files = os.listdir(BASE_DIR)
classes = []
for f in files:
    classes.append(f.split('_')[1])
data = pd.DataFrame({"file":files, "class":classes})

# process_data('v_Bowling_g25_c02.json')
# process_data('v_Archery_g03_c01.json')

threads = []
for _, d in tqdm(data.iterrows(), total=len(data)):
    # if process_data(d['file']) is None: break

    process_data('v_RockClimbingIndoor_g25_c05.json')
    break

    t = threading.Thread(target=process_data, args=(d['file'],))
    t.start()
    threads.append(t)
    
    while len(threads) == MAX_THREADS:
        threads = [t for t in threads if t.is_alive()]

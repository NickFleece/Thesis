import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

BASE_DIR = "C:/Users/Nick/Desktop/School/Thesis/THUMOS14_skeletons/training"

vidcap = cv2.VideoCapture("C:/Users/Nick/Desktop/School/Thesis/THUMOS14_skeletons/videos/v_BaseballPitch_g01_c01.avi")
_, frame = vidcap.read()

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

    with open(f"{BASE_DIR}/{f}", 'rb') as f:
        j = json.load(f)



    f_skeleton_indices = []
    for i in skeleton:
        f_skeleton_indices.append([
            j['categories'][0]['keypoints'].index(i[0]),
            j['categories'][0]['keypoints'].index(i[1])
        ])

    vector_keypoints = []
    for c in tqdm(range(len(j['annotations']))):
        x = j['annotations'][c]
        keypoints = list(divide_chunks(x['keypoints'], 3))

        plt.figure(figsize=(15,15))
        plt.grid()
        plt.xlim((0,320))
        plt.ylim((0,240))
        # plt.imshow(frame)

        vector_keypoints_one_frame = []
        for s in f_skeleton_indices:
            # print('---')
            # print(s[0])
            # print(s[1])
            if keypoints[s[0] - 1][2] == 0 or keypoints[s[1] - 1][2] == 0:
                vector_keypoints_one_frame.append([None, None])
                continue

            skeleton_keypoints = [keypoints[s[0] - 1][:2], keypoints[s[1] - 1][:2]]
            vector_keypoint = [skeleton_keypoints[1][0] - skeleton_keypoints[0][0],
                               (240 - skeleton_keypoints[1][1]) - (240 - skeleton_keypoints[0][1])]
            vector_keypoints_one_frame.append(vector_keypoint)

            # plt.plot([keypoints[s[0] - 1][0], keypoints[s[1] - 1][0]], [240 - keypoints[s[0] - 1][1], 240 - keypoints[s[1] - 1][1]])
            # plt.plot([0,vector_keypoint[0]], [0,vector_keypoint[1]])

        for k in keypoints:
            plt.scatter(k[0], k[1])

        _, frame = vidcap.read()

        # plt.show()
        plt.savefig(f"test/{c}.png")
        plt.close()

        # vector_keypoints.append(vector_keypoints_one_frame)

    return
    vector_keypoints = np.asarray(vector_keypoints)

    all_vector_movements = []
    for c in range(len(skeleton_indices)):
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

    return all_vector_movements

files = os.listdir(BASE_DIR)
classes = []
for f in files:
    classes.append(f.split('_')[1])
data = pd.DataFrame({"file":files, "class":classes})[2000:]

process_data('v_Bowling_g25_c02.json')

# for _, d in tqdm(data.iterrows(), total=len(data)):
#     try:
#         process_data(d['file'])
#     except:
#         print(d)
#         break
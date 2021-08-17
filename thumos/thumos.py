import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

BASE_DIR = "C:/Users/Nick/Desktop/School/Thesis/THUMOS14_skeletons/training"

vidcap = cv2.VideoCapture("C:/Users/Nick/Desktop/School/Thesis/THUMOS14_skeletons/videos/v_BaseballPitch_g01_c01.avi")
_, frame = vidcap.read()

# og_skeleton_indices = [[2, 1], [3, 2], [5, 6], [4, 5], [4, 3], [3, 13], [4, 14], [13, 14], [14, 15], [15, 16], [13, 12], [12, 11], [6, 23], [1, 22], [6, 25], [1, 24], [13, 19], [14, 21]]
skeleton_indices = [[2, 1], [3, 2], [5, 6], [4, 5], [3, 13], [4, 14], [14, 15], [15, 16], [13, 12], [12, 11], [6, 23], [1, 22], [6, 25], [1, 24], [13, 19], [14, 21]]
joint_indices = ['ankler', 'kneer', 'hipr', 'hipl', 'kneel', 'anklel', 'pelv', 'thrx', 'neck', 'head', 'wristr', 'elbowr', 'shoulderr', 'shoulderl', 'elbowl', 'wristl', 'nose', 'eyer', 'earr', 'eyel', 'earl', 'toer', 'toel', 'heelr', 'heell']

# for s in skeleton_indices:
#     print(f"{s} : {joint_indices[s[0]-1]} to {joint_indices[s[1]-1]}")

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

for i in os.listdir(BASE_DIR):
    if "Baseball" not in i:
        continue

    with open(f"{BASE_DIR}/{i}", 'rb') as f:
        j = json.load(f)

    # print(j['categories'][0]['keypoints'])
    # print(j['categories'][0]['skeleton'])

    vector_keypoints = []
    for c in range(len(j['annotations'])):
        x = j['annotations'][c]
        keypoints = list(divide_chunks(x['keypoints'],3))

        # plt.figure(figsize=(15,15))
        # plt.grid()
        # plt.xlim((-320,320))
        # plt.ylim((-240,240))
        # plt.imshow(frame)

        vector_keypoints_one_frame = []
        for s in skeleton_indices:
            if keypoints[s[0]-1][2] == 0 or keypoints[s[1]-1][2] == 0:
                vector_keypoints_one_frame.append([None,None])
                continue

            skeleton_keypoints = [keypoints[s[0]-1][:2], keypoints[s[1]-1][:2]]
            vector_keypoint = [skeleton_keypoints[1][0] - skeleton_keypoints[0][0], (240-skeleton_keypoints[1][1]) - (240-skeleton_keypoints[0][1])]
            vector_keypoints_one_frame.append(vector_keypoint)

            # plt.plot([keypoints[s[0] - 1][0], keypoints[s[1] - 1][0]], [240 - keypoints[s[0] - 1][1], 240 - keypoints[s[1] - 1][1]])
            # plt.plot([0,vector_keypoint[0]], [0,vector_keypoint[1]])

        # plt.savefig(f"test2/{c}.png")

        _, frame = vidcap.read()

        vector_keypoints.append(vector_keypoints_one_frame)

    vector_keypoints = np.asarray(vector_keypoints)

    # print(vector_keypoints[:,0].shape)

    all_vector_movements = []
    for c in range(len(skeleton_indices)):
        vector_movements_one_joint = []
        for v in range(len(vector_keypoints)):
            if v == 0:
                continue

            prev_index = v - 1
            prev_vector = None
            while prev_index > 0:
                if vector_keypoints[:,c][prev_index][0] is not None:
                    prev_vector = vector_keypoints[:,c][prev_index]
                    break
                prev_index -= 1

            if prev_vector is None or vector_keypoints[:,c][v][0] is None:
                vector_movements_one_joint.append([0.0, 0.0])
                continue

            vec_1 = vector_keypoints[:,c][prev_index]
            vec_2 = vector_keypoints[:,c][v]

            unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
            unit_vec_2 = vec_2 / np.linalg.norm(vec_2)

            ang1 = np.arctan2(*unit_vec_1[::-1])
            ang2 = np.arctan2(*unit_vec_2[::-1])
            angle = (ang2 - ang1) % (2 * np.pi)
            if angle > np.pi:
                angle = angle - (2*np.pi)
            angle = angle / np.pi

            magnitude_vec_1 = np.linalg.norm(vec_1)
            magnitude_vec_2 = np.linalg.norm(vec_2)

            magnitude_change = (magnitude_vec_2 / magnitude_vec_1) - 1

            vector_movements_one_joint.append([angle, magnitude_change])

            # print(f"{angle}, {magnitude_change}")

            # plt.plot([0,vec_1[0]], [0, vec_1[1]], label="one")
            # plt.plot([0,vec_2[0]], [0, vec_2[1]], label="two")
            # plt.legend()
            # plt.show()

        all_vector_movements.append(vector_movements_one_joint)

    all_vector_movements = np.asarray(all_vector_movements)

    # plt.figure()
    # plt.imshow(all_vector_movements) # have to add a 0 to make it 3 channels if you want visualizations
    # plt.savefig('hist.png')

    break
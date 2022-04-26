import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

from ntu_skeleton_config import BONE_CONNECTIONS, BONE_CONNECTIONS_V2, IGNORE_FILES

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir

SKELETON_FILES_DIR = f"{BASE_DIR}/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"
VIDEO_FILES_DIR = f"{BASE_DIR}/ntu_dataset/nturgb+d_rgb"
EXPORT_DIR = f"{BASE_DIR}/ntu_processed_skeleton_data"

pbar = tqdm(total = len(os.listdir(SKELETON_FILES_DIR)))

for i in os.listdir(SKELETON_FILES_DIR):

    if i[:-9] in IGNORE_FILES: continue

    with open(f"{SKELETON_FILES_DIR}/{i}") as f:

        pbar.set_description(i)

        num_frames = int(f.readline())
        for frame in range(num_frames):

            num_people = int(f.readline())
            
            if frame == 0:
                people_bones = {}
            for person in range(num_people):
                if person not in people_bones:
                    people_bones[person] = []

            for person in range(num_people):

                random_details = f.readline()
                num_points = int(f.readline())

                if num_points != 25:
                    print("WHAT?")
                    print(i)

                skeleton_joints = []
                for _ in range(num_points):
                    
                    point = f.readline().split(" ")[:2]
                    for p in range(len(point)): point[p] = float(point[p])
                    skeleton_joints.append(point)

                skeleton_bones = []
                for bone in BONE_CONNECTIONS_V2:
                    j1 = skeleton_joints[bone[0]-1]
                    j2 = skeleton_joints[bone[1]-1]
                    j3 = skeleton_joints[bone[2]-1]
                    skeleton_bones.append([j1, j2, j3])

                people_bones[person].append(skeleton_bones)

    bone_movements = {}

    for person in people_bones.keys():
        person_bones = people_bones[person]

        one_person_bone_movements = []

        for frame_index in range(1, len(person_bones)):

            one_frame_bone_movements = []

            for bone, prev_bone in zip(person_bones[frame_index], person_bones[frame_index - 1]):

                # plt.figure()
                # plt.xlim((-2,2))
                # plt.ylim((-2,2))

                #get our 3 joints
                j1 = bone[0]
                j2 = bone[1]
                j3 = bone[2]
                
                bone_vector_1 = [j1[0] - j2[0], j1[1] - j2[1]]
                bone_vector_2 = [j3[0] - j2[0], j3[1] - j2[1]]

                if np.linalg.norm(bone_vector_1) == 0 or np.linalg.norm(bone_vector_2) == 1:
                    one_frame_bone_movements.append([0,0,0])
                    continue

                #calculate the angle between the two bones
                bone_unit_vector_1 = bone_vector_1 / np.linalg.norm(bone_vector_1)
                bone_unit_vector_2 = bone_vector_2 / np.linalg.norm(bone_vector_2)

                #this is using the following formula: atan2( ax*by-ay*bx, ax*bx+ay*by ).
                bone_angle = np.degrees(np.arctan2(
                    bone_unit_vector_2[0] * bone_unit_vector_1[1] - bone_unit_vector_2[1] * bone_unit_vector_1[0],
                    bone_unit_vector_2[0] * bone_unit_vector_1[0] - bone_unit_vector_2[1] * bone_unit_vector_1[1]
                ))

                #now change it to the previous bone
                j1 = prev_bone[0]
                j2 = prev_bone[1]
                j3 = prev_bone[2]

                prev_bone_vector_1 = [j1[0] - j2[0], j1[1] - j2[1]]
                prev_bone_vector_2 = [j3[0] - j2[0], j3[1] - j2[1]]

                if np.linalg.norm(prev_bone_vector_1) == 0 or np.linalg.norm(prev_bone_vector_2) == 0:
                    one_frame_bone_movements.append([0,0,0])
                    continue

                #calculate the angle between the two bones
                prev_bone_unit_vector_1 = prev_bone_vector_1 / np.linalg.norm(prev_bone_vector_1)
                prev_bone_unit_vector_2 = prev_bone_vector_2 / np.linalg.norm(prev_bone_vector_2)

                #this is using the following formula: atan2( ax*by-ay*bx, ax*bx+ay*by ).
                prev_bone_angle = np.degrees(np.arctan2(
                    prev_bone_unit_vector_2[0] * prev_bone_unit_vector_1[1] - prev_bone_unit_vector_2[1] * prev_bone_unit_vector_1[0],
                    prev_bone_unit_vector_2[0] * prev_bone_unit_vector_1[0] - prev_bone_unit_vector_2[1] * prev_bone_unit_vector_1[1]
                ))

                #make both bone angles positive
                if bone_angle < 0:
                    bone_angle = 360 + bone_angle
                if prev_bone_angle < 0:
                    prev_bone_angle = 360 + prev_bone_angle

                angle_diff = bone_angle - prev_bone_angle

                if angle_diff < -180:
                    angle_diff = angle_diff + 360
                elif angle_diff > 180:
                    angle_diff = angle_diff - 360

                # print(bone_unit_vector_1)
                # print(bone_unit_vector_2)
                # print(prev_bone_unit_vector_1)
                # print(prev_bone_unit_vector_2)
                # print(bone_angle)
                # print(prev_bone_angle)
                # print(angle_diff)
                # plt.plot([0,bone_unit_vector_1[0]], [0, bone_unit_vector_1[1]])
                # plt.plot([0,bone_unit_vector_2[0]], [0, bone_unit_vector_2[1]], label="curr")
                # plt.plot([0,prev_bone_unit_vector_1[0]], [0, prev_bone_unit_vector_1[1]])
                # plt.plot([0,prev_bone_unit_vector_2[0]], [0, prev_bone_unit_vector_2[1]], label="prev")
                # plt.legend()
                # plt.grid()
                # plt.show()

                bone_1_magnitude_change = np.linalg.norm(bone_vector_1) / np.linalg.norm(prev_bone_vector_1)
                bone_2_magnitude_change = np.linalg.norm(bone_vector_2) / np.linalg.norm(prev_bone_vector_2)

                one_frame_bone_movements.append([angle_diff, bone_1_magnitude_change, bone_2_magnitude_change])

            one_person_bone_movements.append(one_frame_bone_movements)

        if one_person_bone_movements == []: continue

        one_person_bone_movements = np.rot90(np.asarray(one_person_bone_movements))
        bone_movements[person] = one_person_bone_movements.tolist()
        
        # print(one_person_bone_movements)
        # input()
        # plt.imshow(one_person_bone_movements)
        # plt.show()

    # export the bone movements to a json (not sure if a more efficient data type)
    with open(f"{EXPORT_DIR}/{i[:-9]}.json", 'w') as outfile:
        json.dump(bone_movements, outfile)
    
    pbar.update(1)
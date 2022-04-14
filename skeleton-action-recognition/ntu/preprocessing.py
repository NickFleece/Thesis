import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import json

from ntu_skeleton_config import BONE_CONNECTIONS, IGNORE_FILES

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
                for bone in BONE_CONNECTIONS:
                    j1 = skeleton_joints[bone[0]-1]
                    j2 = skeleton_joints[bone[1]-1]
                    skeleton_bones.append([j1, j2])

                people_bones[person].append(skeleton_bones)

    bone_movements = {}

    for person in people_bones.keys():
        person_bones = people_bones[person]

        one_person_bone_movements = []

        for frame_index in range(1, len(person_bones)):

            one_frame_bone_movements = []

            plt.figure()
            plt.xlim((-5,5))
            plt.ylim((-5, 5))
            for bone, prev_bone in zip(person_bones[frame_index], person_bones[frame_index - 1]):

                #set j1 and j2 to be the current bone
                j1 = bone[0]
                j2 = bone[1]
                bone_vector = [j2[0] - j1[0], j2[1] - j1[1]]
                
                #now change it to the previous bone
                j1 = prev_bone[0]
                j2 = prev_bone[1]
                prev_bone_vector = [j2[0] - j1[0], j2[1] - j1[1]]

                plt.plot([bone[0][0], bone[1][0]],[bone[0][1], bone[1][1]])
                # plt.plot([0,prev_bone_vector[0]],[0,prev_bone_vector[1]])

                # if one of the bones is at 0,0 it causes nan and inf issues, for now just put 0,0 and move on
                if np.linalg.norm(bone_vector) == 0 or np.linalg.norm(prev_bone_vector) == 0:
                    one_frame_bone_movements.append([0,0])

                else:
                    #calculate the magnitude difference
                    #taking the ratio increase from the previous bone to this one
                    #minus one to give a reduction in size a negative value
                    magnitude_change = (np.linalg.norm(bone_vector) / np.linalg.norm(prev_bone_vector)) - 1

                    #now calculating the angle between the two vectors
                    prev_bone_unit_vector = prev_bone_vector / np.linalg.norm(prev_bone_vector)
                    bone_unit_vector = bone_vector / np.linalg.norm(bone_vector)

                    prev_bone_angle = np.arctan2(*prev_bone_unit_vector[::-1])
                    bone_angle = np.arctan2(*bone_unit_vector[::-1])
                    angle = (bone_angle - prev_bone_angle) % (2 * np.pi)
                    if angle > np.pi:
                        angle = angle - (2 * np.pi)
                    angle = angle / np.pi

                    one_frame_bone_movements.append([angle, magnitude_change]) #note: for visualization, can add a 0 here to the last channel to fill rgb

            one_person_bone_movements.append(one_frame_bone_movements)

            plt.show()
            break

        if one_person_bone_movements == []: continue

        one_person_bone_movements = np.rot90(np.asarray(one_person_bone_movements))
        bone_movements[person] = one_person_bone_movements.tolist()

    # export the bone movements to a json (not sure if a more efficient data type)
    with open(f"{EXPORT_DIR}/{i[:-9]}.json", 'w') as outfile:
        json.dump(bone_movements, outfile)
    
    pbar.update(1)
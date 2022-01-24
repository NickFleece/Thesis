import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from ntu_skeleton_config import BONE_CONNECTIONS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir

SKELETON_FILES_DIR = f"{BASE_DIR}/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"
VIDEO_FILES_DIR = f"{BASE_DIR}/ntu_dataset/nturgb+d_rgb"
NEW_EXPORT_DIR = f"{BASE_DIR}/processed_skeleton_data"

from ntu_skeleton_config import BONE_CONNECTIONS

for i in tqdm(os.listdir(SKELETON_FILES_DIR)):

    with open(f"{SKELETON_FILES_DIR}/{i}") as f:
        num_frames = int(f.readline())
        for frame in range(num_frames):

            num_people = int(f.readline())
            
            if frame == 0:
                people_bones = {}
                for person in range(num_people):
                    people_bones[person] = []

            for person in range(num_people):

                random_details = f.readline()
                num_points = int(f.readline())

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

        for person in people_bones.keys():
            person_bones = people_bones[person]
            

        break
    break
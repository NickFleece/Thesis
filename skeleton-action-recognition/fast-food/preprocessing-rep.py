import json
import pandas
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

DRIVE_DIR = args.drive_dir
INPUT_SIZE=256

folders = [
    "825312072753_31.Oct.2018_10.12.10"
]

for folder in folders:

    folder_dir = f"{DRIVE_DIR}/{folder}"

    with open(f"{folder_dir}/825312072753_31.Oct.2018_10.12.10_with_category_id_16July19.json", 'r') as f: #TODO Change this to be more general
        annotations = json.load(f)

    with open(f"{folder_dir}/file_names_825312072753_31.Oct.2018_10.12.10.txt", 'r') as f: #TODO Change this to be more general
        file_names = f.read().splitlines()

    with open(f"{folder_dir}/keypoints.json", 'r') as f:
        keypoints = json.load(f)

    keypoint_annotations = {}
    # {
    #     category : {
    #         category_instance_id : {
    #             file_position_id : image_name:"",
    #             ...
    #         }, ...
    #     }, ...
    # }

    for file in tqdm(annotations.keys()):
        for a in annotations[file]['annotations']:
            if a['category'] != '':

                if a['category'] not in keypoint_annotations: keypoint_annotations[a['category']] = {}
                if a['category_instance_id'] not in keypoint_annotations[a['category']]: keypoint_annotations[a['category']][a['category_instance_id']] = {}

                keypoint_annotations[a['category']][a['category_instance_id']][file_names.index(file)] = file

    for category in keypoint_annotations.keys():

        for instance_id in keypoint_annotations[category].keys():

            file_positions = keypoint_annotations[category][instance_id].keys()
            print(category)
            print(instance_id)
            print(list(file_positions))
            input()
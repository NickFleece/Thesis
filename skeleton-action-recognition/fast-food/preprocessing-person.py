import json
import pandas as pd
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from config import BONE_CONNECTIONS
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--overwrite_existing', default=0)
args = parser.parse_args()

DRIVE_DIR = args.drive_dir
OVERWRITE_EXISTING = args.overwrite_existing
INPUT_SIZE=256

folders = [
    "825312072753_31.Oct.2018_10.12.10",
    "825312072753_31.Oct.2018_13.22.27",
    "825312073198_31.Oct.2018_10.11.56",
    "825312073198_31.Oct.2018_13.22.19"
]

annotation_files = [
    "825312072753_31.Oct.2018_10.12.10_with_category_id_16July19.json",
    "825312072753_31.Oct.2018_13.22.27_with_categories_ids_14_Aug_2019.json",
    "825312073198_31.Oct.2018_10.11.56_with_category_tracks_16July19.json",
    "825312073198_31.Oct.2018_13.22.19_with_categories_ids_14_Aug_2019.json"
]

data_summary = []

for folder, annotation_file in zip(folders, annotation_files):

    folder_dir = f"{DRIVE_DIR}/{folder}"

    if not os.path.exists(f"{folder_dir}/cropped_people"):
        os.mkdir(f"{folder_dir}/cropped_people")

    with open(f"{folder_dir}/{annotation_file}", 'r') as f: #TODO Change this to be more general
        annotations = json.load(f)

    person_annotations = {}

    print(folder)
    pbar = tqdm(total=len(list(annotations.keys())))
    for file in list(annotations.keys()):

        for a in annotations[file]['annotations']:

            if a['category_instance_id'] is None:
                a['category_instance_id'] = "None"

            pbar.set_description(f"{file}")

            if not os.path.exists(f"{folder_dir}/color/{file}"): 
                # print(f"Skipping file: {file}")
                continue

            a['category'] = str(a['category'])
            a['category_instance_id'] = str(a['category_instance_id'])
            a['id'] = str(a['id'])

            if a['category'] not in person_annotations: person_annotations[a['category']] = {}
            if a['category_instance_id'] not in person_annotations[a['category']]: person_annotations[a['category']][a['category_instance_id']] = {}
            if a['id'] not in person_annotations[a['category']][a['category_instance_id']]: person_annotations[a['category']][a['category_instance_id']][a['id']] = []

            # if os.path.exists(f"{folder_dir}/extracted_pose/{a['category']}~{a['category_instance_id']}~{a['id']}~{file}.json") and OVERWRITE_EXISTING == 0:
            #     # print(f"File already exists: {a['id']} - {file}")
            #     with open(f"{folder_dir}/extracted_pose/{a['category']}~{a['category_instance_id']}~{a['id']}~{file}.json") as f:
            #         bone_angle_annotations[a['category']][a['category_instance_id']][a['id']][file] = json.load(f)
            #     continue

            frame = Image.open(f"{folder_dir}/color/{file}")
            frame_shape = np.asarray(frame).shape

            x1 = a['x']
            y1 = a['y']
            w = a['width']
            h = a['height']

            if w is None or h is None:
                print(f"Null height/width: {file}")
                continue

            x1 -= 40
            y1 -= 40
            w += 80
            h += 80

            if x1 < 0:
                w = w + x1
                x1 = 0
            if y1 < 0:
                h = h + y1
                y1 = 0

            if x1 + w > frame_shape[1]:
                w = frame_shape[1] - x1
            if y1 + h > frame_shape[0]:
                h = frame_shape[0] - y1

            frame = np.asarray(frame.crop((x1, y1, x1+w, y1+h)))

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            vertical_pad = max(frame_width-frame_height,0)
            horizontal_pad = max(frame_height-frame_width,0)

            frame = cv2.copyMakeBorder(frame, 0, vertical_pad, 0, horizontal_pad, cv2.BORDER_CONSTANT, 0)
            resized_frame = cv2.resize(frame, dsize=(INPUT_SIZE,INPUT_SIZE))

            plt.imshow(resized_frame)
            plt.show()

            person_annotations[a['category']][a['category_instance_id']][a['id']].append(resized_frame)

            # if not os.path.exists(f"{folder_dir}/cropped_people/{a['category']}~{a['category_instance_id']}~{a['id']}"):
            #     os.mkdir(f"{folder_dir}/cropped_people/{a['category']}~{a['category_instance_id']}~{a['id']}")

        pbar.update(1)

    for category in person_annotations.keys():

        for instance_id in person_annotations[category].keys():

            for person_id in person_annotations[category][instance_id].keys():

                # if not os.path.exists(f"{folder_dir}/cropped_people/{category}~{instance_id}~{person_id}"):
                #     os.mkdir(f"{folder_dir}/cropped_people/{category}~{instance_id}~{person_id}")
                
                with open(f"{folder_dir}/cropped_people/{category}~{instance_id}~{person_id}.json", 'w') as f:
                    json.dump(person_annotations[category][instance_id][person_id])

                data_summary.append({
                    "category":category,
                    "instance_id":instance_id,
                    "folder":folder,
                    "person_id":person_id
                })

pd.DataFrame(data=data_summary).to_csv(f"{DRIVE_DIR}/person_data_summary.csv", index_label=False)
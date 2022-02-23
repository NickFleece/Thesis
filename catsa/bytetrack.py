from genericpath import exists
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.animation as animation
import pandas as pd
from tqdm import tqdm
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
args = parser.parse_args()

def load_annotations(annotation_row):

    id = annotation_row['id']
    frame_start = int(annotation_row['start_frame'][:-4])
    frame_end = int(annotation_row['end_frame'][:-4])

    path_arr = annotation_row['path_to_video_segment'].split('/')
    date = path_arr[0]
    file = f"{path_arr[1]}.txt"

    with open(f"{args.drive_dir}/ByteTrack_Annotations/{date}/{file}") as f:
        annotation_data = f.readlines()
        
    annotation_separated = {}
    for frame_annotation in annotation_data:
        frame_annotation = frame_annotation.split(',')

        frame = int(frame_annotation[0]) - 1
        person_id = int(frame_annotation[1])

        if not (frame >= frame_start and frame <= frame_end): continue

        x1 = float(frame_annotation[2])
        y1 = float(frame_annotation[3])
        w = float(frame_annotation[4])
        h = float(frame_annotation[5])

        if x1 < 0:
            w = w + x1
            x1 = 0
        if y1 < 0:
            h = h + y1
            y1 = 0

        frame_img = Image.open(f"{args.drive_dir}/CATSA_FRAMES/{date}/{file[:-4]}/{str(frame).zfill(5)}.jpg")
        
        frame_shape = np.asarray(frame_img).shape
        if x1 + w > frame_shape[1]:
            w = frame_shape[1] - x1
        if y1 + h > frame_shape[0]:
            h = frame_shape[0] - y1
        
        frame_img = frame_img.crop((x1, y1, x1+w, y1+h))

        # plt.imshow(frame_img)
        # plt.show()
        
        if not person_id in annotation_separated.keys(): annotation_separated[person_id] = []
        annotation_separated[person_id].append(np.asarray(frame_img))

    for person_id in annotation_separated.keys():
        max_x = 0
        max_y = 0

        for frame in annotation_separated[person_id]:
            max_x = max(frame.shape[1], max_x)
            max_y = max(frame.shape[0], max_y)
        
        padded_annotations = []
        for frame in annotation_separated[person_id]:
            padded_frame = np.pad(frame, ((0,max_y - frame.shape[0]),(0,max_x - frame.shape[1]),(0,0)))
            padded_annotations.append(padded_frame)
        annotation_separated[person_id] = padded_annotations

    # images = []
    # fig = plt.figure()
    # for im in annotation_separated[1]:
    #     images.append([plt.imshow(im, animated=True)])
    # ani = animation.ArtistAnimation(fig, images, interval=1000)
    # plt.show()

    # print(type(annotation_separated))

    os.makedirs(f"{args.drive_dir}/ByteTrack_Frames/{id}", exist_ok=True)
    for person in annotation_separated.keys():
        os.makedirs(f"{args.drive_dir}/ByteTrack_Frames/{id}/{person}", exist_ok=True)

        frame_count = 0
        for frame in annotation_separated[person]:
            Image.fromarray(frame).save(f"{args.drive_dir}/ByteTrack_Frames/{id}/{person}/{str(frame_count).zfill(5)}.jpg")
            frame_count += 1

    return annotation_separated

annotation_file = pd.read_csv(f"{args.drive_dir}/CATSA_FRAMES/annotations.csv")
pbar = tqdm(total=len(annotation_file))
for _, annotation_row in annotation_file.iterrows():
    print(annotation_row['path_to_video_segment'])
    load_annotations(annotation_row)
    pbar.update(1)
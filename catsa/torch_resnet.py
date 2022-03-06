# HYPERPARAMETERS FOR NETWORK
LEARNING_RATE = 1e-5
EPOCHS = 10
IMAGE_RESHAPE_SIZE = 320
BATCH_SIZE = 2

import os
import pandas as pd
import argparse
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torchvision.models.video import r3d_18
import torch.nn as nn
from torch.nn import functional as F
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir
BYTETRACK_FRAMES_DIR = f"{BASE_DIR}/ByteTrack_Frames"
VERSION = args.version

annotation_csv = pd.read_csv(f"{BASE_DIR}/annotations.csv")

annotations = {}
for _, row in annotation_csv.iterrows():
    annotation_path = row['path_to_video_segment']
    annotation_class = row['activity_class_id']

    if not annotation_path in annotations.keys(): 
        annotations[annotation_path] = {}
    
    if not annotation_class in annotations[annotation_path].keys():
        annotations[annotation_path][annotation_class] = []
    
    annotations[annotation_path][annotation_class].append(row['id'])

new_annotations = pd.DataFrame()
for file in annotations.keys():
    for c in annotations[file].keys():
        new_annotations = pd.concat([new_annotations, pd.DataFrame({
            "activity_class_id":[c],
            "annotation_ids": [annotations[file][c]],
            "camera":[annotation_csv.iloc[annotations[file][c][0]]['camera']],
            "activity_class_name":[annotation_csv.iloc[annotations[file][c][0]]['activity_class_name']]
        })])
annotations = new_annotations

def get_frames(annotation):

    all_frames = []
    for id in annotation['annotation_ids']:

        person_dir = f"{BYTETRACK_FRAMES_DIR}/{id}"
        for person in os.listdir(person_dir):

            person_frames = []
            frames_dir = f"{person_dir}/{person}"
            for frame in os.listdir(frames_dir):
                frame_arr = np.asarray(iio.imread(f"{frames_dir}/{frame}"))

                frame_arr = frame_arr / np.max(frame_arr)
                
                # square and reshape image
                max_dim = max(frame_arr.shape[0], frame_arr.shape[1])
                frame_arr = np.pad(frame_arr, ((0, max_dim - frame_arr.shape[0]), (0, max_dim - frame_arr.shape[1]), (0,0)))
                frame_arr = cv2.resize(frame_arr, (IMAGE_RESHAPE_SIZE,IMAGE_RESHAPE_SIZE))

                person_frames.append(frame_arr)
            
            person_frames = np.asarray(person_frames)
            new_person_frames = []
            for i in range(person_frames.shape[3]):
                new_person_frames.append(person_frames[:,:,:,i])
            person_frames = np.asarray(new_person_frames)

            person_frames = torch.tensor([person_frames], dtype=torch.float32)
            all_frames.append(person_frames)
    
    return all_frames

class VideoRecognitionModel(nn.Module):

    def __init__(self):
        super(VideoRecognitionModel, self).__init__()

        self.pretrained_model = nn.Sequential(*list(r3d_18(pretrained=True, progress=True).children())[:-1])
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 5)

        return None

    def forward(self, x):

        x = self.pretrained_model(x)
        print(x.shape)

        return x

model = VideoRecognitionModel()

#training loop
for e in range(EPOCHS):
    print(f"Epoch {e}")

    for _, sample in annotations.iterrows():

        for person in get_frames(sample):
            
            model(person)

        break
    break
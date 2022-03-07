# HYPERPARAMETERS FOR NETWORK
LEARNING_RATE = 1e-5
EPOCHS = 10
IMAGE_RESHAPE_SIZE = 112
BATCH_SIZE = 2
FRAME_SUBSAMPLING = 2
CLIP_LENGTH = 128

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
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

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

    # no person detected in annotation
    if len(os.listdir(f"{BYTETRACK_FRAMES_DIR}/{row['id']}")) == 0:
        # print(f"No person detected: {row}")
        continue

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
            for frame in np.asarray(os.listdir(frames_dir))[::FRAME_SUBSAMPLING]:
                frame_arr = np.asarray(iio.imread(f"{frames_dir}/{frame}"))

                frame_arr = frame_arr / np.max(frame_arr)
                
                # square and reshape image
                max_dim = max(frame_arr.shape[0], frame_arr.shape[1])
                frame_arr = np.pad(frame_arr, ((0, max_dim - frame_arr.shape[0]), (0, max_dim - frame_arr.shape[1]), (0,0)))
                frame_arr = cv2.resize(frame_arr, (IMAGE_RESHAPE_SIZE,IMAGE_RESHAPE_SIZE))

                person_frames.append(frame_arr)

            # person_frames = np.asarray(person_frames)
            

            # print(person_frames.shape)

            frame_count = 0
            cycle = 0
            while frame_count < len(person_frames):
                
                new_person_frames = []
                for i in range(cycle * CLIP_LENGTH, CLIP_LENGTH + cycle * CLIP_LENGTH):
                    
                    if frame_count + i > len(person_frames):
                        new_person_frames.append(np.zeros((IMAGE_RESHAPE_SIZE, IMAGE_RESHAPE_SIZE, 3)))
                    else:
                        new_person_frames.append(person_frames[i])

                    frame_count += 1
                
                new_person_frames = np.asarray(new_person_frames)

                channel_first_person_frames = []
                for i in range(new_person_frames.shape[3]):
                    channel_first_person_frames.append(new_person_frames[:,:,:,i])

                all_frames.append(channel_first_person_frames)
                
                cycle += 1

    return torch.tensor(all_frames, dtype=torch.float32)

# get_frames(annotations.iloc[0])
# raise Exception()

class VideoRecognitionModel(nn.Module):

    def __init__(self):
        super(VideoRecognitionModel, self).__init__()

        self.pretrained_model = nn.Sequential(*list(r3d_18(pretrained=True, progress=True).children())[:-1])
        
        self.fc1 = nn.Linear(512, 512)

        self.rnn = nn.RNN(512, 50, batch_first=True)

        self.fc2 = nn.Linear(50, 7)

    def forward(self, x):

        # ensure the pretrained model is frozen
        self.pretrained_model.requires_grad_ = False

        x = self.pretrained_model(x).squeeze()

        # if a batch of size 1 was put through, ensure that the batch is preserved
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        x = self.fc1(x)
        x = F.relu(x)

        x = x.unsqueeze(dim=0)

        # outputs = []

        # for x in sample_input:
        #     print(x.shape)

        #     # pass input through pretrained resnet module
        #     x = self.pretrained_model(x).squeeze()

        #     # if a batch of size 1 was put through, ensure that the batch is preserved
        #     if len(x.shape) == 1:
        #         x = x.unsqueeze(dim=0)
            
        #     # our fc layers we are training
        #     x = self.fc1(x)
        #     x = F.relu(x)

        #     outputs.append(x)

        # x = torch.cat(outputs)

        # x = x.unsqueeze(dim=0)

        # pass through rnn to generate final output
        x, _ = self.rnn(x)
        x = x[:,-1]

        x = self.fc2(x)
        x = F.softmax(x)

        print(x.shape)

        return x

model = VideoRecognitionModel()
model.to(device)
if device != cpu:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#training loop
for e in range(EPOCHS):
    print(f"Epoch {e}")

    batch_samples = []
    batch_actual = []

    # shuffle annotations
    # annotations = annotations.sample(frac=1)

    pbar = tqdm(total=len(annotations))
    for _, sample in annotations.iterrows():    
        # print(sample)

        batch_samples.append(get_frames(sample))
        batch_actual.append(sample['activity_class_id'])

        if len(batch_samples) == BATCH_SIZE:

            print(len(batch_samples))

            model_out = []
            for sample in batch_samples:
                model_out.append(model(sample.to(device)))
            model_out = torch.cat(model_out)

            print(model_out.shape)
            print(model_out)

            loss = criterion(
                model_out,
                torch.tensor(batch_actual).to(device).long()
            )
            loss.backward()
            optimizer.step()

            # print(loss.item())

            batch_samples = []
            batch_actual = []

            optimizer.zero_grad()
        
        pbar.update(1)

    break

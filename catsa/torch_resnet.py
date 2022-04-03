# HYPERPARAMETERS FOR NETWORK
LEARNING_RATE = 3e-5
EPOCHS = 100
IMAGE_RESHAPE_SIZE = 80
BATCH_SIZE = 1
FRAME_SUBSAMPLING = 4
FLIP_PROB = 0.5

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
from sklearn.model_selection import train_test_split
import time
import random
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
MODEL_SAVE_DIR = f"{BASE_DIR}/catsa_models"
VERSION = args.version

if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

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

# Filter on camera = FP
annotations = annotations.loc[annotations["camera"] == "FD"]

used_labels = [
    0,
    # 1, #not used, 79 samples
    2,
    3,
    # 4, #not used, 41 samples
    5,
    6 #not used, 88 samples
]

def splitDataset(data):

    train_output = pd.DataFrame()
    test_output = pd.DataFrame()
    for label in used_labels:
        label_train, label_test = train_test_split(data.loc[data["activity_class_id"] == label], test_size=0.2)

        train_output = pd.concat([train_output, label_train])
        test_output = pd.concat([test_output, label_test])

    return train_output, test_output

train, test = splitDataset(annotations)

def subsampleDataset(data):

    #shuffle data
    data = data.sample(frac=1)

    min_samples = None
    label_value_counts = data['activity_class_id'].value_counts()
    for label in used_labels:
        if min_samples == None:
            min_samples = label_value_counts[label]
        else:
            min_samples = min(min_samples, label_value_counts[label])
    
    subsampled_data = pd.DataFrame()
    for label in used_labels:
        label_data = data.loc[data['activity_class_id'] == label]
        label_data = label_data.head(min_samples)
        subsampled_data = pd.concat([subsampled_data, label_data])

    subsampled_data = subsampled_data.sample(frac=1)
    return subsampled_data

def getFrames(annotation):

    all_frames = []
    annotation_ids = annotation['annotation_ids']
    random.shuffle(annotation_ids)

    for id in annotation_ids:

        person_dir = f"{BYTETRACK_FRAMES_DIR}/{id}"
        person_files = os.listdir(person_dir)
        random.shuffle(person_files)

        for person in person_files:

            person_frames = []
            frames_dir = f"{person_dir}/{person}"

            flip_video = random.random() < FLIP_PROB

            for frame in np.asarray(os.listdir(frames_dir))[::FRAME_SUBSAMPLING]:
                frame_arr = np.asarray(iio.imread(f"{frames_dir}/{frame}"))

                if flip_video:
                    frame_arr = np.fliplr(frame_arr)

                frame_arr = frame_arr / np.max(frame_arr)
                
                # square and reshape image
                max_dim = max(frame_arr.shape[0], frame_arr.shape[1])
                frame_arr = np.pad(frame_arr, ((0, max_dim - frame_arr.shape[0]), (0, max_dim - frame_arr.shape[1]), (0,0)))
                frame_arr = cv2.resize(frame_arr, (IMAGE_RESHAPE_SIZE,IMAGE_RESHAPE_SIZE))

                # person_frames.append(frame_arr)
                all_frames.append(frame_arr)

    all_frames = np.asarray(all_frames)
    channel_first_person_frames = []
    for i in range(all_frames.shape[3]):
        channel_first_person_frames.append(all_frames[:,:,:,i])

    return torch.tensor(channel_first_person_frames, dtype=torch.float32).to(device)

class VideoRecognitionModel(nn.Module):

    def __init__(self):
        super(VideoRecognitionModel, self).__init__()

        self.pretrained_model = nn.Sequential(*list(r3d_18(pretrained=True, progress=True).children())[:-1])
        
        self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, len(used_labels))

    def forward(self, x):

        # ensure the pretrained model is frozen
        self.pretrained_model.requires_grad_ = False

        x = self.pretrained_model(x).squeeze()

        # if a batch of size 1 was put through, ensure that the batch is preserved
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        x = F.dropout(x, 0.5)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

model = VideoRecognitionModel()
model.to(device)
# model = nn.DataParallel(model)

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

    # get the subsampled data
    subsampled_train = subsampleDataset(train)

    losses = []
    train_correct = 0
    train_total = 0

    pbar = tqdm(total=len(subsampled_train))
    for _, sample in subsampled_train.iterrows():

        batch_samples.append(getFrames(sample))
        batch_actual.append(used_labels.index(sample['activity_class_id']))

        if len(batch_samples) == BATCH_SIZE:

            max_len = 0
            for sample in batch_samples:
                max_len = max(max_len, sample.shape[1])
            
            batch_input = []
            for sample in batch_samples:
                batch_input.append(F.pad(sample, (0,0,0,0,0,max_len - sample.shape[1])).unsqueeze(dim=0))
            batch_input = torch.cat(batch_input)

            model_out = model(batch_input)

            for output, label in zip(model_out.argmax(dim=1).cpu().detach().numpy(), batch_actual):
                if output == label:
                    train_correct += 1
                train_total += 1

            loss = criterion(
                model_out,
                torch.tensor(batch_actual).to(device).long()
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            pbar.set_description(f"{str(sum(losses) / len(losses))} - {(train_correct / train_total) * 100}")

            batch_samples = []
            batch_actual = []

            optimizer.zero_grad()
        
        pbar.update(1)
    
    pbar.close()

    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    with torch.no_grad():

        val_correct = 0
        count = 0

        val_outputs = []
        val_actual = []

        pbar = tqdm(total=len(test))
        for _, sample in test.iterrows():

            sample_frames = getFrames(sample)

            model_out = model(torch.unsqueeze(sample_frames, 0)).argmax(dim=1).item()

            if model_out == used_labels.index(sample['activity_class_id']):
                val_correct += 1
            
            val_outputs.append(model_out)
            val_actual.append(used_labels.index(sample['activity_class_id']))
            
            count += 1
            pbar.set_description(f"{(val_correct / count) * 100}% Validation Correct :)")
            pbar.update(1)
        
        pbar.close()
        time.sleep(1)

        print(f"Epoch {e} Validation Accuracy: {val_correct / len(test)}")

        print(confusion_matrix(val_actual, val_outputs))
    
    print("---------------------------------------------------------------")

    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
        'val_outputs':val_outputs,
        'val_actual':val_actual
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

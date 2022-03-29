IMAGE_RESHAPE_SIZE = 224
FRAME_SUBSAMPLING = 1

import torch
import argparse
from torch import nn
from torch.nn import functional as F
from torchvision.models.video import r3d_18
from sklearn import metrics
import pandas as pd
import numpy as np
from PIL import Image
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', required=True)
parser.add_argument('--model_file', required=True)
parser.add_argument('--bytetrack_annotation_dir', required=True)
parser.add_argument('--annotation_file')
args = parser.parse_args()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Running on: {device}")

used_labels = [
    0,
    # 1, #not used, 79 samples
    2,
    3,
    # 4, #not used, 41 samples
    5,
    # 6 #not used, 88 samples
]

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

        # x = F.dropout(x, 0.5)

        x = self.fc1(x)
        x = F.relu(x)
        # x = F.dropout(x, 0.5)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

model = VideoRecognitionModel()
model.to(device)

model_data = torch.load(args.model_file, map_location=device)

accuracy = metrics.accuracy_score(model_data['val_actual'], model_data['val_outputs'])
recall = metrics.recall_score(model_data['val_actual'], model_data['val_outputs'], average='macro')
precision = metrics.precision_score(model_data['val_actual'], model_data['val_outputs'], average='macro')

print("Metrics:")
print("Acc:" + str(accuracy))
print("Recall:" + str(recall))
print("Precision" + str(precision))
print("\n")

print(model.load_state_dict(model_data['model_state_dict']))
model.eval()

annotation_csv = pd.read_csv(args.annotation_file)

annotations = {}
for _, row in annotation_csv.iterrows():
    annotation_path = row['path_to_video_segment']
    annotation_class = row['activity_class_id']

    if not annotation_path in annotations.keys(): 
        annotations[annotation_path] = []
    
    if not annotation_class in annotations[annotation_path]:
        annotations[annotation_path].append(annotation_class)

new_annotations = pd.DataFrame()
for file in annotations.keys():
    new_annotations = pd.concat([new_annotations, pd.DataFrame({
        "date":[file.split('/')[0]],
        "segment":[file.split('/')[1]],
        "path":[file],
        "activity_class_ids":[annotations[file]],
        "camera":[file.split('/')[1].split('_')[0]],
    })])
annotations = new_annotations

annotations = annotations.sample(frac=1)

for _, annotation in annotations.iterrows():

    #camera restriction
    # if annotation['camera'] != 'FD': continue

    found_unused_labels = False
    for i in annotation['activity_class_ids']:
        if i not in used_labels: found_unused_labels = True
    if found_unused_labels: continue

    bytetrack_annotation_path = f"{args.bytetrack_annotation_dir}/{annotation['date']}/{annotation['segment']}.txt"

    with open(bytetrack_annotation_path) as f:
        annotation_data = f.readlines()

    annotation_separated = {}
    for frame_annotation in annotation_data:
        frame_annotation = frame_annotation.split(',')

        frame = int(frame_annotation[0]) - 1
        person_id = int(frame_annotation[1])

        x1 = float(frame_annotation[2])
        y1 = float(frame_annotation[3])
        w = float(frame_annotation[4])
        h = float(frame_annotation[5])

        x1 -= 40
        w += 80
        h += 40

        if x1 < 0:
            w = w + x1
            x1 = 0
        if y1 < 0:
            h = h + y1
            y1 = 0

        frame_img = Image.open(f"{args.frames_dir}/{annotation['path']}/{str(frame).zfill(5)}.jpg")
        
        frame_shape = np.asarray(frame_img).shape
        if x1 + w > frame_shape[1]:
            w = frame_shape[1] - x1
        if y1 + h > frame_shape[0]:
            h = frame_shape[0] - y1
        
        frame_img = frame_img.crop((x1, y1, x1+w, y1+h))
        
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

    predicted_labels = []
    for person in annotation_separated.keys():

        frames = np.asarray(annotation_separated[person])

        #subsample the frames
        frames = frames[::FRAME_SUBSAMPLING]

        reshaped_frames = []
        for frame in frames:
            
            frame = frame / np.max(frame)

            max_dim = max(frame.shape[0], frame.shape[1])
            frame = np.pad(frame, ((0, max_dim - frame.shape[0]), (0, max_dim - frame.shape[1]), (0,0)))
            frame = cv2.resize(frame, (IMAGE_RESHAPE_SIZE,IMAGE_RESHAPE_SIZE))

            reshaped_frames.append(frame)

        while len(reshaped_frames) <= 100:

            reshaped_frames.append(np.zeros((IMAGE_RESHAPE_SIZE, IMAGE_RESHAPE_SIZE, 3)))

        frames = np.asarray(reshaped_frames)

        channel_first_frames = []
        for i in range(frames.shape[3]):
            channel_first_frames.append(frames[:,:,:,i])
        
        frames = torch.tensor([channel_first_frames], dtype=torch.float32).to(device)

        with torch.no_grad():
            predicted_label = model(frames).argmax(dim=1).item()
            print(model(frames))

        if not used_labels[predicted_label] in predicted_labels:
            predicted_labels.append(used_labels[predicted_label])
    
    actual_labels = annotation['activity_class_ids']
    if not 0 in predicted_labels and 0 in actual_labels and len(actual_labels) != 1:
        actual_labels.remove(0)

    predicted_labels.sort()
    actual_labels.sort()
    
    print(f"Predicted: {predicted_labels}")
    print(f"Actual: {actual_labels}")
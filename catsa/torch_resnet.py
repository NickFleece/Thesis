# Hyperparameters, these are what need to be tuned most of the time
LEARNING_RATE = 3e-5
EPOCHS = 100
IMAGE_RESHAPE_SIZE = 80
BATCH_SIZE = 1
FRAME_SUBSAMPLING = 4
FLIP_PROB = 0.5

#All of the imports needed, matplotlib not required but useful for imshow on frames
import os
import pandas as pd
import argparse
import imageio as iio
import numpy as np
# import matplotlib.pyplot as plt
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

#Set the device the code will run on
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
print(f"Running on: {device}")

#The arguments that must be passed into the program via the command line
parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--load_checkpoint')
parser.add_argument('--version', required=True)
args = parser.parse_args()

#Loading in some of the arguments
BASE_DIR = args.drive_dir
BYTETRACK_FRAMES_DIR = f"{BASE_DIR}/ByteTrack_Frames"
MODEL_SAVE_DIR = f"{BASE_DIR}/catsa_models"
VERSION = args.version

#Make the model folder if it doesn't exist already
if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

#Read in the annotation csv
annotation_csv = pd.read_csv(f"{BASE_DIR}/annotations.csv")

#This is some preprocessing that needs to be done to the annotations
#Basically just combining different segments from the same car, i.e. if the underside clip got split, we treat it as one complete annotation
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

#Uncomment these if you want to train on only one camera angle, can also be replaced with R or RP if required later on
#annotations = annotations.loc[annotations["camera"] == "FD"]
#annotations = annotations.loc[annotations["camera"] == "FP"]

#Just printing some counts for ease of use
print(annotations["activity_class_name"].value_counts())

#This defines the labels that will be used by the model, comment any labels that you don't want used
used_labels = [
    0, # Background
    # 1, #not used - Roof Top Sides
    2, #Underside
    3, #Hood
    # 4, #not used - Trunk
    5, #Front Passenger Compartment
    6 #Rear Passenger Compartment
]

#Function to split the data into train and test, right now it uses 80/20, but can be modified by changing the test_size below from 0.2 to the desired size
def splitDataset(data):

    train_output = pd.DataFrame()
    test_output = pd.DataFrame()
    for label in used_labels:
        label_train, label_test = train_test_split(data.loc[data["activity_class_id"] == label], test_size=0.2)

        train_output = pd.concat([train_output, label_train])
        test_output = pd.concat([test_output, label_test])

    return train_output, test_output
 
#Perform the actual split
train, test = splitDataset(annotations)

#Subsample the data at every epoch
#Because we use batch size 1, it is important to subsample the dataset at every training epoch to avoid biases
def subsampleDataset(data):

    #shuffle data
    data = data.sample(frac=1)

    #Find the minimum number of samples used by a class
    min_samples = None
    label_value_counts = data['activity_class_id'].value_counts()
    for label in used_labels:
        if min_samples == None:
            min_samples = label_value_counts[label]
        else:
            min_samples = min(min_samples, label_value_counts[label])
    
    #Sample each label to use the minimum number of examples
    #So we have an equal number of samples for each class
    subsampled_data = pd.DataFrame()
    for label in used_labels:
        label_data = data.loc[data['activity_class_id'] == label]
        label_data = label_data.head(min_samples)
        subsampled_data = pd.concat([subsampled_data, label_data])

    #Shuffle the data once more before return
    subsampled_data = subsampled_data.sample(frac=1)
    
    return subsampled_data

#Function to retrieve the actual frames
def getFrames(annotation):

    all_frames = []
    annotation_ids = annotation['annotation_ids']

    #Shuffling to confuse the model
    random.shuffle(annotation_ids)

    for id in annotation_ids:

        #Grab the directory that contains all of the people within the annotation
        person_dir = f"{BYTETRACK_FRAMES_DIR}/{id}"
        person_files = os.listdir(person_dir)
        
        #More shuffling to try and confuse the model
        random.shuffle(person_files)

        for person in person_files:

            frames_dir = f"{person_dir}/{person}"

            #This just sets whether we will flip a particular person
            flip_video = random.random() < FLIP_PROB

            for frame in np.asarray(os.listdir(frames_dir))[::FRAME_SUBSAMPLING]:

                #Read frame
                frame_arr = np.asarray(iio.imread(f"{frames_dir}/{frame}"))

                #Flip if needed
                if flip_video:
                    frame_arr = np.fliplr(frame_arr)

                #Normalize
                frame_arr = frame_arr / np.max(frame_arr)
                
                #Square and reshape image
                max_dim = max(frame_arr.shape[0], frame_arr.shape[1])
                frame_arr = np.pad(frame_arr, ((0, max_dim - frame_arr.shape[0]), (0, max_dim - frame_arr.shape[1]), (0,0)))
                frame_arr = cv2.resize(frame_arr, (IMAGE_RESHAPE_SIZE,IMAGE_RESHAPE_SIZE))

                all_frames.append(frame_arr)

    all_frames = np.asarray(all_frames)

    #This is just because the model we use takes the frames in with the channel dimension first, so we have to reshape
    channel_first_person_frames = []
    for i in range(all_frames.shape[3]):
        channel_first_person_frames.append(all_frames[:,:,:,i])

    return torch.tensor(channel_first_person_frames, dtype=torch.float32).to(device)

#The model itself!
class VideoRecognitionModel(nn.Module):

    def __init__(self):
        super(VideoRecognitionModel, self).__init__()

        #The big part of the model
        #Pretrained Resnet 3D, provided by pytorch, you can see the documentation through their packages
        self.pretrained_model = nn.Sequential(*list(r3d_18(pretrained=True, progress=True).children())[:-1])
        
        #Our part we're training, super simple nothing fancy, two fully connected layers
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, len(used_labels))

    def forward(self, x):

        #Ensure the pretrained model is frozen
        #Training this model could be useful if you want, but may require a lot of tuning and adjustments
        self.pretrained_model.requires_grad_ = False

        #Squeeze to remove some unneeded dimensions
        x = self.pretrained_model(x).squeeze()

        # if a batch of size 1 was put through, ensure that the batch is preserved
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        #Dropout
        x = F.dropout(x, 0.5)

        #The first fully connected layer, followed by relu and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)

        #The output layer
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

#Create the model
model = VideoRecognitionModel()
#Send the model to the device (should be GPU)
model.to(device)
#This can be used to parallelize across multiple gpu's if needed to increase batch size
# model = nn.DataParallel(model)

#Create the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

#Training loop
for e in range(EPOCHS):
    print(f"Epoch {e}")

    batch_samples = []
    batch_actual = []

    # get the subsampled data
    subsampled_train = subsampleDataset(train)

    #Holders for losses and accuracies
    losses = []
    train_correct = 0
    train_total = 0

    #Progress bar
    pbar = tqdm(total=len(subsampled_train))
    for _, sample in subsampled_train.iterrows():

        #Get the samples and labels
        batch_samples.append(getFrames(sample))
        batch_actual.append(used_labels.index(sample['activity_class_id']))

        if len(batch_samples) == BATCH_SIZE:

            #Make all samples in the batch the same length so they can be put through the model
            max_len = 0
            for sample in batch_samples:
                max_len = max(max_len, sample.shape[1])
            batch_input = []
            for sample in batch_samples:
                batch_input.append(F.pad(sample, (0,0,0,0,0,max_len - sample.shape[1])).unsqueeze(dim=0))
            batch_input = torch.cat(batch_input)

            #Pass the batch through the model
            model_out = model(batch_input)

            #Log whether the model predicted correctly
            for output, label in zip(model_out.argmax(dim=1).cpu().detach().numpy(), batch_actual):
                if output == label:
                    train_correct += 1
                train_total += 1

            #Calculate Loss & Gradient Descent
            loss = criterion(
                model_out,
                torch.tensor(batch_actual).to(device).long()
            )
            loss.backward()
            optimizer.step()

            #Append the loss so that we can log it later
            losses.append(loss.item())

            #Update the progress bar with the avg loss and accuracy
            pbar.set_description(f"{str(sum(losses) / len(losses))} - {(train_correct / train_total) * 100}")

            batch_samples = []
            batch_actual = []

            #Zero the gradients
            optimizer.zero_grad()
        
        #Take a step on the progress bar
        pbar.update(1)

    #Close the progress bar or you'll get some weird display errors    
    pbar.close()

    print(f"Epoch {e} Loss: {sum(losses) / len(losses)}, Accuracy: {train_correct / train_total}")

    #Don't do gradients for the testing run
    with torch.no_grad():

        #Same here, just logging
        val_correct = 0
        count = 0

        val_outputs = []
        val_actual = []

        pbar = tqdm(total=len(test))
        for _, sample in test.iterrows():

            #Similar idea, sample the frames
            sample_frames = getFrames(sample)

            #Get the model output, this time we can just grab the predicted class
            model_out = model(torch.unsqueeze(sample_frames, 0)).argmax(dim=1).item()

            #Do some output storage and set the progress bar description
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

        #This is just outputting the confusion matrix since the accuracy score can be a bit misleading sometimes
        print(confusion_matrix(val_actual, val_outputs))
    
    print("---------------------------------------------------------------")

    #For each epoch we save the model as well as some other values for plotting if we wish, but the main thing is the model_state_dict
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
        'val_outputs':val_outputs,
        'val_actual':val_actual
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

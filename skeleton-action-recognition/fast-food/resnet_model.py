# Hyperparameters, these are what need to be tuned most of the time
LEARNING_RATE = 0.1
EPOCHS = 500
IMAGE_RESHAPE_SIZE = 112
BATCH_SIZE = 2
FRAME_SUBSAMPLING = 1
FLIP_PROB = 0.5
RANDOM_STATE = 42

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
import pickle
from sklearn.utils import shuffle

#Set the device the code will run on
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
MODEL_SAVE_DIR = f"{BASE_DIR}/models"
VERSION = args.version

data_summary = pd.read_csv(f"{BASE_DIR}/person_data_summary.csv")
data_summary = data_summary.fillna("Background")

print(data_summary)

categories = list(data_summary['category'].unique())
categories.remove("Background")
categories.remove(" ")
category_counts = data_summary['category'].value_counts()

print(categories)
print(category_counts)

x = []
y = []

for _, d in data_summary.iterrows():

    if d['category'] == "Background": continue
    if d['category'] == ' ': continue
    if d['instance_id'] == "Background": continue

    y.append(
        categories.index(d['category'])
    )

    x.append(
        f"{d['folder']}/cropped_people/{d['category']}~{d['instance_id']}~{d['person_id']}"
    )

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True)

new_x_train = []
new_y_train = []
flip = []

for x_1, y_1 in zip(X_train, y_train):

    new_x_train.append(x_1)
    new_y_train.append(y_1)
    flip.append(False)

    if y_1 == categories.index("picking_up"): continue

    new_x_train.append(x_1)
    new_y_train.append(y_1)
    flip.append(True)

X_train = new_x_train
y_train = new_y_train

#Make the model folder if it doesn't exist already
if not os.path.isdir(f"{MODEL_SAVE_DIR}/m_{VERSION}"):
    os.mkdir(f"{MODEL_SAVE_DIR}/m_{VERSION}")

#Function to Load Data
def getFrames(path, flip=False):

    with open(f"{BASE_DIR}/{path}.pickle", 'rb') as f:
        all_frames = np.asarray(pickle.load(f))[::FRAME_SUBSAMPLING]

    # print(all_frames.shape)

    if flip:
        flipped_frames = []
        for frame in all_frames:
            flipped_frames.append(np.fliplr(frame))
        all_frames = np.asarray(flipped_frames)

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

        #Ensure the pretrained model is frozen
        #Training this model could be useful if you want, but may require a lot of tuning and adjustments
        self.pretrained_model.requires_grad_ = False
        
        #Our part we're training, super simple nothing fancy, two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(512, len(categories))

    def forward(self, x):

        #Squeeze to remove some unneeded dimensions
        with torch.no_grad():
            x = self.pretrained_model(x).squeeze()

        # if a batch of size 1 was put through, ensure that the batch is preserved
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        #fully connected layers
        x = self.fc(x)

        #The output layer
        x = self.fc_final(x)
        x = F.softmax(x, dim=1)

        return x

#Create the model
model = VideoRecognitionModel()
#Send the model to the device (should be GPU)
model.to(device)
#This can be used to parallelize across multiple gpu's if needed to increase batch size
model = nn.DataParallel(model)

#Create the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    factor=0.1,
    patience=5
)

#Training loop
for e in range(EPOCHS):
    print(f"Epoch {e}")

    batch_samples = []
    batch_actual = []

    #shuffle dataset
    X_train, y_train, flip = shuffle(X_train, y_train, flip, random_state=RANDOM_STATE)

    #Holders for losses and accuracies
    losses = []
    train_correct = 0
    train_total = 0
    train_actual = []
    train_predicted = []

    #Progress bar
    pbar = tqdm(total=len(X_train))
    for X, y, f in zip(X_train, y_train, flip):

        #Get the samples and labels
        batch_samples.append(getFrames(X, f))
        batch_actual.append(y)

        if len(batch_samples) == BATCH_SIZE or (X == X_train[-1] and f == flip[-1]):

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

                train_actual.append(label)
                train_predicted.append(output)

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
    print(confusion_matrix(train_actual, train_predicted))

    #Don't do gradients for the testing run
    with torch.no_grad():

        #Same here, just logging
        val_correct = 0
        count = 0

        val_predicted = []
        val_actual = []

        val_loss = 0

        pbar = tqdm(total=len(X_test))
        for X, y in zip(X_test, y_test):

            #Similar idea, sample the frames
            sample_frames = getFrames(X)

            #Get the model output, this time we can just grab the predicted class
            out = model(torch.unsqueeze(sample_frames, 0))
            pred = out.argmax(dim=1).item()

            val_loss += criterion(out, torch.tensor([y]).to(device).long())

            #Do some output storage and set the progress bar description
            if pred == y:
                val_correct += 1
            
            val_predicted.append(pred)
            val_actual.append(y)
            
            count += 1
            pbar.set_description(f"{(val_correct / count) * 100}% Validation Correct :)")
            pbar.update(1)

        scheduler.step(val_loss/len(X_test))
        
        pbar.close()
        time.sleep(1)

        print(f"Epoch {e} Validation Accuracy: {val_correct / len(X_test)}")

        #This is just outputting the confusion matrix since the accuracy score can be a bit misleading sometimes
        print(confusion_matrix(val_actual, val_predicted))
    
    print("---------------------------------------------------------------")

    #For each epoch we save the model as well as some other values for plotting if we wish, but the main thing is the model_state_dict
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses,
        'val_predicted':val_predicted,
        'val_actual':val_actual
    }, f"{MODEL_SAVE_DIR}/m_{VERSION}/{e}")

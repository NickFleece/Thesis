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
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

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

                person_frames.append(tf.convert_to_tensor(frame_arr, dtype=tf.float32))
            
            all_frames.append(person_frames)
    
    return all_frames

model_url = "https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/3"

encoder = hub.KerasLayer(model_url, trainable=True, name="movinet")

inputs = tf.keras.layers.Input(
    shape=[None, IMAGE_RESHAPE_SIZE, IMAGE_RESHAPE_SIZE, 3],
    dtype=tf.float32,
    name='image')

outputs = encoder(dict(image=inputs))

dense_1 = tf.keras.layers.Dense(600)(outputs)
final_dense = tf.keras.layers.Dense(6)(dense_1)

model = tf.keras.Model(inputs, final_dense, name='catsa-movinet')

print(model.summary())

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy()

for e in range(EPOCHS):

    print(f"\n\nEpoch {e} / {EPOCHS} : {(e / EPOCHS) * 100}")

    #shuffle samples
    annotations = annotations.sample(frac=1)

    pbar = tqdm(total = len(annotations))
    
    batch = []
    batch_labels = []
    for _, annotation in annotations.iterrows():

        pbar.update(1)

        batch.append(get_frames(annotation))
        batch_labels.append(annotation['activity_class_id'])

        if len(batch) == BATCH_SIZE:
            print("TEST")
            with tf.GradientTape() as tape:

                for sample in batch:

                    sample_preds = model([sample[0]])
                    print("WHAT")
                    print("THE HECK")
                    break
    break
import os
import pandas as pd
import argparse
import imageio as iio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

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
                person_frames.append(np.asarray(iio.imread(f"{frames_dir}/{frame}")))
            
            all_frames.append(person_frames)
    
    return all_frames

model_url = "https://tfhub.dev/tensorflow/movinet/a5/base/kinetics-600/classification/3"

encoder = hub.KerasLayer(model_url, trainable=True)

print(type(encoder))

inputs = tf.keras.layers.Input(
    shape=[None, 320, 320, 3],
    dtype=tf.float32,
    name='image')

outputs = encoder(dict(image=inputs))

final_dense = tf.keras.layers.Dense(60)(outputs)

model = tf.keras.Model(inputs, final_dense, name='movinet')

print(model.summary())

example_input = tf.ones([1, 8, 320, 320, 3])
example_output = model(example_input)

print(example_output)

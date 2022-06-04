import json
import pandas
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--model_file', required=True)
args = parser.parse_args()

DRIVE_DIR = args.drive_dir
TF_MODEL_FILE = args.model_file
INPUT_SIZE=256

folders = [
    "825312072753_31.Oct.2018_10.12.10"
]

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE)
interpreter.allocate_tensors()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

for folder in folders:

    image_folder = f"{DRIVE_DIR}/{folder}/color"

    with open(f"{DRIVE_DIR}/{folder}/file_names_{folder}.txt") as f:
        file_names = f.read().splitlines()
    
    with open(f"{DRIVE_DIR}/{folder}/person_annotations.txt") as f:
        annotations = f.read().splitlines()

    for a in tqdm(annotations):
        frame_annotation = a.split(',')
        image = frame_annotation[-1].split('/')[-1]
        
        if not os.path.exists(f"{image_folder}/{image}"): continue

        frame = Image.open(f"{image_folder}/{image}")
        frame_shape = np.asarray(frame).shape

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

        if x1 + w > frame_shape[1]:
            w = frame_shape[1] - x1
        if y1 + h > frame_shape[0]:
            h = frame_shape[0] - y1

        frame = np.asarray(frame.crop((x1, y1, x1+w, y1+h)))

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        vertical_pad = max(frame_width-frame_height,0)
        horizontal_pad = max(frame_height-frame_width,0)

        frame = cv2.copyMakeBorder(frame, vertical_pad//2, vertical_pad//2, horizontal_pad//2, horizontal_pad//2, cv2.BORDER_CONSTANT, 0)
        resized_frame = tf.image.resize_with_pad([frame], INPUT_SIZE, INPUT_SIZE)

        keypoints = movenet(resized_frame)[0][0]

        plt.imshow(frame)
        for keypoint in keypoints:
            if keypoint[2] < 0.2: continue
            print(keypoint)
            print(frame.shape)
            plt.scatter([keypoint[1]*frame.shape[0]],[keypoint[0]*frame.shape[1]],c='green',s=10)
        plt.show()
        
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio

INPUT_SIZE=256

# base_path = "/media/nick/External Boi/CATSA"
base_path = "H:/CATSA"
model_file = f"{base_path}/pose_model.tflite"
video_file = f"{base_path}/CATSA_DATA/FP(2668)_y2021m07d28h10m00s00.wmv"

# get the frame I'll be running the demo on
cap = cv2.VideoCapture(video_file)
for _ in range(12375):
    _, frame = cap.read()

gif_frames = []
for _ in range(100):
    _, frame = cap.read()
    gif_frames.append(frame)

# gif_frames = np.asarray(gif_frames)
# imageio.mimsave(f"{base_path}/test.gif", gif_frames, fps=15)

frame_shape = frame.shape
frame_height = frame_shape[0]
frame_width = frame_shape[1]
vertical_pad = max(frame_width-frame_height,0)
horizontal_pad = max(frame_height-frame_width,0)

frame = cv2.copyMakeBorder(frame, vertical_pad//2, vertical_pad//2, horizontal_pad//2, horizontal_pad//2, cv2.BORDER_CONSTANT, 0)
#resized_frame = np.asarray([resize(frame, (INPUT_SIZE, INPUT_SIZE))])
resized_frame = tf.image.resize_with_pad([frame], INPUT_SIZE, INPUT_SIZE)

#plt.imshow(resized_frame)
#plt.savefig("frame_resized.png")
# plt.imshow(frame)
# plt.savefig(f"{base_path}/frame.png")

interpreter = tf.lite.Interpreter(model_path=model_file)
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

keypoints = movenet(resized_frame)

plt.figure(figsize=(20,20))
plt.imshow(frame)
for keypoint in keypoints[0][0]:
    print(keypoint)
    plt.scatter([keypoint[1]*frame.shape[0]],[keypoint[0]*frame.shape[1]],c='green',s=100)
plt.savefig(f"{base_path}/test.png")

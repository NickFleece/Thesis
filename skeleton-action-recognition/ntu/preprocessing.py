import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

VIDEOS_DIR = "H:/ntu_dataset/nturgb+d_rgb"
POSE_MODEL_DIR = "H:/ntu_dataset/pose_model.tflite"

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

for i in os.listdir(VIDEOS_DIR):
    print(i)
    cap = cv2.VideoCapture(f"{VIDEOS_DIR}/{i}")
    _, frame = cap.read()
    plt.imshow(frame)
    plt.show()
    break
import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

VIDEO_FOLDER = "D:/CATSA/CATSA_DATA_SEGMENTED"

pbar = tqdm()

for date in os.listdir(VIDEO_FOLDER):
    for video_file in os.listdir(f"{VIDEO_FOLDER}/{date}"):

        #skip any existing folders (we only want the video files)
        if not ".wmv" in video_file:
            continue

        FRAMES_PATH = f"{VIDEO_FOLDER}/{date}/{video_file[:-4]}"
        
        #delete a folder if it already exists (basically just create a new empty one)
        if os.path.exists(FRAMES_PATH):
            shutil.rmtree(FRAMES_PATH)

        #make the new dir to put the frames
        os.mkdir(FRAMES_PATH)

        vidcap = cv2.VideoCapture(f"{VIDEO_FOLDER}/{date}/{video_file}")
        
        frame_read_success, frame = vidcap.read()
        frame_count = 0
        frame_subsampling = 0
        previous_frame = np.array([])
        while frame_read_success:
            if frame_subsampling % 5 == 0:
                    cv2.imwrite(f"{FRAMES_PATH}/{str(frame_count).zfill(5)}.jpg", frame)
                    frame_count += 1
            frame_subsampling += 1

            previous_frame = frame
            frame_read_success, frame = vidcap.read()

            pbar.set_description(f"{date}, {video_file}, {str(frame_count).zfill(5)}")
        
        pbar.update(1)
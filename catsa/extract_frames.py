import os
import shutil
import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--max_threads', default=1)
args = parser.parse_args()

ALL_VIDEO_FOLDERS = {
    "home":"H:/CATSA/CATSA_DATA_SEGMENTED",
    "laptop":"/media/nick/Thesis/CATSA/CATSA_DATA_SEGMENTED"
}
VIDEO_FOLDER = ALL_VIDEO_FOLDERS[args.env]

ALL_EXPORT_DIR = {
    "home":"H:/CATSA/CATSA_FRAMES"
}
EXPORT_DIR = ALL_EXPORT_DIR[args.env]

save_format='jpg'

total_videos = 0
for date in os.listdir(VIDEO_FOLDER):
    total_videos += len(os.listdir(f"{VIDEO_FOLDER}/{date}"))

curr_file = 0
threads = []
for date in os.listdir(VIDEO_FOLDER):

    for video_file in os.listdir(f"{VIDEO_FOLDER}/{date}"):

        curr_file += 1

        #skip any existing folders (we only want the video files)
        if not ".wmv" in video_file:
            continue

        frames_path = f"{EXPORT_DIR}/{date}/{video_file[:-4]}"
        video_file_path = f"{VIDEO_FOLDER}/{date}/{video_file}"
        
        #delete a folder if it already exists (basically just create a new empty one)
        if os.path.exists(frames_path):
            shutil.rmtree(frames_path)

        #make the new dir to put the frames
        os.mkdir(frames_path)
    
        print(f"{video_file_path}, video {curr_file} / {total_videos}")

        start_time = time.time()

        vidcap = cv2.VideoCapture(f"{VIDEO_FOLDER}/{date}/{video_file}")
        frame_read_success, frame = vidcap.read()
        frame_count = 0
        while frame_read_success:
            cv2.imwrite(f"{frames_path}/{str(frame_count).zfill(5)}.{save_format}", frame)
            frame_count += 1
            frame_read_success, frame = vidcap.read()
            print(frame_count, end='\r')
        
        print(f"\nTook: {time.time() - start_time}, Left: {(time.time() - start_time) * (total_videos - (curr_file + 1))}\n")
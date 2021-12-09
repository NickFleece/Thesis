import os
import shutil
import cv2
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
parser.add_argument('--dataset_index', required=True)
args = parser.parse_args()

ALL_DATES = {
    "1": [
        "y2021m11d08h08m00s00",
        "y2021m11d09h08m00s00",
        "y2021m11d09h20m00s00",
        "y2021m11d10h19m00s00",
        "y2021m11d12h07m00s00",
        "y2021m11d13h19m00s00",
        "y2021m11d15h09m00s00",
        "y2021m11d16h11m00s00",
        "y2021m11d17h07m00s00",
        "y2021m11d17h11m00s00"
    ]
}
DATES = ALL_DATES[args.dataset_index]

ALL_VIDEO_FOLDERS = {
    "home":"H:/CATSA/CATSA_DATA_SEGMENTED",
    "laptop":"/media/nick/Thesis/CATSA/CATSA_DATA_SEGMENTED",
    "wrnch":"/nvme_comm_dat/CATSA/CATSA_DATA_SEGMENTED"
}
VIDEO_FOLDER = ALL_VIDEO_FOLDERS[args.env]

ALL_EXPORT_DIR = {
    "home":"H:/CATSA/CATSA_FRAMES",
    "wrnch":"/nvme_com_dat/CATSA/CATSA_DATA_SEGMENTED"
}
EXPORT_DIR = ALL_EXPORT_DIR[args.env]

save_format='jpg'

total_videos = 0
for date in DATES:
    total_videos += len(os.listdir(f"{VIDEO_FOLDER}/{date}"))

curr_file = 0
threads = []
for date in DATES:

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
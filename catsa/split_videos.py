import pandas
import os
from pathlib import Path
from tqdm import tqdm

SPLIT_CSV_FILE = "/media/nick/External Boi/CATSA/car_splits.csv" #dir of the csv split file
VIDEOS_FOLDER = "/media/nick/External Boi/CATSA/CATSA_DATA" #dir that contains all videos
EXPORT_DIR = "/media/nick/External Boi/CATSA/CATSA_DATA_SEGMENTED" #dir to export segmented videos to, they will be exported in folders according to date

# Camera angle prefixes, first is the prefix of the video, second is the prefix the exported video will have, is usually simpler
CAMERA_ANGLE_PREFIXES = [
    ["FD1(2667)", "FD"],
    ["FP(2668)", "FP"],
    ["RP(2670)", "RP"]
]

splits = pandas.read_csv(SPLIT_CSV_FILE)

pbar = tqdm(total=len(splits) * len(CAMERA_ANGLE_PREFIXES))
for split in splits.iterrows():
    split = split[1]

    Path(f"{EXPORT_DIR}/{split['File']}").mkdir(parents=True, exist_ok=True)

    for camera_prefix in CAMERA_ANGLE_PREFIXES:
        os.system(f"ffmpeg -y -hide_banner -loglevel error -to {split['Car_End']} -i '{VIDEOS_FOLDER}/{camera_prefix[0]}_{split['File']}.wmv' -ss {split['Car_Start']} -c copy '{EXPORT_DIR}/{split['File']}/{camera_prefix[1]}_{split['Car_Index']}.wmv'")
        pbar.update(1)

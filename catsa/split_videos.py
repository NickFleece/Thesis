import pandas
import os
from pathlib import Path
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_index', required=True)

args = parser.parse_args()

SPLIT_CSV_FILE = "H:/CATSA/car_splits.csv" #dir of the csv split file
EXPORT_DIR = "H:/CATSA/CATSA_DATA_SEGMENTED" #dir to export segmented videos to, they will be exported in folders according to date

#dir that contains all videos
ALL_VIDEOS_FOLDER = {
    0:"H:/CATSA/CATSA_DATA",
    1:"H:/CATSA/CATSA_videos_24Nov21",
    2:"H:/CATSA/CATSA_videos_11Jan22",
}

# Camera angle prefixes, first is the prefix of the video, second is the prefix the exported video will have, is usually simpler
ALL_CAMERA_ANGLE_PREFIXES = {
    0:
        [
            ["FD1(2667)", "FD"],
            ["FP(2668)", "FP"],
            ["RP(2670)", "RP"]
        ],
    1:
        [
            ["YYC_NPS-V_N_SB1_FD1(2667)", "FD"],
            ["YYC_NPS-V_N_SB1_FP(2668)", "FP"],
            ["YYC_NPS-V_N_SB1_RP(2670)", "RP"]
        ],
    2:
        [
            ["YYC_NPS-V_N_SB1_FD1(2667)", "FD"],
            ["YYC_NPS-V_N_SB1_FP(2668)", "FP"],
            ["YYC_NPS-V_N_SB1_RP(2670)", "RP"]
        ]
}

VIDEOS_FOLDER = ALL_VIDEOS_FOLDER[int(args.dataset_index)]
CAMERA_ANGLE_PREFIXES = ALL_CAMERA_ANGLE_PREFIXES[int(args.dataset_index)]

splits = pandas.read_csv(SPLIT_CSV_FILE)
splits = splits.loc[splits['Dataset_Index'] == int(args.dataset_index)]

pbar = tqdm(total=len(splits) * len(CAMERA_ANGLE_PREFIXES))
for split in splits.iterrows():
    split = split[1]

    Path(f"{EXPORT_DIR}/{split['File']}").mkdir(parents=True, exist_ok=True)

    for camera_prefix in CAMERA_ANGLE_PREFIXES:
        pbar.set_description(f"{camera_prefix[0]}_{split['File']}.wmv; {split['Car_Start']} to {split['Car_End']}")
        os.system(f"ffmpeg -y -hide_banner -loglevel error -to {split['Car_End']} -i \"{VIDEOS_FOLDER}/{camera_prefix[0]}_{split['File']}.wmv\" -ss {split['Car_Start']} -c copy \"{EXPORT_DIR}/{split['File']}/{camera_prefix[1]}_{split['Car_Index']}.wmv\"")
        pbar.update(1)
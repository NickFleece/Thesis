import os
import torch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--version', required=True)
args = parser.parse_args()

BASE_DIR = args.drive_dir
VERSION = args.version
MODEL_SAVE_DIR = f"{BASE_DIR}/catsa_models/m_{VERSION}"

count = 0
while True:

    dict_path = f"{MODEL_SAVE_DIR}/{count}"

    if not os.path.exists(dict_path): break

    print(torch.load(dict_path, map_location=device).keys())
    print(torch.load(dict_path, map_location=device)['val_outputs'])
    print()
    print(torch.load(dict_path, map_location=device)['val_correct'])

    count += 1
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', required=True)
args = parser.parse_args()

FILE_DIR = args.file_dir

results = torch.load(FILE_DIR, map_location=torch.device('cpu'))
print(results['val_predicted'][:10])
print(results['val_actual'][:10])
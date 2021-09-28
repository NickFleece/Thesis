import os
import matplotlib.pyplot as plt
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--version')
parser.add_argument('--env')
args = parser.parse_args()

version = args.version

if args.env == 'alpha':
    MODEL_DIR = f"/comm_dat/nfleece/models/m_{version}"

with open(f"{MODEL_DIR}/hist", 'rb') as hist_file:
    hist = pickle.load(hist_file)

plt.figure()
plt.ylim((0,1))
plt.plot(range(len(hist['Train'])), hist['Train'])
plt.plot(range(len(hist['Val'])), hist['Val'])
plt.savefig(f"hist_figure_m_{version}")

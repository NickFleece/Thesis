import pandas as pd
import torch
import os
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import _thread

result_dir = "/comm_dat/nfleece/JHMDB/models"

print(f"Analyzing {len(os.listdir(result_dir))} total models...")
results = {}
count = 0

def readModelData(modelName):
    max_acc = 0
    max_train_acc = 0

    max_e = None

    for e in os.listdir(f"{result_dir}/{modelName}"):

        if e == 'model': continue

        res = torch.load(f"{result_dir}/{modelName}/{e}", map_location=torch.device('cpu'))
        acc = accuracy_score(res['val_actual'], res['val_predicted'])
        train_acc = accuracy_score(res['train_actual'], res['train_predicted'])

        if acc > max_acc:
            max_acc = acc
            max_e = e
        if train_acc > max_train_acc:
            max_train_acc = train_acc

    results[m] = max_acc

for m in os.listdir(f"{result_dir}"):

    count += 1
    print(f"{count} - {m}")

    _thread.start_new_thread(readModelData, (m))
    # time.sleep(1)

    # max_acc = 0
    # max_train_acc = 0

    # max_e = None

    # for e in os.listdir(f"{result_dir}/{m}"):

    #     if e == 'model': continue

    #     res = torch.load(f"{result_dir}/{m}/{e}", map_location=torch.device('cpu'))
    #     acc = accuracy_score(res['val_actual'], res['val_predicted'])
    #     train_acc = accuracy_score(res['train_actual'], res['train_predicted'])

    #     if acc > max_acc:
    #         max_acc = acc
    #         max_e = e
    #     if train_acc > max_train_acc:
    #         max_train_acc = train_acc

    # results[m] = max_acc

    # print(f"Epoch {max_e}")
    # print(f"Val Accuracy: {max_acc}")
    # print(f"Train Accuracy: {max_train_acc}")

    # print("____________________________")

s = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
for k in s.keys():
    print(f"{k} - {s[k]}")

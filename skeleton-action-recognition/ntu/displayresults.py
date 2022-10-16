import os
import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
args = parser.parse_args()

RESULTS_DIR = args.results_dir

accuracies = {}

print("starting...")
for folder in os.listdir(RESULTS_DIR):
    print(f"\n Version: {folder}")

    max_acc = 0.0
    for result_file in os.listdir(f"{RESULTS_DIR}/{folder}"):

        result = torch.load(f"{RESULTS_DIR}/{folder}/{result_file}", map_location=torch.device('cpu'))

        max_epoch = None

        acc = accuracy_score(result['val_actual'], result['val_predicted'])
        if acc > max_acc:

            max_acc = acc
            max_prec = precision_score(result['val_actual'], result['val_predicted'], average="micro")
            max_rec = recall_score(result['val_actual'], result['val_predicted'], average="micro")
            conf_matrix = confusion_matrix(result['val_actual'], result['val_predicted'])

            max_epoch = result_file
        
    print(f"Max Accuracy: {max_acc}")
    print(f"Max Precision: {max_acc}")
    print(f"Max Recall: {max_rec}")
    print(f"Epoch: {max_epoch}")
    print(conf_matrix)

    accuracies[folder] = max_acc

print("---")
print("Sorted:")
sorted = {k: v for k, v in sorted(accuracies.items(), key=lambda item: item[1])}
for k in sorted.keys():
    print(f"{k} - {sorted[k]}")
import torch
import argparse
from sklearn import metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--models_dir', required=True)
args = parser.parse_args()

for model in os.listdir(args.models_dir):

    model_path = f"{args.models_dir}/{model}"

    print(f"\nRunning tests on model: {model}")
    best_accuracy = 0
    best_epoch = None

    for epoch in os.listdir(model_path):
        
        results = torch.load(f"{model_path}/{epoch}", map_location=torch.device("cpu"))

        accuracy = metrics.accuracy_score(results['val_actual'], results['val_outputs'])
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
    
    print(f"Best epoch is {epoch} with accuracy {accuracy}")
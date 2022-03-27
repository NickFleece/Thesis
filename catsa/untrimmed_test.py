import torch
import argparse
from torch import nn
import torch.functional as F
from torchvision.models.video import r3d_18
from sklearn import metrics
import os

parser = argparse.ArgumentParser()
parser.add_argument('--frames_dir', required=True)
parser.add_argument('--model_file', required=True)
parser.add_argument('--bytetrack_annotation_dir', required=True)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

used_labels = [
    0,
    # 1, #not used, 79 samples
    2,
    3,
    # 4, #not used, 41 samples
    5,
    # 6 #not used, 88 samples
]

class VideoRecognitionModel(nn.Module):

    def __init__(self):
        super(VideoRecognitionModel, self).__init__()

        self.pretrained_model = nn.Sequential(*list(r3d_18(pretrained=True, progress=True).children())[:-1])
        
        self.fc1 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(512, len(used_labels))

    def forward(self, x):

        # ensure the pretrained model is frozen
        self.pretrained_model.requires_grad_ = False

        x = self.pretrained_model(x).squeeze()

        # if a batch of size 1 was put through, ensure that the batch is preserved
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        x = F.dropout(x, 0.5)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x

model = VideoRecognitionModel()
model.to(device)

model_data = torch.load(args.model_file, map_location=device)

accuracy = metrics.accuracy_score(model_data['val_actual'], model_data['val_outputs'])
recall = metrics.recall_score(model_data['val_actual'], model_data['val_outputs'], average='macro')
precision = metrics.precision_score(model_data['val_actual'], model_data['val_outputs'], average='macro')

print("Metrics:")
print("Acc:" + str(accuracy))
print("Recall:" + str(recall))
print("Precision" + str(precision))

model.load_state_dict(model_data['model_state_dict'])
model.eval()

#Time to test!
for date in os.listdir(args.frames_dir):
    print(date)
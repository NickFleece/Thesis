import pandas as pd
import argparse
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
import cv2

matplotlib.use('TKAgg')

parser = argparse.ArgumentParser()
parser.add_argument('--frames_folder', required=True)
args = parser.parse_args()

FRAMES_DIR = args.frames_folder

annotations = pd.read_csv(f"{FRAMES_DIR}/annotations.csv")

count = 0

for _, annotation in annotations.iterrows():

    count += 1
    print(f"{count} / {len(annotations)} - {count / len(annotations)}")
    print(annotation)

    if count < 2845: continue

    fig = plt.figure()

    start = int(annotation['start_frame'][:-4])
    end = int(annotation['end_frame'][:-4])

    images = []

    bbox = annotation['enclosing_bbox'].split(" ")
    x_1 = max(int(bbox[0]), 0)
    y_1 = max(int(bbox[1]), 0)
    x_2 = int(bbox[2])
    y_2 = int(bbox[3])

    for f in range(start, end+1, 6):

        image = np.asarray(imageio.imread(f"{FRAMES_DIR}/{annotation['path_to_video_segment']}/{str(f).zfill(5)}.jpg"))[y_1:y_2,x_1:x_2]
        images.append([plt.imshow(image, animated=True)])

    ani = animation.ArtistAnimation(fig, images, interval=1)
    plt.title(f"{count} - {annotation['path_to_video_segment']} - {annotation['activity_class_name']}")
    plt.show()
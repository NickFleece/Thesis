import pandas as pd
import argparse
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
args = parser.parse_args()

ALL_FRAMES_DIR = {
    "home":"H:/CATSA/CATSA_FRAMES",
    "wrnch":"/nvme_com_dat/CATSA/CATSA_DATA_SEGMENTED"
}
FRAMES_DIR = ALL_FRAMES_DIR[args.env]

annotations = pd.read_csv(f"{FRAMES_DIR}/annotations.csv")

count = 0

for _, annotation in annotations.iterrows():

    count += 1
    print(f"{count} / {len(annotations)} - {count / len(annotations)}")
    print(annotation)

    if count < 1051: continue

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
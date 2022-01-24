import os
import cv2
import matplotlib.pyplot as plt

SKELETON_FILES_DIR = "H:/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"
VIDEO_FILES_DIR = "H:/ntu_dataset/nturgb+d_rgb"

for i in os.listdir(SKELETON_FILES_DIR):

    cap = cv2.VideoCapture(f"{VIDEO_FILES_DIR}/{i[:-9]}_rgb.avi")

    _, frame = cap.read()

    plt.imshow(frame)
    plt.show()

    with open(f"{SKELETON_FILES_DIR}/{i}") as f:
        num_frames = int(f.readline())
        for frame in range(num_frames):

            num_people = int(f.readline())
            
            for person in range(num_people):
                plt.figure()
                plt.ylim((-2.5,2.5))
                plt.xlim((-2.5,2.5))

                random_details = f.readline()
                num_points = int(f.readline())

                skeleton_points = []

                for _ in range(num_points):

                    point = f.readline().split(" ")[:2]
                    for p in range(len(point)): point[p] = float(point[p])
                    plt.scatter(point[0], point[1])

                plt.show()
            break
        break

    break
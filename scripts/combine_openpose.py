DATA_PATH = "../../../../../nvme_comm_dat/JHMDB_Potion/JHMDB/OpenPose_Heatmaps"

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import threading
from queue import Queue
import time

MAX_THREADS = 3
pbar = tqdm(total=928)
q = Queue()

#name is j
#hmap_folder is heatmaps_folder
def createnpz(name, hmap_folder):
    all_heatmaps = []
    heatmap_index = 0
    while (os.path.exists(hmap_folder + "/" + name + "_" + str(heatmap_index).zfill(12) + "_pose_heatmaps.png")):
        all_heatmaps.append(
            plt.imread(hmap_folder + "/" + name + "_" + str(heatmap_index).zfill(12) + "_pose_heatmaps.png"))
        heatmap_index += 1

    with open(hmap_folder + "/" + name + ".npy", 'wb') as f:
        np.savez_compressed(f, np.asarray(all_heatmaps))

    q.task_done()

tasks_in_progress = 0
for i in os.listdir(DATA_PATH):
    pbar.set_description(i)
    class_folder_path = DATA_PATH + "/" + i
    
    for j in os.listdir(class_folder_path):
        heatmaps_folder = class_folder_path + "/" + j

        if tasks_in_progress >= MAX_THREADS:
            q.join()
            tasks_in_progress = 0

        t = threading.Thread(target=createnpz, args=(j,heatmaps_folder))
        q.put(t)
        t.start()
        tasks_in_progress += 1

        time.sleep(0.5)
        pbar.update(1)

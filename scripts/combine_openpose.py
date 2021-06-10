DATA_PATH = "../../../../../nvme_comm_dat/JHMDB_Potion/JHMDB/OpenPose_Heatmaps"

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

pbar = tqdm(total=928)
for i in os.listdir(DATA_PATH):
    pbar.set_description(i)
    class_folder_path = DATA_PATH + "/" + i
    
    for j in os.listdir(class_folder_path):
        heatmaps_folder = class_folder_path + "/" + j
        
        all_heatmaps = []
        heatmap_index = 0
        #while(os.path.exists(heatmaps_folder + "/" + j + "_" + str(heatmap_index).zfill(12) + "_pose_heatmaps.png")):
            #all_heatmaps.append(plt.imread(heatmaps_folder + "/" + j + "_" + str(heatmap_index).zfill(12) + "_pose_heatmaps.png"))
            #heatmap_index += 1

        #with open(heatmaps_folder + "/" + j + ".npy", 'wb') as f:
            #np.save(f, np.asarray(all_heatmaps))
        
        with open(heatmaps_folder + "/" + j + ".npy", 'rb') as f:
            np.load(f)

        pbar.update(1)

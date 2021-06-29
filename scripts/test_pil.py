import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DATA_PATH = "../JHMDB/OpenPose_Heatmaps/catch/Ballfangen_catch_u_cm_np1_fr_goo_0"

print("Loading using matplotlib...")
start = time.time()
all = []
for i in range(30):
    all.append(
        plt.imread(f"{DATA_PATH}/Ballfangen_catch_u_cm_np1_fr_goo_0_{str(i).zfill(12)}_pose_heatmaps.png")
    )
print(f"{time.time() - start}s")

print("Loading using pillow...")
start = time.time()
all = []
for i in range(30):
    all.append(
        np.asarray(Image.open(f"{DATA_PATH}/Ballfangen_catch_u_cm_np1_fr_goo_0_{str(i).zfill(12)}_pose_heatmaps.png"))
    )
print(f"{time.time() - start}s")
print(np.asarray(all).shape)
import os
# SELECT WHAT GPU TO USE
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
# ONLY PRINT ERRORS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout, ReLU
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import queue
import threading
import pickle
import random

# Constants/File Paths
DATA_PATH = "../../../../comm_dat/nfleece/JHMDB"
MODEL_SAVE_DIR = "models/tf_test_model"
HIST_SAVE_DIR = "models/tf_test_model_hist.pickle"
EPOCHS = 200
SLICE_INDEX = 1
BATCH_SIZE = 16
RANDOM_SEED = 123
NUM_WORKERS = tf.data.AUTOTUNE

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
for i in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(i, True)
    except:
        print("Something went wrong while setting memory growth...")
        continue

classes = [
    'kick_ball',
    'pick',
    'climb_stairs',
    'clap',
    'pullup',
    'golf',
    'brush_hair',
    'catch',
    'pour',
    'jump',
    'swing_baseball',
    'shoot_bow',
    'walk',
    'run',
    'sit',
    'throw',
    'push',
    'shoot_ball',
    'stand',
    'shoot_gun',
    'wave'
]

parts = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel"
]

part_side = [
    "T",
    "T",
    "R",
    "R",
    "R",
    "L",
    "L",
    "L",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "T",
    "T",
    "T",
    "T",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
]

sides = ["T", "B", "L", "R"]

data = []
#pbar = tqdm(total=len(classes))
for c in classes:
    #pbar.set_description(f"Class: {c}")

    for i in range(1, 4):

        lines = open(f"{DATA_PATH}/splits/{c}_test_split{i}.txt").read().splitlines()

        for L in lines:
            f, train_or_test = L.split(' ')

            if train_or_test == '1':
                split = "train"

            else:
                split = "test"

            data.append({
                "file": f,
                "class": c,
                "split": split,
                "ind": i
            })

    #pbar.update(1)
    #time.sleep(0.1)

data = pd.DataFrame(data)

train_data = data.loc[data['ind'] == SLICE_INDEX].loc[data['split'] == 'train'][["file", "class"]].values
val_data = data.loc[data['ind'] == SLICE_INDEX].loc[data['split'] == 'test'][["file", "class"]].values

train_ds = tf.data.Dataset.from_tensor_slices(train_data)
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

def process_data(data_tensor):
    data_arr = data_tensor.numpy()
    c = data_arr[1].decode('ascii')
    f = data_arr[0].decode('ascii')

    openpose_heatmaps_dir = f"{DATA_PATH}/OpenPose_Heatmaps/{c}/{f[:-4]}"

    all_heatmaps = np.load(f"{openpose_heatmaps_dir}/{f[:-4]}.npz")['arr_0']

    slice_size = int(np.asarray(all_heatmaps).shape[2] / 25)

    flip_img = random.random() > 0.5
    
    hide_parts = random.random() > 0.4
    part_to_hide = random.choice(parts)

    U = []
    I = []
    N = []

    for i in range(len(parts)):

        part_heatmaps = []
        for heatmap in all_heatmaps:
            part_map = heatmap[:, slice_size * i:slice_size * i + slice_size]
            part_heatmaps.append(part_map)

        channels = 3
        concat_heatmap = np.zeros((part_heatmaps[0].shape[0], part_heatmaps[0].shape[1], channels))

        if not (part_side[i] == part_to_hide and hide_parts):
            for j in range(len(all_heatmaps)):
                part_map = part_heatmaps[j]

                t = (j + 1) / len(all_heatmaps)

                for k in range(channels):
                    step = (1 / (channels - 1)) * k
                    channel_map = part_map * max((-(channels - 1) * (abs(t - step)) + 1), 0)
                    concat_heatmap[:, :, k] += channel_map

        if not concat_heatmap.max() == 0:
            for k in range(channels):
                concat_heatmap[:, :, k] /= concat_heatmap.max()
        Uj = concat_heatmap

        if flip_img:
            Uj = np.flip(Uj, axis=1)
        
        U.append(Uj)

        Ij = np.sum(Uj, axis=2)
        I.append(Ij)

        i3 = np.reshape(Ij, (Ij.shape[0], Ij.shape[1], 1))
        i3 = np.repeat(i3, 3, axis=2)
        Nj = Uj / (i3 + 1)
        N.append(Nj)

    U = np.asarray(U)
    I = np.asarray(I)
    N = np.asarray(N)

    single_result_arr = []

    # go from (x,x,3) to (3,x,x), same idea for the others, just stacking all channels
    for u in U:
        for j in range(3):
            single_result_arr.append(u[:, :, j])

    for i in I:
        single_result_arr.append(i)

    for n in N:
        for j in range(3):
            single_result_arr.append(n[:, :, j])

    single_target = np.zeros(len(classes))
    single_target[classes.index(c)] = 1

    return np.asarray(single_result_arr), np.asarray(single_target)

def load_data(data_tensor):
    return tf.py_function(process_data, inp=[data_tensor], Tout=[tf.int64, tf.int64])

train_ds = train_ds.shuffle(len(train_data), reshuffle_each_iteration=True).map(load_data, num_parallel_calls=NUM_WORKERS).batch(BATCH_SIZE)
val_ds = val_ds.shuffle(len(val_data), reshuffle_each_iteration=True).map(load_data, num_parallel_calls=NUM_WORKERS).batch(BATCH_SIZE)

#regularizer = l2(0.005)

model_init = tf.keras.initializers.GlorotNormal(seed=RANDOM_SEED)

model = Sequential() #add model layers

model.add(Conv2D(128, kernel_size=3, strides=(2,2), input_shape=(175,368,496), kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(128, kernel_size=3, kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())

model.add(Conv2D(256, kernel_size=3, strides=(2,2), kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(256, kernel_size=3, kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())

model.add(Conv2D(512, kernel_size=3, strides=(2,2), kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Conv2D(512, kernel_size=3, kernel_initializer=model_init))
model.add(BatchNormalization())
model.add(ReLU())

model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.8))
# model.add(Dense(512))
model.add(Dropout(0.8))
model.add(Dense(21, activation='softmax', kernel_initializer=model_init))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss='categorical_crossentropy',
    metrics=[categorical_accuracy]
)

hist = model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    # steps_per_epoch=len(train_data)//BATCH_SIZE,
    # validation_steps=len(val_data)//BATCH_SIZE,
    # workers=3,
    # use_multiprocessing=True,
)

model.save(MODEL_SAVE_DIR)
with open(HIST_SAVE_DIR, 'wb') as f:
    pickle.dump(hist.history, f)

# Constants/File Paths
DATA_PATH = "../../../../nvme_comm_dat/JHMDB_Potion/JHMDB"
MODEL_SAVE_DIR = "models/first_test_model"
EPOCHS = 100
SLICE_INDEX = 1
BATCH_SIZE = 15

#Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

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

data = []
pbar = tqdm(total=len(classes))
for c in classes:
    pbar.set_description(f"Class: {c}")

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

    pbar.update(1)
    time.sleep(0.1)

data = pd.DataFrame(data)

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, ind, batch_size=8, verbose=False):
        self.seed = 15
        self.df_train = df.loc[df['ind'] == ind].loc[df['split'] == 'train'].sample(random_state=self.seed, frac=1)
        self.df_test = df.loc[df['ind'] == ind].loc[df['split'] == 'test'].sample(random_state=self.seed, frac=1)
        self.batch_size = batch_size
        self.verbose = verbose

        self.target_class_map = [
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

        self.parts = [
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

    def __len__(self):
        return len(self.df_train) // self.batch_size

    def __getitem__(self, index):
        batch_df = self.df_train.iloc[index * self.batch_size: (index + 1) * self.batch_size]

        result_arr = []
        target = []

        if self.verbose:
            pbar = tqdm(total=self.batch_size, desc=f"Batch {index} / {self.__len__()}")

        for i in batch_df.iterrows():
            f = i[1]['file']
            c = i[1]['class']

            openpose_heatmaps_dir = f"{DATA_PATH}/OpenPose_Heatmaps/{c}/{f[:-4]}"

            with open(f"{openpose_heatmaps_dir}/{f[:-4]}.npy", 'rb') as f:
                all_heatmaps = np.load(f)

            slice_size = int(np.asarray(all_heatmaps).shape[2]/25)

            U = []
            I = []
            N = []

            for i in range(len(self.parts)):

                part_heatmaps = []
                for heatmap in all_heatmaps:
                    part_map = heatmap[:, slice_size * i:slice_size * i + slice_size]
                    part_heatmaps.append(part_map)

                channels = 3
                concat_heatmap = np.zeros((part_heatmaps[0].shape[0], part_heatmaps[0].shape[1], channels))

                for j in range(len(all_heatmaps)):
                    part_map = part_heatmaps[j]

                    t = (j + 1) / len(all_heatmaps)

                    for k in range(channels):
                        step = (1 / (channels - 1)) * k
                        channel_map = part_map * max((-(channels - 1) * (abs(t - step)) + 1), 0)
                        concat_heatmap[:, :, k] += channel_map

                for k in range(channels):
                    concat_heatmap[:, :, k] /= concat_heatmap.max()

                Uj = concat_heatmap
                U.append(concat_heatmap)

                Ij = np.sum(Uj, axis=2)
                I.append(Ij)

                i3 = np.reshape(Ij, (Ij.shape[0], Ij.shape[1], 1))
                i3 = np.repeat(i3, 3, axis=2)
                Nj = Uj/(i3+1)
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

            result_arr.append(single_result_arr)

            single_target = np.zeros(len(self.target_class_map))
            single_target[self.target_class_map.index(c)] = 1
            target.append(single_target)

            if self.verbose:
                pbar.update(1)

        result_arr = np.asarray(result_arr)
        target = np.asarray(target)

        return result_arr, target

    def on_epoch_end(self):
        self.df_train = self.df_train.sample(random_state=self.seed, frac=1)
        # self.df_test = self.df_test.sample(random_state=self.seed, frac=1)

dg = DataGenerator(data, 1, verbose=True)

RANDOM_SEED = 123
model_init = tf.keras.initializers.GlorotNormal(seed=RANDOM_SEED)

model = Sequential() #add model layers

model.add(Conv2D(128, kernel_size=3, strides=(2,2), activation='relu', input_shape=(175,368,496), kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=3, strides=(2,2), activation='relu', kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=3, strides=(2,2), activation='relu', kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size=3, activation='relu', kernel_initializer=model_init))
model.add(Dropout(0.25))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(21, activation='softmax', kernel_initializer=model_init))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=[categorical_accuracy]
)

generator = DataGenerator(data, SLICE_INDEX, batch_size=BATCH_SIZE)

model.fit(x=generator, epochs=EPOCHS, verbose=1)

model.save(MODEL_SAVE_DIR)
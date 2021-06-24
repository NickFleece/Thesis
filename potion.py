import os
# SELECT WHAT GPU TO USE
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# ONLY PRINT ERRORS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Constants/File Paths
DATA_PATH = "../../../../comm_dat/nfleece/JHMDB"
MODEL_SAVE_DIR = "models/2_test_model"
HIST_SAVE_DIR = "models/2_test_model_hist.pickle"
EPOCHS = 100
SLICE_INDEX = 1
BATCH_SIZE = 8
RANDOM_SEED = 123

#Imports
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import time
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import queue
import threading
import pickle
import random

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

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, ind, batch_size=8, train_or_test='train'):
        self.seed = RANDOM_SEED
        self.batch_size = batch_size

        self.data_df = df.loc[df['ind'] == ind].loc[df['split'] == train_or_test].sample(random_state=self.seed, frac=1)

        self.job_queue = queue.Queue()
        self.result_queue = queue.Queue()

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
        return len(self.data_df) // self.batch_size

    def getDataItem(self, f, c):
        openpose_heatmaps_dir = f"{DATA_PATH}/OpenPose_Heatmaps/{c}/{f[:-4]}"

        all_heatmaps = np.load(f"{openpose_heatmaps_dir}/{f[:-4]}.npz")['arr_0']

        slice_size = int(np.asarray(all_heatmaps).shape[2] / 25)

        flip_img = random.random() > 0.5

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

        single_target = np.zeros(len(self.target_class_map))
        single_target[self.target_class_map.index(c)] = 1

        self.result_queue.put((single_target, single_result_arr))
        self.job_queue.task_done()

    def __getitem__(self, index):
        batch_df = self.data_df.iloc[index * self.batch_size: (index + 1) * self.batch_size]

        result_arr = []
        target = []

        thread_count = 0
        for i in batch_df.iterrows():
            f = i[1]['file']
            c = i[1]['class']

            #MULTITHREADING HERE
            t = threading.Thread(target=self.getDataItem, args=(f,c))
            t.start()
            self.job_queue.put(t)
            thread_count += 1

            if thread_count == 9:
                self.job_queue.join()
                thread_count = 0

        self.job_queue.join()

        while not self.result_queue.empty():
            single_target, single_result_arr = self.result_queue.get()

            result_arr.append(single_result_arr)
            target.append(single_target)

        result_arr = np.asarray(result_arr)
        target = np.asarray(target)

        return result_arr, target

    def on_epoch_end(self):
        self.data_df = self.data_df.sample(random_state=self.seed, frac=1)

#dg = DataGenerator(data, 1, batch_size=20)
#print(dg.__getitem__(0))

regularizer = l2(0.0005)

model_init = tf.keras.initializers.GlorotNormal(seed=RANDOM_SEED)

model = Sequential() #add model layers

model.add(Conv2D(128, kernel_size=3, strides=(2,2), activation='relu', input_shape=(175,368,496), kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=3, strides=(2,2), activation='relu', kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size=3, activation='relu', kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=3, strides=(2,2), activation='relu', kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())
model.add(Conv2D(512, kernel_size=3, activation='relu', kernel_initializer=model_init, kernel_regularizer=regularizer, bias_regularizer=regularizer))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.8))
# model.add(Dense(512))
model.add(Dropout(0.8))
model.add(Dense(21, activation='softmax', kernel_initializer=model_init))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=[categorical_accuracy]
)

generator = DataGenerator(data, SLICE_INDEX, batch_size=BATCH_SIZE)
val_generator = DataGenerator(data, SLICE_INDEX, batch_size=BATCH_SIZE, train_or_test='test')

hist = model.fit(
    x=generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
    workers=3,
    use_multiprocessing=True,
)

model.save(MODEL_SAVE_DIR)
with open(HIST_SAVE_DIR, 'wb') as f:
    pickle.dump(hist.history, f)

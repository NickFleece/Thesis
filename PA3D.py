import os
# SELECT WHAT GPU TO USE
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# ONLY PRINT ERRORS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Flatten, GlobalAveragePooling2D, Dropout, ReLU, Conv3D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
import pickle
from PIL import Image

# Constants/File Paths
DATA_PATH = "../../../../comm_dat/nfleece/JHMDB"
MODEL_SAVE_DIR = "models/pa3d_1_model"
HIST_SAVE_DIR = "models/pa3d_1_model_hist.pickle"
EPOCHS = 200
SLICE_INDEX = 1
BATCH_SIZE = 16
RANDOM_SEED = 123
VIDEO_PADDED_LEN = 40
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

    all_heatmaps = []
    for i in range(VIDEO_PADDED_LEN):
        if not os.path.exists(f"{openpose_heatmaps_dir}/{c}_{str(i).zfill(12)}_pose_heatmaps.png"):
            continue

        all_heatmaps.append(
            np.asarray(Image.open(f"{openpose_heatmaps_dir}/{c}_{str(i).zfill(12)}_pose_heatmaps.png"))
        )

    img_shape = all_heatmaps[0].shape
    while len(all_heatmaps) != VIDEO_PADDED_LEN:
        all_heatmaps.append(np.zeros(img_shape))

    images = np.asarray(all_heatmaps)

    target = np.zeros(len(classes))
    target[classes.index(c)] = 1

    return images, target

def load_data(data_tensor):
    return tf.py_function(process_data, inp=[data_tensor], Tout=[tf.int64, tf.int64])

FILTERS_1D = 6

class Split_Joints(tf.keras.layers.Layer):
    def call(self, inputs):
        x = tf.concat(tf.split(inputs, 25, axis=3), axis=1)
        x = tf.concat(tf.split(x, FILTERS_1D, axis=4), axis=1)
        x = tf.reshape(x, tf.shape(x)[:-1])
        return x

train_ds = train_ds.shuffle(len(train_data), reshuffle_each_iteration=True).map(load_data, num_parallel_calls=NUM_WORKERS).batch(BATCH_SIZE)
val_ds = val_ds.shuffle(len(val_data), reshuffle_each_iteration=True).map(load_data, num_parallel_calls=NUM_WORKERS).batch(BATCH_SIZE)

model_init = tf.keras.initializers.GlorotNormal(seed=RANDOM_SEED)

model = Sequential() #add model layers

model.add(Conv3D(FILTERS_1D, (VIDEO_PADDED_LEN,1,1), input_shape=(VIDEO_PADDED_LEN, 368, 12400,1)))
model.add(Split_Joints())

model.add(Conv2D(128, kernel_size=3, strides=(2,2), input_shape=(FILTERS_1D * 25,368,496), kernel_initializer=model_init))
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
model.add(Dropout(0.8))
model.add(Dense(21, activation='softmax', kernel_initializer=model_init))

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=[categorical_accuracy]
)

hist = model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

model.save(MODEL_SAVE_DIR)
with open(HIST_SAVE_DIR, 'wb') as f:
    pickle.dump(hist.history, f)

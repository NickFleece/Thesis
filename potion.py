# Constants
DATA_PATH = "../../nvme_comm_dat/JHMDB_Potion/JHMDB"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, ind, batch_size=8, verbose=False):
        self.seed = 15
        self.df_train = df.loc[df['ind'] == ind].loc[df['split'] == 'train'].sample(random_state=self.seed, frac=1)
        self.df_test = df.loc[df['ind'] == ind].loc[df['split'] == 'test'].sample(random_state=self.seed, frac=1)
        self.batch_size = batch_size
        self.video_folder_path = f"/content/drive/MyDrive/School/Thesis/JHMDB/Videos"
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

            all_heatmaps = []
            heatmap_index = 0
            while (os.path.exists(f"{openpose_heatmaps_dir}/{f[:-4]}_{str(heatmap_index).zfill(12)}_pose_heatmaps.png")):
                all_heatmaps.append(plt.imread(f"{openpose_heatmaps_dir}/{f[:-4]}_{str(heatmap_index).zfill(12)}_pose_heatmaps.png"))
                heatmap_index += 1

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

            # U = np.load(f"{self.video_folder_path}/{c}/U_heatmaps/{f[:-4]}.npz")['arr_0']
            # I = np.load(f"{self.video_folder_path}/{c}/I_heatmaps/{f[:-4]}.npz")['arr_0']
            # N = np.load(f"{self.video_folder_path}/{c}/N_heatmaps/{f[:-4]}.npz")['arr_0']

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
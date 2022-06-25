import json
import pandas as pd
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from config import BONE_CONNECTIONS
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--drive_dir', required=True)
parser.add_argument('--model_file', required=True)
parser.add_argument('--overwrite_pose', default=0)
parser.add_argument('--overwrite_rep', default=0)
args = parser.parse_args()

DRIVE_DIR = args.drive_dir
TF_MODEL_FILE = args.model_file
OVERWRITE_POSE = args.overwrite_pose
OVERWRITE_REP = args.overwrite_rep
INPUT_SIZE=256

interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE)
interpreter.allocate_tensors()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

folders = [
    "825312072753_31.Oct.2018_10.12.10",
    "825312072753_31.Oct.2018_13.22.27",
    "825312073198_31.Oct.2018_10.11.56",
    "825312073198_31.Oct.2018_13.22.19"
]

annotation_files = [
    "825312072753_31.Oct.2018_10.12.10_with_category_id_16July19.json",
    "825312072753_31.Oct.2018_13.22.27_with_categories_ids_14_Aug_2019.json",
    "825312073198_31.Oct.2018_10.11.56_with_category_tracks_16July19.json",
    "825312073198_31.Oct.2018_13.22.19_with_categories_ids_14_Aug_2019.json"
]

data_summary = []

for folder, annotation_file in zip(folders, annotation_files):

    folder_dir = f"{DRIVE_DIR}/{folder}"

    with open(f"{folder_dir}/{annotation_file}", 'r') as f:
        annotations = json.load(f)

    bone_angle_annotations = {}

    print(folder)
    # pbar = tqdm(total=len(list(annotations.keys())))
    for file in list(annotations.keys()):

        for a in annotations[file]['annotations']:

            if a['category_instance_id'] is None:
                a['category_instance_id'] = "None"

            # pbar.set_description(f"{file}")

            if not os.path.exists(f"{folder_dir}/color/{file}"): 
                # print(f"Skipping file: {file}")
                continue

            a['category'] = str(a['category'])
            a['category_instance_id'] = str(a['category_instance_id'])
            a['id'] = str(a['id'])

            if a['category'] not in bone_angle_annotations: bone_angle_annotations[a['category']] = {}
            if a['category_instance_id'] not in bone_angle_annotations[a['category']]: bone_angle_annotations[a['category']][a['category_instance_id']] = {}
            if a['id'] not in bone_angle_annotations[a['category']][a['category_instance_id']]: bone_angle_annotations[a['category']][a['category_instance_id']][a['id']] = {}

            if os.path.exists(f"{folder_dir}/extracted_pose/{a['category']}~{a['category_instance_id']}~{a['id']}~{file}.json") and OVERWRITE_POSE == 0:
                # print(f"File already exists: {a['id']} - {file}")
                with open(f"{folder_dir}/extracted_pose/{a['category']}~{a['category_instance_id']}~{a['id']}~{file}.json") as f:
                    bone_angle_annotations[a['category']][a['category_instance_id']][a['id']][file] = json.load(f)
                continue

            frame = Image.open(f"{folder_dir}/color/{file}")
            frame_shape = np.asarray(frame).shape

            x1 = a['x']
            y1 = a['y']
            w = a['width']
            h = a['height']

            if w is None or h is None:
                print(f"Null height/width: {file}")
                continue

            x1 -= 40
            y1 -= 40
            w += 80
            h += 80

            if x1 < 0:
                w = w + x1
                x1 = 0
            if y1 < 0:
                h = h + y1
                y1 = 0

            if x1 + w > frame_shape[1]:
                w = frame_shape[1] - x1
            if y1 + h > frame_shape[0]:
                h = frame_shape[0] - y1

            frame = np.asarray(frame.crop((x1, y1, x1+w, y1+h)))

            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            vertical_pad = max(frame_width-frame_height,0)
            horizontal_pad = max(frame_height-frame_width,0)

            frame = cv2.copyMakeBorder(frame, vertical_pad//2, vertical_pad//2, horizontal_pad//2, horizontal_pad//2, cv2.BORDER_CONSTANT, 0)
            resized_frame = tf.image.resize_with_pad([frame], INPUT_SIZE, INPUT_SIZE)

            keypoints = movenet(resized_frame)[0][0]

            # plt.imshow(frame)
            # for keypoint in keypoints:
            #     if keypoint[2] < 0.2: continue
            #     print(keypoint)
            #     print(frame.shape)
            #     plt.scatter([keypoint[1]*frame.shape[0]],[keypoint[0]*frame.shape[1]],c='green',s=10)
            # plt.show()

            new_keypoints = []
            for keypoint in keypoints:
                new_keypoints.append(
                    [
                        keypoint[0]*frame.shape[1],
                        keypoint[1]*frame.shape[0]
                    ]
                )
            keypoints = new_keypoints

            bone_angles = []

            for bone_connection in BONE_CONNECTIONS:

                j1 = keypoints[bone_connection[0]]
                j2 = keypoints[bone_connection[1]]
                j3 = keypoints[bone_connection[2]]

                bone_vector_1 = [j1[0] - j2[0], j1[1] - j2[1]]
                bone_vector_2 = [j3[0] - j2[0], j3[1] - j2[1]]

                if np.linalg.norm(bone_vector_1) == 0.0 or np.linalg.norm(bone_vector_2) == 0.0:
                    bone_angles.append(None)

                bone_unit_vector_1 = bone_vector_1 / np.linalg.norm(bone_vector_1)
                bone_unit_vector_2 = bone_vector_2 / np.linalg.norm(bone_vector_2)

                #this is using the following formula: atan2( ax*by-ay*bx, ax*bx+ay*by ).
                bone_angle = np.degrees(np.arctan2(
                    bone_unit_vector_2[0] * bone_unit_vector_1[1] - bone_unit_vector_2[1] * bone_unit_vector_1[0],
                    bone_unit_vector_2[0] * bone_unit_vector_1[0] - bone_unit_vector_2[1] * bone_unit_vector_1[1]
                ))

                bone_angles.append([
                    bone_angle,
                    np.linalg.norm(bone_vector_1),
                    np.linalg.norm(bone_vector_2)
                ])

            bone_angle_annotations[a['category']][a['category_instance_id']][a['id']][file] = bone_angles

            with open(f"{folder_dir}/extracted_pose/{a['category']}~{a['category_instance_id']}~{a['id']}~{file}.json", 'w') as f:
                json.dump(bone_angles, f)

        # pbar.update(1)

    for category in bone_angle_annotations.keys():

        for instance_id in bone_angle_annotations[category].keys():

            for person_id in bone_angle_annotations[category][instance_id].keys():

                frame_ids = list(bone_angle_annotations[category][instance_id][person_id].keys())

                data = []

                for j in range(len(BONE_CONNECTIONS)):

                    bone_frames_data = []
                    
                    for i in range(1, len(frame_ids)):

                        angle_changes = []

                        for k in [1,5,10]:

                            frame_id = frame_ids[i]

                            if k > i:
                                angle_changes.append(0)
                                continue

                            prev_frame_id = frame_ids[i-k]

                            if bone_angle_annotations[category][instance_id][person_id][frame_id] is None:
                                angle_changes.append(0)
                                continue

                            bone_data = bone_angle_annotations[category][instance_id][person_id][frame_id][j]

                            if bone_angle_annotations[category][instance_id][person_id][prev_frame_id] is None:
                                prev_bone_data = None
                                l = i - (k + 1)
                                while l > 0:
                                    prev_frame_id = frame_ids[l]
                                    if not bone_angle_annotations[category][instance_id][person_id][prev_frame_id] is None:
                                        prev_bone_data = bone_angle_annotations[category][instance_id][person_id][prev_frame_id][j]
                                        break
                                if prev_bone_data is None:
                                    angle_changes.append(0)
                                    continue
                            else:
                                prev_bone_data = bone_angle_annotations[category][instance_id][person_id][prev_frame_id][j]

                            angle_diff = bone_data[0] - prev_bone_data[0]

                            if angle_diff < -180:
                                angle_diff = angle_diff + 360
                            elif angle_diff > 180:
                                angle_diff = angle_diff - 360

                            angle_diff = angle_diff / 180

                            angle_changes.append(angle_diff)
                        
                        bone_frames_data.append(angle_changes)
                    
                    data.append(bone_frames_data)

                # if category == '': new_category = 'bg'
                if category == '': continue
                else: new_category = category

                if not os.path.exists(f"{folder_dir}/processed_extracted_pose_new"):
                    print(f"Making dir: {folder_dir}/processed_extracted_pose_new...")
                    os.mkdir(f"{folder_dir}/processed_extracted_pose_new")

                if not os.path.exists(f"{folder_dir}/processed_extracted_pose_new/{new_category}~{instance_id}~{person_id}.json") or OVERWRITE_REP == 1:
                    with open(f"{folder_dir}/processed_extracted_pose_new/{new_category}~{instance_id}~{person_id}.json", 'w') as f:
                        json.dump(data, f)
                else:
                    print(f"File already esists: {new_category}~{instance_id}~{person_id}.json")

                print(np.asarray(data).shape)

                data_summary.append({
                    "category":new_category,
                    "instance_id":instance_id,
                    "folder":folder,
                    "person_id":person_id,
                    "shape":np.asarray(data).shape
                })

pd.DataFrame(data=data_summary).to_csv(f"{DRIVE_DIR}/data_summary_v2.csv", index_label=False)
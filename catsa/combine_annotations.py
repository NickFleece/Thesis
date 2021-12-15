import json
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True)
args = parser.parse_args()

ALL_FRAMES_DIR = {
    "home":"H:/CATSA/CATSA_FRAMES",
    "wrnch":"/nvme_com_dat/CATSA/CATSA_DATA_SEGMENTED"
}
FRAMES_DIR = ALL_FRAMES_DIR[args.env]

ACTIVITIES = {
    "background":0,
    "roof-top-sides":1,
    "underside":2,
    "hood-front_exterior-engine":3,
    "trunk-rear_exterior":4,
    "front-passenger_compartment":5,
    "rear-passenger_compartment":6
}

csv_annotations = pd.DataFrame({
    "path_to_video_segment":[],
    "start_frame":[],
    "end_frame":[],
    "enclosing_bbox":[],
    "activity_class_id":[],
    "activity_class_name":[]
})

for date in os.listdir(FRAMES_DIR):
    json_files = [f for f in os.listdir(f"{FRAMES_DIR}/{date}") if '.json' in f]
    for json_file in json_files:
        with open(f"{FRAMES_DIR}/{date}/{json_file}") as f:
            json_annotations = json.load(f)

        annotation_actions = {}
        json_frame_annotations = json_annotations.keys()
        for frame_annotation in json_frame_annotations:

            if len(json_annotations[frame_annotation]['annotations']) > 1:
                print("wtf?")
                print(json_annotations[frame_annotation]['annotations'])
                continue
            if len(json_annotations[frame_annotation]['annotations']) == 0:
                continue

            category = json_annotations[frame_annotation]['annotations'][0]['category']
            category_instance_id = int(json_annotations[frame_annotation]['annotations'][0]['category_instance_id'])

            if not category in annotation_actions.keys():
                annotation_actions[category] = {category_instance_id: [json_annotations[frame_annotation]]}
            elif not category_instance_id in annotation_actions[category].keys():
                annotation_actions[category][category_instance_id] = [json_annotations[frame_annotation]]
            else:
                annotation_actions[category][category_instance_id].append(json_annotations[frame_annotation])

        for category in annotation_actions.keys():
            for instance in annotation_actions[category].keys():

                first_image = 99999
                last_image = 0
                min_x = 99999
                max_x = 0
                min_y = 99999
                max_y = 0

                for a in annotation_actions[category][instance]:
                    a_annotation = a['annotations'][0]

                    name = int(a['name'][:-4])

                    if name < first_image:
                        first_image = name
                    
                    if name > last_image:
                        last_image = name

                    x_1 = int(a_annotation['x'])
                    y_1 = int(a_annotation['y'])
                    x_2 = x_1 + int(a_annotation['width'])
                    y_2 = x_1 + int(a_annotation['height'])

                    if x_1 < min_x:
                        min_x = x_1
                    if y_1 < min_y:
                        min_y = y_1

                    if x_2 > max_x:
                        max_x = x_2
                    if y_2 > max_y:
                        max_y = y_2

                #debugging purposes
                if first_image == last_image:
                    print("\n")
                    print(first_image)
                    print(last_image)
                    print(annotation_actions[category].keys())
                    print("Class only has one image!")
                    print(date)
                    print(json_file)
                    print(category)

                csv_annotations = csv_annotations.append(pd.DataFrame({
                    "path_to_video_segment":[f"{date}/{json_file[:-5]}"],
                    "start_frame":[f"{str(first_image).zfill(5)}.jpg"],
                    "end_frame":[f"{str(last_image).zfill(5)}.jpg"],
                    "enclosing_bbox":[[x_1, y_1, x_2, y_2]],
                    "activity_class_id":[ACTIVITIES[category]],
                    "activity_class_name":[category]
                }), ignore_index=True)
                
csv_annotations.to_csv(f"{FRAMES_DIR}/annotations.csv")
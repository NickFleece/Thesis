import json
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--frames_folder', required=True)
args = parser.parse_args()

FRAMES_DIR = args.frames_folder


ALL_DATES = [
    #0
    "y2021m07d19h19m00s00",
    "y2021m07d21h18m00s00",
    "y2021m07d27h07m00s00",
    "y2021m07d28h08m00s00",
    "y2021m07d28h10m00s00",
    "y2021m07d29h08m00s00",
    "y2021m07d30h13m00s00",
    "y2021m07d30h19m00s00",
    "y2021m08d02h06m00s00",
    "y2021m08d02h13m00s00",

    #1
    "y2021m11d08h08m00s00",
    "y2021m11d09h08m00s00",
    "y2021m11d09h20m00s00",
    "y2021m11d10h19m00s00",
    "y2021m11d12h07m00s00",
    "y2021m11d13h19m00s00",
    "y2021m11d15h09m00s00",
    "y2021m11d16h11m00s00",
    "y2021m11d17h07m00s00",
    "y2021m11d17h11m00s00",
    
    #2
    "y2021m12d13h16m00s00",
    "y2021m12d14h13m00s00",
    "y2021m12d15h18m00s00",
    "y2021m12d16h12m00s00",
    "y2021m12d17h08m00s00",
    "y2021m12d18h11m00s00",
    "y2021m12d19h15m00s00",
    "y2021m12d20h06m00s00",
    "y2021m12d21h13m00s00",
    "y2021m12d22h10m00s00",
    "y2021m12d23h19m00s00",
    "y2021m12d24h14m00s00",
    "y2021m12d27h17m00s00",
    "y2021m12d28h06m00s00"
]

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

if os.path.exists(f"{FRAMES_DIR}/annotations.csv"):
    os.remove(f"{FRAMES_DIR}/annotations.csv")

for date in ALL_DATES:
    json_files = [f for f in os.listdir(f"{FRAMES_DIR}/{date}") if '.json' in f]
    for json_file in json_files:
        with open(f"{FRAMES_DIR}/{date}/{json_file}") as f:
            json_annotations = json.load(f)

        annotation_actions = {}
        json_frame_annotations = json_annotations.keys()
        for frame_annotation in json_frame_annotations:

            if len(json_annotations[frame_annotation]['annotations']) > 1:
                print("--------")
                print("Two annotations on one frame?")
                print(date)
                print(json_file)
                
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
                    y_2 = y_1 + int(a_annotation['height'])

                    if int(a_annotation['width']) < 15 or int(a_annotation['height']) < 15: continue

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
                    print("--------")
                    print("Class only has one image!")
                    print("frame: " + str(first_image))
                    print(date)
                    print(json_file)
                    print(category)
                    continue

                csv_annotations = csv_annotations.append(pd.DataFrame({
                    "path_to_video_segment":[f"{date}/{json_file[12:-5]}"],
                    "camera":[json_file[12:14]],
                    "start_frame":[f"{str(first_image).zfill(5)}.jpg"],
                    "end_frame":[f"{str(last_image).zfill(5)}.jpg"],
                    "enclosing_bbox":f"{min_x} {min_y} {max_x} {max_y}",
                    "activity_class_id":[ACTIVITIES[category]],
                    "activity_class_name":[category]
                }), ignore_index=True)
                
csv_annotations.to_csv(f"{FRAMES_DIR}/annotations.csv")
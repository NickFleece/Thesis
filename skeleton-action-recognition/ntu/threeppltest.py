import os
import json
from tqdm import tqdm
from ntu_skeleton_config import *

triple_files = []
for json_file in tqdm(os.listdir("H:/ntu_processed_skeleton_data")):
    with open(f"H:/ntu_processed_skeleton_data/{json_file}") as f:
        skeleton_json = json.load(f)
    
    if len(skeleton_json.keys()) > 2: 
        print(json_file)
        triple_files.append(json_file)

with open("f_files.json", 'w') as f:
    json.dump(triple_files, f)
import json
import numpy as np
import cv2
from tqdm import tqdm # type: ignore
import os
import shutil
from glob import glob

labels = set({})

colors = {
        'artery': 80,
        'blood-flow': 180,
        'calcified-plaque': 255,
        'non-calcified-plaque': 80,
        "calcification": 255,
        "flow": 180,
        "plaque": 80,
        "thrombus": 80,
        "celiac trunk": 180,
        "dissection": 80,
        "abdominal aorta": 180,
        "splenogastric trunk": 180,
    }

def create_mask(annotations, name, image_height, image_width, to_save):
    global labels
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for annotation in annotations[::-1]:
        label = annotation['label']
        labels.add(label)
        points = np.array(annotation['points'], dtype=np.int32)
        if label in colors.keys():
            cv2.fillPoly(mask, [points], color=colors[label])
        elif label.split(" ")[-1] == 'artery':
            cv2.fillPoly(mask, [points], color=180)

    cv2.imwrite(f'{to_save}/{name}_mask.png', mask)
    
def main(dataset_path):
    dataset_path = dataset_path+"/" if dataset_path[-1] != "/" else dataset_path

    # Reading all images and labels of dataset
    json_files = glob(dataset_path+"*.json")
    images_names = glob(dataset_path+"*.png")

    # Sorting the dataset
    json_files.sort()
    images_names.sort()
    
    # Assertions to handle inputs
    assert len(json_files) != 0, "No labeling file found!"
    assert len(json_files) == len(images_names), "Images & labels must be of same length"

    # Save to destination set
    save_to = "/".join(dataset_path.split("/")[:-2])+"/MASKS/"

    # Make directory to save resultant masks
    os.makedirs(save_to, exist_ok=True)

    for i in tqdm(json_files):
        JSON = json.load(open(i))
        create_mask(JSON['shapes'], name = i.split('/')[-1].replace(".json",""), image_height=512, image_width=512, to_save=save_to)

    print(f"Masks saved at {save_to}")

    print("\n\n")
    for i in labels:
        if i not in colors.keys() and i.split(" ")[-1] != 'artery':
            print(i)

if __name__ == "__main__":
    for dataset_path in glob("DATASET/*/PNG_IMAGES"):
        try:
            main(dataset_path)
        except Exception as e:
            print(f"Error {e} While Processing ~ {dataset_path}")

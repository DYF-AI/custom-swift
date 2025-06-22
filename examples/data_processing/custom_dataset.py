import json
import os.path
from tqdm import tqdm

from modelscope.msdatasets import MsDataset
train_ds = MsDataset.load("coco_2014_caption", namespace="modelscope", split="train")
print(train_ds[0])
val_ds = MsDataset.load("coco_2014_caption", namespace="modelscope", split="validation")
print(val_ds[0])

train_num, val_num = 2000, 500

#save_path = "/mnt/n/dataset/coco_2014_caption"
save_path = "/mnt/g/dongyongfei786/custom-swift/examples/data_processing/output"
train_file = f"{save_path}/train_with_angle.jsonl"
val_file = f"{save_path}/val_with_angle.jsonl"

with open(train_file, "w", encoding="utf-8") as f1:
    for idx, row_data in enumerate(tqdm(train_ds)):
        print(row_data)
        save_image_path = f"{save_path}/train/{row_data['image_id']}.jpg"
        if not os.path.exists(os.path.dirname(save_image_path)):
            os.makedirs(os.path.dirname(save_image_path))
        image = row_data["image"]
        image.save(save_image_path)
        if idx >= train_num:
            break
        new_row_data = {
            "query": "please describe the image.",
            "response": row_data["caption"],
            "images": [save_image_path],
            "angles": [0]
        }
        print(new_row_data)
        f1.write(json.dumps(new_row_data, ensure_ascii=False) + "\n")
        f1.flush()

with open(val_file, "w", encoding="utf-8") as f2:
    for idx, row_data in enumerate(tqdm(val_ds)):
        print(row_data)
        save_image_path = f"{save_path}/validation/{row_data['image_id']}.jpg"
        if not os.path.exists(os.path.dirname(save_image_path)):
            os.makedirs(os.path.dirname(save_image_path))
        image = row_data["image"]
        image.save(save_image_path)
        if idx >= val_num:
            break
        new_row_data = {
            "query": "Please describe the image in detail.",
            "response": row_data["caption"],
            "images": [save_image_path],
            "angles": [0]
        }
        print(new_row_data)
        f2.write(json.dumps(new_row_data, ensure_ascii=False) + "\n")
        f2.flush()

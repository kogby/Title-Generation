import json
import random
import argparse

parser = argparse.ArgumentParser(
    description="Split train.jsonl into training and validation sets"
)
parser.add_argument(
    "--data_path", type=str, required=True, help="Path to the input train.jsonl file"
)
parser.add_argument(
    "--split_ratio",
    type=float,
    default=0.85,
    help="Split ratio for training and validation (default is 0.8)",
)
args = parser.parse_args()

input_file = args.data_path
split_ratio = args.split_ratio

# 讀取 train.jsonl 檔案
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 隨機打亂資料順序
random.shuffle(lines)

# 切分資料
split_point = int(len(lines) * split_ratio)
training_data = lines[:split_point]
validation_data = lines[split_point:]

# 將資料寫入 training.jsonl 和 validation.jsonl
with open("./data/training.jsonl", "w", encoding="utf-8") as f_train:
    for line in training_data:
        f_train.write(line)

with open("./data/validation.jsonl", "w", encoding="utf-8") as f_val:
    for line in validation_data:
        f_val.write(line)

print(f"完成切分，訓練資料 {len(training_data)} 條，驗證資料 {len(validation_data)} 條")

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
    help="Split ratio for training and validation (default is 0.85)",
)
args = parser.parse_args()

input_file = args.data_path
split_ratio = args.split_ratio

# 讀取 train.jsonl 檔案
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 解析每一行為 JSON 物件
json_data = [json.loads(line) for line in lines]

# 隨機打亂資料順序
random.seed(0)
random.shuffle(json_data)

# 切分資料
split_point = int(len(json_data) * split_ratio)
training_data = json_data[:split_point]
validation_data = json_data[split_point:]

# 將資料寫入 training.json 和 validation.json
with open("./data/training.json", "w", encoding="utf-8") as f_train:
    json.dump(training_data, f_train, ensure_ascii=False, indent=4)

with open("./data/validation.json", "w", encoding="utf-8") as f_val:
    json.dump(validation_data, f_val, ensure_ascii=False, indent=4)

print(f"完成切分，訓練資料 {len(training_data)} 條，驗證資料 {len(validation_data)} 條")

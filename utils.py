import json
import random
import torch
import numpy as np


def read_jsonl(file_path: str) -> list:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl(data_list: list, path: str) -> None:
    with open(path, "w") as fp:
        for data in data_list:
            fp.write(json.dumps(data, ensure_ascii=False))
            fp.write("\n")
    return


def init_random_seeds(seed: int = 0) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return

import torch

import nltk
from tqdm import tqdm
from argparse import Namespace, ArgumentParser
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
from utils import (
    init_random_seeds,
    read_jsonl,
    write_jsonl,
)

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 64


class NewsDataset(Dataset):
    def __init__(self, data_list, transform=False):
        self.data_list = (
            [transform(data) for data in tqdm(data_list)]
            if transform is not None
            else data_list
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if k in ["title", "id"] else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict


def preprocess_tokenize(data, tokenizer, train: bool = True):
    tokenized_data = tokenizer(
        data["maintext"],
        max_length=MAX_SOURCE_LEN,
        padding="max_length",
        truncation=True,
    )
    tokenized_data["id"] = data["id"]

    if train:
        label = tokenizer(
            text_target=data["title"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
        )["input_ids"]
        label = [(l if l != tokenizer.pad_token_id else -100) for l in label]
        tokenized_data["title"] = data["title"]
        tokenized_data["labels"] = label

    return tokenized_data


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Title Summarization")
    parser.add_argument("--data_path", type=str, default="data/public.jsonl")
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="best_checkpoint",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="best_checkpoint",
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--output_path", type=str, default="submission.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    init_random_seeds()
    args = parse_arguments()

    # Dataset Preprcoessing
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=True, trust_remote_code=False
    )
    test_data_list = read_jsonl(args.data_path)
    preprocess_tokenize = partial(preprocess_tokenize, tokenizer=tokenizer, train=False)
    test_dataset = NewsDataset(test_data_list, preprocess_tokenize)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=collate_func, shuffle=False
    )

    # Load Model, and Prepare for Inference
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    )
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        trust_remote_code=False,
    ).to(device)

    sampling_params = {
        "do_sample": args.do_sample,
    }

    # Set Sampling Parameters
    if args.do_sample:
        if args.top_p > 0:
            sampling_params["top_p"] = args.top_p
        if args.top_k > 0:
            sampling_params["top_k"] = args.top_k
        if args.temperature > 0:
            sampling_params["temperature"] = args.temperature
    # print(sampling_params)

    model.eval()
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    for _, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            batch_data = {
                k: v.to(device) if not isinstance(v, list) else v
                for k, v in batch_data.items()
            }
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=MAX_TARGET_LEN,
                num_beams=args.num_beams,
                **sampling_params,
            )
            generations = [
                "\n".join(nltk.sent_tokenize(data.strip()))
                for data in tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
            ]
            prediction_list.extend(
                [
                    {"title": pred, "id": ID}
                    for ID, pred in zip(batch_data["id"], generations)
                ]
            )
    write_jsonl(prediction_list, args.output_path)

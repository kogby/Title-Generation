#!/bin/bash
# Reference https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization

python preprocess.py --data_path ./data/train.jsonl 

python train.py --train_file ./data/training.jsonl \
				--validation_file ./data/validation.jsonl \
                --num_beams 5 \
                --model_name_or_path best_checkpoint \
				--tokenizer_name best_checkpoint \
                --per_device_train_batch_size 8 \
                --learning_rate 1e-3 \
				--num_train_epochs 15 \
				--gradient_accumulation_steps 4 \
                --text_column 'maintext' \
				--summary_column 'title' \
				--num_warmup_steps 0 \
                --output_dir ./train_checkpoint \
                --with_tracking \
				--ignore_pad_token_for_loss True \
				--max_source_length 256 \
				--max_target_length 64

# --model_name_or_path google/mt5-small \
# --tokenizer_name google/mt5-small \
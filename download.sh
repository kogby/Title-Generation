#!/bin/bash
FILE_ID_CHECKPOINTS="1CWbJsTPJ21MoKwCyshnBrtr6EcP-1_Js"

# 定義下載檔案的輸出路徑
OUTPUT_CHECKPOINTS="best_checkpoint.zip"

# Download ntlk (punkt)
echo "正在下載 punkt..."
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo "下載完成！"

if [ ! -d data ]; then
	echo "正在解壓 data.zip..."
	unzip data.zip
fi

if [ ! -d best_checkpoint ]; then
    # 下載 best_model_checkpoints.zip
    echo "正在下載 best_checkpoint.zip..."
    gdown "https://drive.google.com/uc?id=$FILE_ID_CHECKPOINTS" -O $OUTPUT_CHECKPOINTS
    echo "下載完成！"
	echo "正在解壓 best_checkpoint.zip..."
    unzip best_checkpoint.zip
fi
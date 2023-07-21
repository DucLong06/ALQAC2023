#!/bin/bash

MODEL="keepitreal/vietnamese-sbert"
INPUT_QUESTIONS="data/raw/V1.1/train.json"
INPUT_ARTICLES="data/training/all_articles_2023.json"
TOP_BM25=20
BATCH_SIZE=128
MAX_EPOCHS=5

python train.py \
    --base_model "$MODEL" \
    --input_questions "$INPUT_QUESTIONS" \
    --input_articles "$INPUT_ARTICLES" \
    --top_bm25 "$TOP_BM25" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS"

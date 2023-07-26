#!/bin/bash

MODEL="sentence-transformers/paraphrase-xlm-r-multilingual-v1"
INPUT_QUESTIONS="data/raw/V1.1/train.json"
INPUT_ARTICLES="data/training/all_articles_2023.json"
TOP_BM25=10
BATCH_SIZE=1
MAX_EPOCHS=30

python train.py \
    --base_model "$MODEL" \
    --input_questions "$INPUT_QUESTIONS" \
    --input_articles "$INPUT_ARTICLES" \
    --top_bm25 "$TOP_BM25" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS"

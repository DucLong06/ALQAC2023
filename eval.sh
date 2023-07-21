#!/bin/bash

MODEL="models/keepitreal-vietnamese-sbert_20230722010521.pth"
INPUT_QUESTIONS="data/raw/V1.1/train.json"
INPUT_ARTICLES="data/training/all_articles_2023.json"
COMPARE=True
ALPHA=0.7
TOP_ARTICLES=20

python evaluate.py \
    --model "$MODEL" \
    --input_questions "$INPUT_QUESTIONS" \
    --input_articles "$INPUT_ARTICLES" \
    --compare "$COMPARE" \
    --alpha "$ALPHA" \
    --top_articles "$TOP_ARTICLES"


#15 ngày


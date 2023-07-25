#!/bin/bash

MODEL="models/F_sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230724040020.pth"
INPUT_QUESTIONS="data/private_test.json"
INPUT_ARTICLES="data/training/all_articles_2023.json"
COMPARE="False"
ALPHA=0.4
TOP_ARTICLES=60

python evaluate.py \
    --model "$MODEL" \
    --input_questions "$INPUT_QUESTIONS" \
    --input_articles "$INPUT_ARTICLES" \
    --compare "$COMPARE" \
    --alpha "$ALPHA" \
    --top_articles "$TOP_ARTICLES"


#15 ngày


#!/bin/bash

MODEL="model_new.pth"
INPUT_QUESTIONS="path/to/input_questions"
INPUT_ARTICLES="path/to/input_articles"
COMPARE=True
ALPHA=0
TOP_ARTICLES=1

python evaluate.py \
    --model "$MODEL" \
    --input_questions "$INPUT_QUESTIONS" \
    --input_articles "$INPUT_ARTICLES" \
    --compare "$COMPARE" \
    --alpha "$ALPHA" \
    --top_articles "$TOP_ARTICLES"

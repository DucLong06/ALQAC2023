#!/bin/bash
MODEL="google/flan-t5-xxl"
INPUT_QUESTIONS="data/training/gg_question_train.json"
INPUT_ARTICLES="data/training/gg_all_articles_2023.json"
PROMPTS="prompts/prompts.json"
COMPARE="False" 

python prompt.py \
    --model "$MODEL" \
    --questions "$INPUT_QUESTIONS" \
    --articles "$INPUT_ARTICLES" \
    --prompts "$PROMPTS" \
    --compare "$COMPARE" \

import os
import asyncio
import concurrent.futures
from bot_telegram import send_message
import my_env
import evaluate
import my_logger
import train

logger = my_logger.Logger("training", my_env.LOG)
models = my_env.list_base_model
input_q_train = my_env.PATH_TO_PUBLIC_TRAIN
input_a_train = my_env.PATH_TO_CORPUS_2023
input_all_q = my_env.PATH_TO_QUESTION_ALL
input_all_a = my_env.PATH_TO_CORPUS_ALL
top_fake_bm25 = 20
batch_size = 128
max_epochs = 25


def train_model(model, input_questions, input_articles, top_fake_bm25, batch_size, max_epochs):
    try:
        index = models.index(model)
        if index == 0:
            input_questions = input_all_q
            input_articles = input_all_a
        logger.info(f"Training base model: {model}")
        logger.info(f"Input Questions (Train): {input_questions}")
        logger.info(f"Input Articles (Train): {input_articles}")
        logger.info(f"Top Fake BM25: {top_fake_bm25}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Max Epochs: {max_epochs}")
        train.train(model, input_questions, input_articles,
                    top_fake_bm25, batch_size, max_epochs)
        logger.info(f"Done: {model}")
    except Exception as e:
        logger.error(f"{model}: {str(e)}")
        asyncio.run(send_message(f"base model error : {model}"))


def main():
    for model in models:
        input_questions = input_q_train
        input_articles = input_a_train
        train_model(model, input_questions, input_articles,
                    top_fake_bm25, batch_size, max_epochs)


if __name__ == "__main__":
    main()

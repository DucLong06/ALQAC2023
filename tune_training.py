import asyncio
import os
import concurrent.futures
import src.utils.my_env as my_env
import src.utils.my_logger as my_logger
import src.retrieval.train as train
from src.utils.bot_telegram import send_message

logger = my_logger.Logger("training", my_env.LOG)
models = my_env.list_base_model
input_q_train = my_env.PATH_TO_QUESTION_F
input_a_train = my_env.PATH_TO_CORPUS_2023
# input_all_q = my_env.PATH_TO_QUESTION_ALL
# input_all_a = my_env.PATH_TO_CORPUS_ALL
top_fake_bm25 = 20
batch_size = 32
max_epochs = 20


def train_model(model, input_questions, input_articles, top_fake_bm25, batch_size, max_epochs):
    try:
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


def train_model_wrapper(args):
    train_model(*args)


def main():
    num_threads = 1
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        model_args = [(model, input_q_train, input_a_train,
                       top_fake_bm25, batch_size, max_epochs) for model in models]
        executor.map(train_model_wrapper, model_args)


if __name__ == "__main__":
    main()
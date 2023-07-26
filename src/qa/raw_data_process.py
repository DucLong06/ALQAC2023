import json
import random
import re
import pandas as pd
from tqdm import tqdm
import my_env
from src.utils.rank_bm25 import BM25Okapi

import src.utils.my_logger as my_logger


logger = my_logger.Logger("raw_data_qa", my_env.LOG)


def data_qa_generator(path_json_question: str, path_json_law: str,  top_bm25: int = 10, train_ratio=0.8, val_ratio=0.1):
    with open(path_json_law, 'r') as file:
        data_corpus = json.load(file)

    with open(path_json_question, 'r') as file:
        data_question = json.load(file)

    logger.info(f"Number question: {len(data_question)}")
    logger.info(f"Number corpus: {len(data_corpus)}")

    corpus = list(data_corpus.values())

    bm25 = BM25Okapi(corpus)

    random.shuffle(data_question)

    train_size = int(len(data_question) * train_ratio)
    val_size = int(len(data_question) * val_ratio)

    train_data = data_question[:train_size]
    val_data = data_question[train_size:train_size + val_size]
    test_data = data_question[train_size + val_size:]

    def generate_data(data):
        for item in data:
            question = item['text']
            list_question = []
            if "choices" in item:
                if "answer" in item:
                    pattern = r'^Cả'
                    if re.search(pattern, item['choices'][item["answer"]], re.IGNORECASE):
                        true_answer = re.findall(
                            r"[A-D]", item['choices'][item["answer"]].replace("Cả", ""))
                        if true_answer:
                            for ans in true_answer:
                                list_question.append("{}\n{}".format(
                                    question, item['choices'][ans]))
                        else:
                            del item['choices'][item["answer"]]
                            for ans in item['choices']:
                                list_question.append("{}\n{}".format(
                                    question, item['choices'][ans]))
                    else:
                        list_question.append("{}\n{}".format(
                            question,  item['choices'][item["answer"]]))
            else:
                list_question.append(question)

    train_df = pd.DataFrame(generate_data(train_data))
    val_df = pd.DataFrame(generate_data(val_data))
    test_df = pd.DataFrame(generate_data(test_data))

    logger.info(f"Number of data in train set: {len(train_df)}")
    logger.info(f"Number of data in val set: {len(val_df)}")
    logger.info(f"Number of data in test set: {len(test_df)}")

    return train_df, val_df, test_df


data_qa_generator(my_env.PATH_TO_PUBLIC_TRAIN, my_env.PATH_TO_CORPUS_2023)

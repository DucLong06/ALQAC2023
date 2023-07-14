import json
import pandas as pd
import torch
from tqdm import tqdm
from model_paraformer import Model_Paraformer
import my_env
from processing_data import word_segment

from rank_bm25 import BM25Plus

import pickle


def _data_generator(path_json: str):
    with open(path_json, 'r') as file:
        data = json.load(file)

    for item in data:
        for article in item['articles']:
            id_text = f"{item['id']}@{article['id']}"
            # text = item['id']+" " + article['text']
            text = article['text']
            yield {"id": id_text, "article": text}


def df_create_corpus(path_json: str) -> pd.DataFrame:
    list_data = _data_generator(path_json)
    return pd.DataFrame(list_data)


def _question_generator(path_json: str):
    with open(path_json, 'r') as file:
        data = json.load(file)

    for item in data:
        question = item['text']
        relevant_articles = item['relevant_articles'][:2]
        for relevant_article in relevant_articles:
            law_id = relevant_article['law_id']
            article_id = relevant_article['article_id']
            yield {'question': question, 'relevant_id': f"{law_id}@{article_id}"}


def df_create_questions_test(path_json: str) -> pd.DataFrame:
    list_data = _question_generator(path_json)
    return pd.DataFrame(list_data)


def _data_training_generator(path_json_question: str, path_json_law: str, top_bm25: int = 10):
    with open(path_json_law, 'r') as file:
        data_corpus = json.load(file)

    with open(path_json_question, 'r') as file:
        data_question = json.load(file)

    corpus = list(data_corpus.values())

    bm25 = BM25Plus(corpus)

    for item in data_question:
        question = item['text']
        relevant_articles = item['relevant_articles']
        neg_list = bm25.get_top_n(
            question.split(" "), corpus, n=top_bm25)

        for relevant_article in relevant_articles:
            corpus_id = relevant_article['law_id'] + \
                "@" + relevant_article['article_id']
            if corpus_id in data_corpus.keys():
                for _ in range(top_bm25 // 2):
                    yield {
                        "question": question,
                        "article": [sentence.strip() for sentence in data_corpus[corpus_id].split("\n") if sentence.strip() != ""],
                        "relevant": 1
                    }
                neg_list = [neg for neg in neg_list
                            if neg != data_corpus[corpus_id]]

        for neg in neg_list:
            yield {
                "question": question,
                "article": [sentence.strip() for sentence in neg.split("\n") if sentence.strip() != ""],
                "relevant": 0
            }


def df_create_data_training(path_json_question: str, path_json_law: str, top_bm25: int = 10) -> pd.DataFrame:
    df = pd.DataFrame(_data_training_generator(
        path_json_question, path_json_law, top_bm25))
    return df


def _convert_all_law_to_json(*file_paths):
    result_json = {}

    for path in file_paths:
        with open(path, 'r') as file:
            data = json.load(file)

        for item in tqdm(data):
            for article in item['articles']:
                id_text = f"{item['id']}@{article['id']}"
                text = article['text']
                result_json[id_text] = text

    json_data = json.dumps(result_json, ensure_ascii=False)
    output_file = 'all_articles_2023.json'
    with open(output_file, 'w') as file:
        file.write(json_data)


def merge_json_files(*file_paths):
    merged_data = []

    for path in file_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)

    output_file = 'all_articles.json'
    with open(output_file, 'w') as file:
        json.dump(merged_data, file, ensure_ascii=False)

    print("Merge completed. Output saved to", output_file)


_convert_all_law_to_json(
    "/Users/longhoangduc/Library/CloudStorage/GoogleDrive-hoangduclongg@gmail.com/My Drive/Colab Notebooks/Task1/data/raw/V1.1/law.json")
#                  "/Users/longhoangduc/Library/CloudStorage/GoogleDrive-hoangduclongg@gmail.com/My Drive/Colab Notebooks/Task1/data/raw/V1.1/additional_data/zalo/zalo_corpus.json",
#                  "/Users/longhoangduc/Library/CloudStorage/GoogleDrive-hoangduclongg@gmail.com/My Drive/Colab Notebooks/Task1/data/raw/V1.1/additional_data/ALQAC_2022_training_data/law.json")

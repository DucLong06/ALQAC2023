import datetime
import json
import os
from typing import List
import torch
from tqdm import tqdm
from src.retrieval.evaluate import compare_json, get_top_n_articles
from src.retrieval.model_paraformer import Model_Paraformer
import src.utils.my_env as my_env
import pandas as pd

import src.utils.my_logger as my_logger
from src.retrieval.post_data import convert_ID

logger = my_logger.Logger("ensemble", my_env.LOG)


def read_model_from_path(path_to_model):
    for key_model, value_model in my_env.dict_bast_model.items():
        if key_model in path_to_model:
            model = Model_Paraformer(value_model)
            return model
    return None


def main(models: list, path_to_query: str, path_to_law: str, compare: bool, alpha: float, top_n: int):

    with open(path_to_query, 'r') as file:
        data_question = json.load(file)

    with open(path_to_law, 'r') as file:
        data_corpus = json.load(file)

    list_model_parformer: List[Model_Paraformer] = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for path_model in models:
        model = read_model_from_path(path_model).to(device)
        checkpoint = torch.load(path_model, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        list_model_parformer.append(model)

    output_data = gen_submit(list_model_parformer, data_question,
                             data_corpus, alpha=alpha, top_n=top_n)

    output_file = os.path.join(
        my_env.PATH_TO_SAVE_JSON, f'ensemble_[OUT]_{datetime.now().strftime("%d%m%Y")}.json')

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)
    logger.info(f"Processing complete. Output saved to {output_file}")

    if compare:
        compare_json(output_data, "Ensemble_model", alpha,
                     top_n, path_to_law, "Ensemble")


def gen_submit(models: List[Model_Paraformer], data_question, data_corpus, alpha, top_n):
    out_put = data_question
    results = []
    output_list = []
    for i, item in tqdm(enumerate(data_question)):
        question = item["text"]
        list_question = []
        total_choice = []
        if "choices" in item:
            for ans in item['choices']:
                list_question.append("{}\n{}".format(
                    question, item['choices'][ans]))
        else:
            list_question.append(question)

        for query in list_question:
            top_n_articles = get_top_n_articles(
                query, data_corpus, top_n=top_n)
            list_keys, list_articles, bm25_scores = zip(*top_n_articles)
            article_scores = {}
            for i, article in enumerate(list_articles):
                article = [sentence.strip() for sentence in article.split(
                    "\n") if sentence.strip() != ""]
                for model_index, model in enumerate(models):
                    deep_score = model.get_score(query, article)
                    deep_score = float(deep_score)
                    if f"model_{model_index}" not in article_scores:
                        article_scores[f"model_{model_index}"] = []
                    article_scores[f"model_{model_index}"].append(deep_score)

            max_scores = [{f"model_{model_index}": scores}
                          for model_index, scores in article_scores.items()]

            max_scores.append({"bm25": [float(x) for x in bm25_scores]})
            index_max_scores = {f"model_{model_index}": scores.index(
                max_score) for model_index, scores in article_scores.items() for max_score in [max(scores)]}

            index_max_scores["bm25"] = bm25_scores.index(max(bm25_scores))

            predict_relevents = {k: list_keys[v] if isinstance(
                v, int) else v for k, v in index_max_scores.items()}

            label = item['relevant_articles'][0]['law_id'] + \
                "@" + item['relevant_articles'][0]['article_id']

            result_json = {
                'query': query,
                'max_scores': max_scores,
                'predict_relevents': predict_relevents,
                'list_relevant_articles': list(list_keys),
                'label': label
            }
            output_list.append(result_json)
    output_file = "ensemble.json"
    file_name = "ensemble.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_list, f, ensure_ascii=False, indent=4)
    except:
        with open(file_name, 'w') as file:
            for item in output_list:
                file.write(str(item) + "\n")


list_model = ["models/keepitreal-vietnamese-sbert_20230722211525.pth",
              "models/sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230722135327.pth",
              "models/khanhpd2-sbert_phobert_large_cosine_sim_20230722133026.pth",]
main(list_model, my_env.PATH_TO_PUBLIC_TRAIN,
     my_env.PATH_TO_CORPUS_2023, True, 0.6, 20)

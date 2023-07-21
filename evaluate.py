import argparse
import asyncio
from collections import Counter, defaultdict
import glob
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from eval_metrics import calculate_accuracy, calculate_f2_score, calculate_precision, calculate_recall
import my_env
from bot_telegram import send_message, send_telegram_message

from model_paraformer import Model_Paraformer
from post_data import convert_ID
import pickle
from processing_data import word_segment
from rank_bm25 import BM25Okapi, BM25Plus

import my_logger
from train import train

logger = my_logger.Logger("evaluate", my_env.LOG)


def get_top_n_articles(query: str, data_corpus, top_n: int):
    corpus = list(data_corpus.values())
    tokenized_corpus = [word_segment(doc) for doc in corpus]
    tokenized_corpus = [doc.split(" ") for doc in tokenized_corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = word_segment(query)
    tokenized_query = tokenized_query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    top_n_articles = sorted(list(zip(data_corpus.keys(), data_corpus.values(), bm25_scores)),
                            key=lambda x: x[2], reverse=True)[:top_n]
    return top_n_articles


def gen_submit(model: Model_Paraformer, data_question, data_corpus, alpha, top_n):
    model.eval()
    out_put = data_question
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
            final_scores = []
            top_n_articles = get_top_n_articles(
                query, data_corpus, top_n=top_n)
            list_keys, list_articles, bm25_scores = zip(*top_n_articles)

            for i, article in enumerate(list_articles):
                article = [sentence.strip() for sentence in article.split(
                    "\n") if sentence.strip() != ""]
                deep_score = model.get_score(query, article)
                final_scores.append(alpha*deep_score+(1-alpha)*bm25_scores[i])
            total_choice.append(
                (np.max(final_scores), list_keys[np.argmax(final_scores)]))

        id_corpus = [id for _, id in total_choice]
        counter = Counter(id_corpus)
        most_common = counter.most_common()
        if len(most_common) in [2, 4]:
            _, id_corpus = max(total_choice, key=lambda x: x[0])
        else:
            id_corpus = most_common[0][0]
        # _, id_corpus = max(total_choice, key=lambda x: x[0])
        item.setdefault("relevant_articles", []).append(
            convert_ID(id_corpus))

    return out_put


def compare_json(data, path_to_model, alpha, top_n, path_to_law):
    correct_count = 0
    total_count = len(data)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    error_data = []
    for item in data:
        relevant_articles = item['relevant_articles']

        if relevant_articles[0] == relevant_articles[1]:
            correct_count += 1
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
            error_data.append(item)

    accuracy = calculate_accuracy(correct_count, total_count)
    precision = calculate_precision(true_positive, false_positive)
    recall = calculate_recall(true_positive, false_negative)
    f2_score = calculate_f2_score(precision, recall)

    logger.info(f'Accuracy: {accuracy}')
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F2 Score: {f2_score}')

    count_dict = defaultdict(int)
    for item in error_data:
        count_dict[item["question_type"]] += 1
    with open('error_data.json', 'w') as file:
        json.dump(error_data, file, ensure_ascii=False)

    try:
        asyncio.run(send_telegram_message(
            model_name="[Test] Paraformer",
            model_base=path_to_model,
            data_name=path_to_law,
            alpha=alpha,
            top_k_bm25=top_n,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f2=f2_score,
            note=str(count_dict)
        ))
    except Exception as e:
        logger.error(str(e))


def main(path_to_model: str, path_to_query: str, path_to_law: str, compare: bool, alpha: float, top_n: int):

    with open(path_to_query, 'r') as file:
        data_question = json.load(file)

    # with open(PATH_TO_BIN_CORPUS_ALL, 'rb') as file:
    #     data_corpus = pickle.load(file)

    with open(path_to_law, 'r') as file:
        data_corpus = json.load(file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model_Paraformer().to(device)

    checkpoint = torch.load(path_to_model, map_location=device)
    model.load_state_dict(checkpoint)

    output_data = gen_submit(model, data_question,
                             data_corpus, alpha=alpha, top_n=top_n)

    output_file = "output_train.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)
    logger.info(f"Processing complete. Output saved to {output_file}")

    if compare:
        compare_json(output_data, path_to_model, alpha, top_n, path_to_law)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default=my_env.PATH_TO_MODEL_PARAFORMER)
    parser.add_argument('--input_questions', type=str,
                        default=my_env.PATH_TO_PUBLIC_TRAIN)
    parser.add_argument('--input_articles', type=str,
                        default=my_env.PATH_TO_CORPUS_2023)
    parser.add_argument('--compare', type=bool, default=True)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--top_articles', type=int, default=10)

    opts = parser.parse_args()
    main(opts.model, opts.input_questions, opts.input_articles,
         opts.compare, opts.alpha, opts.top_articles)

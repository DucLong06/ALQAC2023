from collections import Counter
import os
import pickle
from rank_bm25 import *
import pandas as pd
import numpy as np
from post_data import *
from raw_data import *
from eval_metrics import *
from processing_data import *
from tqdm import tqdm
import my_env


PATH_TO_CORPUS_ALL = "/home/longhd/ALQAC2023/data/raw/V1.1/law.json"
PATH_TO_PUBLIC_TRAIN = my_env.PATH_TO_PUBLIC_TRAIN
PATH_TO_PUBLIC_TEST = my_env.PATH_TO_PUBLIC_TEST


def load_bm25_model():
    with open(os.path.join(my_env.PATH_TO_SAVE_MODEL, "my_bm25.pkl"), "rb") as bm_file:
        bm25 = pickle.load(bm_file)
    return bm25


def train_bm25():
    corpus_df = df_create_corpus(PATH_TO_CORPUS_ALL)
    id_corpus = corpus_df["id"].values.flatten().tolist()
    corpus = corpus_df["article"].values.flatten().tolist()

    tokenized_corpus = [word_segment(doc).split(" ") for doc in corpus]
    # tokenized_corpus = [doc for doc in tokenized_corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    with open(os.path.join(my_env.PATH_TO_SAVE_MODEL, "my_bm25.pkl"), "wb") as bm_file:
        pickle.dump(bm25, bm_file)

    with open(PATH_TO_PUBLIC_TRAIN, 'r') as file:
        data = json.load(file)

    correct_predictions = 0
    total_questions = len(data)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for item in data:
        question = item['text']
        relevant_articles = item['relevant_articles'][:2]
        for relevant_article in relevant_articles:
            law_id = relevant_article['law_id']
            article_id = relevant_article['article_id']
            relevant = f"{law_id}@{article_id}"
        total_choice = []
        if "choices" in item:
            for ans in item["choices"].values():
                new_question = "{}\n{}".format(question, ans)
                tokenized_query = word_segment(new_question).split(" ")
                # tokenized_query = new_question.split(" ")

                similarity_scores = bm25.get_scores(tokenized_query)
                max_similarity_index = np.argmax(similarity_scores)
                max_score = similarity_scores[max_similarity_index]
                total_choice.append((max_score, max_similarity_index))

        else:
            tokenized_query = word_segment(question).split(" ")
            # tokenized_query = question.split(" ")
            similarity_scores = bm25.get_scores(tokenized_query)
            max_similarity_index = np.argmax(similarity_scores)

        if total_choice:
            max_similarity_indexes = [idx for _, idx in total_choice]
            counter = Counter(max_similarity_indexes)
            most_common = counter.most_common()
            if len(most_common) in [2, 4]:
                _, max_similarity_index = max(total_choice, key=lambda x: x[0])
            else:
                max_similarity_index = most_common[0][0]

        predicted_id = convert_ID(id_corpus[max_similarity_index])
        label_id = convert_ID(relevant)
        if predicted_id == label_id:
            correct_predictions += 1
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1

    accuracy = calculate_accuracy(correct_predictions, total_questions)
    precision = calculate_precision(true_positive, false_positive)
    recall = calculate_recall(true_positive, false_negative)
    f2_score = calculate_f2_score(precision, recall)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F2-Score:", f2_score)


def test_submit(path_question, path_corpus):

    with open(path_question, 'r') as file:
        data = json.load(file)
    corpus_df = df_create_corpus(path_corpus)
    id_corpus = corpus_df["id"].values.flatten().tolist()
    bm25 = load_bm25_model()
    out_put = data

    for item in data:
        total_choice = []
        question = item['text']
        if "choices" in item:
            for ans in item["choices"].values():
                new_question = "{}\n{}".format(question, ans)
                tokenized_query = word_segment(new_question).split(" ")
                # tokenized_query = new_question.split(" ")

                similarity_scores = bm25.get_scores(tokenized_query)
                max_similarity_index = np.argmax(similarity_scores)
                max_score = similarity_scores[max_similarity_index]
                total_choice.append((max_score, max_similarity_index))

        else:
            tokenized_query = word_segment(question).split(" ")
            # tokenized_query = question.split(" ")
            similarity_scores = bm25.get_scores(tokenized_query)
            max_similarity_index = np.argmax(similarity_scores)

        if total_choice:
            max_similarity_indexes = [idx for _, idx in total_choice]
            counter = Counter(max_similarity_indexes)
            most_common = counter.most_common()
            # if len(most_common) > 1:
            #     _, max_similarity_index = max(total_choice, key=lambda x: x[0])
            # else:
            max_similarity_index = most_common[0][0]

        item.setdefault("relevant_articles", []).append(
            convert_ID(id_corpus[max_similarity_index]))
    output_file = "output_bm25.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_put, f, ensure_ascii=False)


train_bm25()
test_submit(PATH_TO_PUBLIC_TEST, PATH_TO_CORPUS_ALL)

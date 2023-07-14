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


PATH_TO_CORPUS = my_env.PATH_TO_CORPUS
PATH_TO_PUBLIC_TRAIN = my_env.PATH_TO_PUBLIC_TRAIN
PATH_TO_PUBLIC_TEST = my_env.PATH_TO_PUBLIC_TEST

corpus_df = df_create_corpus(PATH_TO_CORPUS)
id_corpus = corpus_df["id"].values.flatten().tolist()
corpus = corpus_df["article"].values.flatten().tolist()


tokenized_corpus = [word_segment(doc) for doc in corpus]
tokenized_corpus = [doc.split(" ") for doc in tokenized_corpus]

bm25 = BM25Plus(tokenized_corpus)
with open(os.path.join(my_env.PATH_TO_SAVE_MODEL, "my_bm25.pkl"), "wb") as bm_file:
    pickle.dump(bm25, bm_file)


def test():
    test_df = df_create_questions_test(PATH_TO_PUBLIC_TRAIN)
    querys = test_df["question"].values.flatten().tolist()
    relevants = test_df["relevant_id"].values.flatten().tolist()
    tokenized_querys = [word_segment(query) for query in querys]

    correct_predictions = 0
    total_questions = len(querys)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for tokenized_query, relevant in tqdm(zip(tokenized_querys, relevants)):
        tokenized_query = tokenized_query.split(" ")
        similarity_scores = bm25.get_scores(tokenized_query)
        max_similarity_index = np.argmax(similarity_scores)

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

    # print("Loss:", loss)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F2-Score:", f2_score)


def test_submit(path_json: str):

    with open(path_json, 'r') as file:
        data = json.load(file)
    out_put = data
    for item in data:
        # relevant_articles = []
        tokenized_question = word_segment(item["text"])
        tokenized_question = tokenized_question.split(" ")
        similarity_scores = bm25.get_scores(tokenized_question)
        max_similarity_index = np.argmax(similarity_scores)

        # relevant_articles.append(convert_ID(id_corpus[max_similarity_index]))
        item.setdefault("relevant_articles", []).append(
            convert_ID(id_corpus[max_similarity_index]))
    output_file = "output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out_put, f, ensure_ascii=False)



# test()
# test_submit(PATH_TO_PUBLIC_TEST)

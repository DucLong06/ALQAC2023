import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from eval_metrics import calculate_accuracy, calculate_f2_score, calculate_precision, calculate_recall
import my_env

from model_paraformer import Model_Paraformer
from post_data import convert_ID
import pickle
from processing_data import word_segment
from rank_bm25 import BM25Okapi, BM25Plus


def get_top_n_articles(query: str, data_corpus, top_n: int = 5):
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


def gen_submit(model: Model_Paraformer, data_question, data_corpus, alpha=0.1):
    model.eval()

    out_put = data_question
    for i, item in tqdm(enumerate(data_question), desc="Processing Question:"):
        final_scores = []
        question = item["text"]

        top_n_articles = get_top_n_articles(
            question, data_corpus, top_n=20)
        list_keys, list_articles, bm25_scores = zip(*top_n_articles)

        for i, article in enumerate(list_articles):
            article = [sentence.strip() for sentence in article.split(
                "\n") if sentence.strip() != ""]
            deep_score = model.get_score(question, article)

            # final_scores.append(alpha*deep_score+(1-alpha)*bm25_scores[i])

        # max_similarity_index = np.argmax(final_scores)
        max_similarity_index = np.argmax(deep_score)
        item.setdefault("relevant_articles", []).append(
            convert_ID(list_keys[max_similarity_index]))

    return out_put


def compare_json(data):
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

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F2 Score: {f2_score}')
    with open('error_data.json', 'w') as file:
        json.dump(error_data, file, ensure_ascii=False)


def main(path_to_model: str, path_to_query: str, path_to_law: str, compare: bool = False):

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

    output_data = gen_submit(model, data_question, data_corpus)

    output_file = "output_train.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)
    print("Processing complete. Output saved to", output_file)

    if compare:
        compare_json(output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default=my_env.PATH_TO_MODEL_PARAFORMER)
    parser.add_argument('--input_questions',
                        default=my_env.PATH_TO_PUBLIC_TRAIN)
    parser.add_argument('--input_articles',
                        default=my_env.PATH_TO_CORPUS_2023)
    parser.add_argument('--compare', default=False)

    opts = parser.parse_args()
    main("model_new.pth", opts.input_questions, opts.input_articles, True)

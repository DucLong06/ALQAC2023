import json
import random
import re
import pandas as pd
from tqdm import tqdm
from src.utils.rank_bm25 import BM25Okapi

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
        if "choices" in item:
            question += ' '.join(item['choices'].values())
        relevant_articles = item['relevant_articles'][:2]
        for relevant_article in relevant_articles:
            law_id = relevant_article['law_id']
            article_id = relevant_article['article_id']
            yield {'question': question, 'relevant_id': f"{law_id}@{article_id}"}


def df_create_questions_test(path_json: str) -> pd.DataFrame:
    list_data = _question_generator(path_json)
    return pd.DataFrame(list_data)


def data_training_generator(path_json_question: str, path_json_law: str,  top_bm25: int = 10, train_ratio=0.8, val_ratio=0.1):
    with open(path_json_law, 'r') as file:
        data_corpus = json.load(file)

    with open(path_json_question, 'r') as file:
        data_question = json.load(file)
    print(f"Number question: {len(data_question)}")
    print(f"Number corpus: {len(data_corpus)}")

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

            relevant_articles = item['relevant_articles']
            for question in list_question:
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

    train_df = pd.DataFrame(generate_data(train_data))
    val_df = pd.DataFrame(generate_data(val_data))
    test_df = pd.DataFrame(generate_data(test_data))

    print(f"Number of data in train set: {len(train_df)}")
    print(f"Number of data in val set: {len(val_df)}")
    print(f"Number of data in test set: {len(test_df)}")

    return train_df, val_df, test_df


def convert_all_law_to_json(*file_paths):
    result_json = {}

    for path in file_paths:

        with open(path, 'r') as file:
            data = json.load(file)
        print(f"File: {path} len{len(data)}")
        for item in tqdm(data):
            for article in item['articles']:
                id_text = f"{item['id']}@{article['id']}"
                text = article['text']
                result_json[id_text] = text

    json_data = json.dumps(result_json, ensure_ascii=False)
    output_file = 'all_articles_2023_f.json'
    with open(output_file, 'w') as file:
        file.write(json_data)


def merge_json_files(*file_paths):

    merged_data = []

    for path in file_paths:
        with open(path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)
            print(f"File: {path} len:{len(data)}")
    output_file = 'all_question_train_f.json'
    with open(output_file, 'w') as file:
        json.dump(merged_data, file, ensure_ascii=False)

    print("Merge completed. Output saved to", output_file)


def extract_law_ids(data_list):
    law_ids = []
    for item in data_list:
        for articles in item['relevant_articles']:
            law_ids.append(articles["law_id"])
    return law_ids


def merge_en_vi(file_en, file_vi, type="articles"):

    with open(file_en, 'r') as file:
        data_en = json.load(file)
    with open(file_vi, 'r') as file:
        data_vi = json.load(file)

    if type == "articles":
        result_json = {}
        for idx, item in tqdm(enumerate(data_en)):
            for article in item['articles']:
                id_text = f"{data_vi[idx]['id']}@{article['id']}"
                text = article['text']
                result_json[id_text] = text
        json_data = json.dumps(result_json, ensure_ascii=False)
        output_file = 'gg_all_articles_2023.json'
        with open(output_file, 'w') as file:
            file.write(json_data)

    elif type == "questions":
        result_json = []
        for idx, item in tqdm(enumerate(data_en)):
            item["relevant_articles"] = data_vi[idx]["relevant_articles"]
            result_json.append(item)

        with open("gg_question_train.json", 'w') as file:
            json.dump(result_json, file, ensure_ascii=False, indent=4)
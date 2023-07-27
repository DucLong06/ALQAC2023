from sentence_transformers import SentenceTransformer, util
import torch
import my_env as my_env
from src.retrieval.post_data import convert_ID
from src.retrieval.processing_data import word_segment
from src.utils.raw_data_process import df_create_corpus, df_create_questions_test
from src.utils.eval_metrics import *
from src.utils.rank_bm25 import *

embedder = SentenceTransformer('keepitreal/vietnamese-sbert')
# embedder = SentenceTransformer('vinai/phobert-base')
# Corpus with example sentences
PATH_TO_CORPUS = my_env.PATH_TO_CORPUS
PATH_TO_PUBLIC_TRAIN = my_env.PATH_TO_PUBLIC_TRAIN
PATH_TO_PUBLIC_TEST = my_env.PATH_TO_PUBLIC_TEST

corpus_df = df_create_corpus(PATH_TO_CORPUS)
id_corpus = corpus_df["id"].values.flatten().tolist()
corpus = corpus_df["article"].values.flatten().tolist()
corpus = [word_segment(doc) for doc in corpus]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


def sbert_test():
    test_df = df_create_questions_test(PATH_TO_PUBLIC_TRAIN)
    queries = test_df["question"].values.flatten().tolist()
    relevants = test_df["relevant_id"].values.flatten().tolist()
    queries = [word_segment(query) for query in queries]

    correct_predictions = 0
    total_questions = len(queries)
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for query, relevant in zip(queries, relevants):
        query_embedding = embedder.encode(query, convert_to_tensor=True)

        # Compute cosine similarity scores between the query and all corpus sentences
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        # Get the index of the most similar sentence
        top_result_index = torch.argmax(cos_scores)
        predicted_id = convert_ID(id_corpus[top_result_index])
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


sbert_test()

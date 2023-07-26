import glob
import os
import src.utils.my_env as my_env
import src.retrieval.evaluate as evaluate
import concurrent.futures


def evaluate_model(model, top_n, i_decimal):
    evaluate.main(model, my_env.PATH_TO_PUBLIC_TRAIN,
                  my_env.PATH_TO_CORPUS_2023, True, i_decimal, top_n)


list_model = [
    "models/F_sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230724040020.pth"]

for model in list_model:
    for top_n in range(66, 76, 10):
        for i in [4, 6, 7, 8, 9]:
            i_decimal = i / 10
            evaluate_model(model, top_n, i_decimal)

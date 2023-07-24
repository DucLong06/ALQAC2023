import glob
import os
import my_env
import evaluate
import concurrent.futures


def evaluate_model(model, top_n, i_decimal):
    evaluate.main(model, my_env.PATH_TO_PUBLIC_TRAIN,
                  my_env.PATH_TO_CORPUS_2023, True, i_decimal, top_n)


list_model = [
    "models/khanhpd2-sbert_phobert_large_cosine_sim_20230722133026.pth"]

for model in list_model:
    for top_n in range(10, 60, 10):
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]:
            i_decimal = i / 10
            evaluate_model(model, top_n, i_decimal)

import glob
import os
import my_env
import evaluate
import concurrent.futures


def evaluate_model(model, top_n, i_decimal):
    evaluate.main(model, my_env.PATH_TO_PUBLIC_TRAIN,
                  my_env.PATH_TO_CORPUS_2023, True, i_decimal, top_n)


list_model = ["models/sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230722135327.pth",
              "models/sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230722214805.pth",
              "models/sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230722232023.pth",
              "models/sentence-transformers-paraphrase-xlm-r-multilingual-v1_20230722233858.pth"]

for model in list_model:
    for top_n in range(10, 30, 10):
        for i in [1, 3, 6, 7, 8, 9]:
            i_decimal = i / 10
            evaluate_model(model, top_n, i_decimal)

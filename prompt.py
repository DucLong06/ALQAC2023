import asyncio
import datetime
import io
import json
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer
import my_env
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from bot_telegram import send_message
import src.utils.my_logger as my_logger
import torch

logger = my_logger.Logger("training", my_env.LOG)


def generate_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=2048, truncation=True).input_ids.to("cuda")
    outputs = model.generate(inputs, max_new_tokens=100,
                             return_dict_in_generate=True, output_scores=True, temperature=1)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    probabilities = torch.exp(transition_scores.sum(axis=1))

    return tokenizer.decode(outputs.sequences[0]).replace("<pad> ", "").replace("</s>", ""), probabilities


wrong_ans = set()


def determine_answer(prediction):
    prediction = prediction.lower()
    if any(word in prediction for word in my_env.TRUE_WORDS):
        return "True"
    elif any(word in prediction for word in my_env.FALSE_WORDS):
        return "False"
    else:
        wrong_ans.add(prediction)
        return prediction


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        prompts_data = json.load(file)
    return prompts_data


def tune(model, tokenizer, data_train, articles, essay_prompts, options_prompts, truefalse_prompts):

    total_questions = {
        "Question": 0,
        "True/False": 0,
        "Essay": 0
    }

    correct_prompts = defaultdict()
    for idx, item in enumerate(data_train):
        question = item['text']
        question_type = item["question_type"]

        if question_type == "Question":
            question_prompt_list = options_prompts
            total_questions["Question"] += 1
        elif question_type == "True/False":
            question_prompt_list = truefalse_prompts
            total_questions["True/False"] += 1
        else:
            question_prompt_list = essay_prompts
            total_questions["Essay"] += 1

        relevant_articles = item.get('relevant_articles', [])
        context = " ".join(
            articles[f"{ra['law_id']}@{ra['article_id']}"] for ra in relevant_articles)

        for idx, prompt in enumerate(question_prompt_list):

            if f"prompt_{question_type}_{idx+1}" not in correct_prompts.keys():
                correct_prompts[f"prompt_{question_type}_{idx+1}"] = 0
            if "choices" in item:
                text_prompt = prompt.format(
                    premise=context, hypothesis=question, choices=item["choices"])
            else:
                text_prompt = prompt.format(
                    premise=context, hypothesis=question)

            model_answer, probabilities = generate_text(
                text_prompt, model=model, tokenizer=tokenizer)
            model_answer = determine_answer(model_answer)
            ground_truth_answer = determine_answer(item['answer'])

            if model_answer == ground_truth_answer:
                correct_prompts[f"prompt_{question_type}_{idx+1}"] += 1
            # else:
                # wrong_ans.add(
                #     f"prompt_{question_type}_{idx+1} | {model_answer} | {ground_truth_answer}")
    with open(os.path.join(my_env.PATH_TO_SAVE_JSON, f"wrong_ans.txt"), "w") as f:
        f.write(str(wrong_ans))
    correct_prompts.update(total_questions)
    return correct_prompts


def plot_accuracy_by_prompt(accuracy, model_name):

    # Create some mock data
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Acuracy', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.savefig("data_visualization.png")


def main(question_data_path, articles_data_path, model, tokenizer, model_name):
    prompts = read_json_file("prompts/prompts.json")
    essay_prompts = prompts["essay"]
    options_prompts = prompts["options"]
    truefalse_prompts = prompts["truefalse"]

    with open(articles_data_path, 'r') as file:
        articles = json.load(file)

    with open(question_data_path, 'r') as file:
        data_train = json.load(file)

    accuracy = tune(model, tokenizer, data_train, articles,
                    essay_prompts, options_prompts, truefalse_prompts)
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    with open(os.path.join(my_env.PATH_TO_SAVE_JSON, f"{model_name.replace(r'/','')}_{time_string}.json"), "w") as f:
        json.dump(accuracy, f, ensure_ascii=False)

    plot_accuracy_by_prompt(accuracy, model_name)


if __name__ == "__main__":
    # list_llm_model = ['google/flan-t5-xl',
    #                   'google/flan-t5-large',
    #                   'facebook/mbart-large-50',
    #                   'Babelscape/rebel-large',

    #                   'declare-lab/flan-alpaca-xxl',
    #                   'declare-lab/flan-alpaca-gpt4-xl',
    #                   'declare-lab/flan-alpaca-xl',
    #                   'google/flan-ul2',
    #                   'bigscience/T0pp'
    #                   'bigscience/bloomz-7b1'

    #                   'google/umt5-base',
    #                   'microsoft/biogpt',
    #                   'microsoft/BioGPT-Large',
    #                   'microsoft/BioGPT-Large-PubMedQA',
    #                   'microsoft/git-large-vatex',
    #                   'microsoft/git-base-msrvtt-qa',
    #                   'microsoft/Promptist',
    #                   'microsoft/git-base-vatex',
    #                   'microsoft/xprophetnet-large-wiki100-cased'
    #                   'microsoft/xprophetnet-large-wiki100-cased-xglue-qg',
    #                   'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg',
    #                   'microsoft/prophetnet-large-uncased-squad-qg',
    #                   'microsoft/prophetnet-large-uncased-cnndm']
    list_llm_model = [1]
    for model_name in list_llm_model:

        model_name = 'google/flan-t5-xl'

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=my_env.PATH_TO_CACHE)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=my_env.PATH_TO_CACHE, device_map="auto")

        question_data_path = "data/training/vit5_question_train.json"
        articles_data_path = "data/training/vit5_all_articles_2023.json"
        main(question_data_path, articles_data_path, model=model,
             tokenizer=tokenizer, model_name=model_name)

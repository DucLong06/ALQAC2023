import argparse
import asyncio
import datetime
import io
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import my_env
from collections import defaultdict
from tqdm import tqdm
import src.utils.my_logger as my_logger
import torch
from googletrans import Translator
logger = my_logger.Logger("training", my_env.LOG)


def generate_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=2048, truncation=True).input_ids.to("cuda")
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100,
                                 return_dict_in_generate=True, output_scores=True, temperature=1)
    # transition_scores = model.compute_transition_scores(
    #     outputs.sequences, outputs.scores, normalize_logits=True
    # )s
    # probabilities = torch.exp(transition_scores.sum(axis=1))
    probabilities = 0
    return tokenizer.decode(outputs.sequences[0]).replace("<pad> ", "").replace("</s>", ""), probabilities


def determine_answer(prediction):
    prediction = prediction.lower()
    cleaned_string = re.sub(r'[^a-zA-Z]', '', prediction)
    if any(word in cleaned_string for word in my_env.TRUE_WORDS):
        return "True"
    elif any(word in cleaned_string for word in my_env.FALSE_WORDS):
        return "False"
    else:
        return prediction


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        prompts_data = json.load(file)
    return prompts_data


def prompt_tuning(model, tokenizer, data_train, articles, essay_prompts, options_prompts, truefalse_prompts, compare):
    total_questions = {
        "Question": 0,
        "True/False": 0,
        "Essay": 0
    }
    wrong_ans = []
    submit_file = []
    translator = Translator()
    correct_prompts = defaultdict()
    for idx, item in tqdm(enumerate(data_train)):
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
        # context = articles[f"{relevant_articles[0]['law_id']}@{relevant_articles[0]['article_id']}"]

        prompt_answers = []
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

            prompt_answers.append(model_answer)
            if compare:
                ground_truth_answer = determine_answer(item['answer'])
                if model_answer == ground_truth_answer:
                    correct_prompts[f"prompt_{question_type}_{idx+1}"] += 1
                else:
                    wrong_ans.append({
                        "prompt error": f"prompt_{question_type}_{idx+1}",
                        "model answer": model_answer,
                        "truth_answer": ground_truth_answer
                    })

        final_answer = max(set(prompt_answers), key=prompt_answers.count)
        if compare:
            if final_answer == ground_truth_answer:
                if f"majority_vote" not in correct_prompts.keys():
                    correct_prompts[f"majority_vote"] = 0
                correct_prompts[f"majority_vote"] += 1
        if final_answer == "True":
            answer = "Đúng"
        elif final_answer == "False":
            answer = "Sai"
        elif len(final_answer) < 2:
            answer = final_answer.upper()
        else:
            answer = translator.translate(
                final_answer, src='en', dest='vi').text

        submit_file.append({
            "question_id": item["question_id"],
            "answer": answer
        })

    with open(os.path.join(my_env.PATH_TO_SAVE_JSON, f"wrong_ans.json"), "w") as f:
        json.dump(wrong_ans, f, ensure_ascii=False)
    with open(os.path.join(my_env.PATH_TO_SAVE_JSON, f"submit_top{len(truefalse_prompts)}.json"), "w") as f:
        json.dump(submit_file, f, ensure_ascii=False)
    correct_prompts.update(total_questions)
    return correct_prompts


def main(model_name, question_data_path, articles_data_path, path_prompts, compare):

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=my_env.PATH_TO_CACHE)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, cache_dir=my_env.PATH_TO_CACHE, device_map="auto")

    with open(articles_data_path, 'r') as file:
        articles = json.load(file)

    with open(question_data_path, 'r') as file:
        data_train = json.load(file)
    for path in path_prompts:
        prompts = read_json_file(path)
        essay_prompts = prompts["essay"]
        options_prompts = prompts["options"]
        truefalse_prompts = prompts["truefalse"]
        accuracy = prompt_tuning(model, tokenizer, data_train, articles,
                                 essay_prompts, options_prompts, truefalse_prompts, compare)
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d%H%M%S")
        with open(os.path.join(my_env.PATH_TO_SAVE_JSON, f"{model_name.replace(r'/','')}_{time_string}.json"), "w") as f:
            json.dump(accuracy, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='google/flan-t5-xxl')
    parser.add_argument('--questions', type=str,
                        default="data/raw_en/gg_question_private.json")
    parser.add_argument('--articles', type=str,
                        default="data/training/gg_all_articles_2023.json")
    parser.add_argument('--prompts', type=str,
                        default="prompts/prompts_top10.json")
    parser.add_argument('--compare', type=bool,
                        default=True)

    opts = parser.parse_args()
    path_prompts = ["prompts/prompts_top1.json",
                    "prompts/prompts_top5.json",
                    "prompts/prompts_top10.json"]
    main(opts.model, opts.questions, opts.articles, path_prompts, False)

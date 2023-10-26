import argparse
import asyncio
import datetime
import io
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from my_env import env
from collections import defaultdict
from tqdm import tqdm
import src.utils.my_logger as my_logger
import torch
# from googletrans import Translator
# from torch.nn import DataParallel

logger = my_logger.Logger("training", env.LOG)


class LanguageType:
    ENGLISH = "en"
    VIETNAMESE = "vie"


class AnswerType:
    TRUE = "true"
    FALSE = "false"
    CORRECT_VI = "đúng"
    FALSE_VI = "sai"


def generate_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=2048).input_ids.to("cuda")

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100,
                                 return_dict_in_generate=True, output_scores=True, temperature=1)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    probabilities = torch.exp(transition_scores.sum(axis=1))
    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), probabilities


def determine_answer(prediction, lang):
    prediction = prediction.lower()
    # cleaned_string = re.sub(r'[^a-zA-Z]', '', prediction)
    if lang == LanguageType.ENGLISH:
        true_words = env.TRUE_WORDS
        false_words = env.FALSE_WORDS
        label_true = AnswerType.TRUE
        label_false = AnswerType.FALSE
    elif lang == LanguageType.VIETNAMESE:
        true_words = env.TRUE_WORDS_VN
        false_words = env.FALSE_WORDS_VN
        label_true = AnswerType.CORRECT_VI
        label_false = AnswerType.FALSE_VI
    if any(word in prediction for word in true_words):
        return label_true
    elif any(word in prediction for word in false_words):
        return label_false
    else:
        return prediction


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        prompts_data = json.load(file)
    return prompts_data


def get_total_questions(language):
    if language == LanguageType.ENGLISH:
        return {"Question": 0, "True/False": 0, "Essay": 0}
    else:
        return {"Trắc nghiệm": 0, "Đúng/Sai": 0, "Tự luận": 0}


def get_question_prompt_list(language, question_type, essay_prompts, options_prompts, truefalse_prompts):
    if language == LanguageType.ENGLISH:
        if question_type == "Question":
            return options_prompts
        elif question_type == "True/False":
            return truefalse_prompts
        else:
            return essay_prompts
    else:
        if question_type == "Trắc nghiệm":
            return options_prompts
        elif question_type == "Đúng/Sai":
            return truefalse_prompts
        else:
            return essay_prompts


def prompt_tuning(model, tokenizer, data_train, articles, essay_prompts, options_prompts, truefalse_prompts, compare, language):
    total_questions = get_total_questions(language)
    wrong_ans = []
    submit_file = []
    # translator = Translator()
    correct_prompts = defaultdict()
    answer_file = []
    for idx, item in tqdm(enumerate(data_train)):
        question = item['text']
        question_type = item["question_type"]
        if language == LanguageType.ENGLISH:
            if question_type == "Question":
                question_prompt_list = options_prompts
                total_questions["Question"] += 1
            elif question_type == "True/False":
                question_prompt_list = truefalse_prompts
                total_questions["True/False"] += 1
            else:
                question_prompt_list = essay_prompts
                total_questions["Essay"] += 1
        else:
            if question_type == "Trắc nghiệm":
                question_prompt_list = options_prompts
                total_questions["Trắc nghiệm"] += 1
            elif question_type == "Đúng/Sai":
                question_prompt_list = truefalse_prompts
                total_questions["Đúng/Sai"] += 1
            else:
                question_prompt_list = essay_prompts
                total_questions["Tự luận"] += 1

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

            model_answer = determine_answer(model_answer, language)

            prompt_answers.append(model_answer)
            if compare:
                ground_truth_answer = determine_answer(
                    item['answer'], language)
                if model_answer == ground_truth_answer:
                    correct_prompts[f"prompt_{question_type}_{idx+1}"] += 1
                else:
                    wrong_ans.append({
                        "prompt error": f"prompt_{question_type}_{idx+1}",
                        "question": question,
                        "model answer": model_answer,
                        "truth answer": ground_truth_answer
                    })

        final_answer = max(set(prompt_answers), key=prompt_answers.count)
        if final_answer == AnswerType.TRUE:
            answer = AnswerType.CORRECT_VI
        elif final_answer == AnswerType.FALSE:
            answer = AnswerType.FALSE_VI
        elif len(final_answer) < 2:
            answer = final_answer.upper()
        elif language == LanguageType.ENGLISH:
            answer = translator.translate(
                final_answer, src='en', dest='vi').text
        elif question_type == "Tự luận":
            answer = final_answer
        else:
            answer = final_answer
        if compare:
            if final_answer == ground_truth_answer:
                if f"majority_vote" not in correct_prompts.keys():
                    correct_prompts[f"majority_vote"] = 0
                correct_prompts[f"majority_vote"] += 1
        else:
            submit_file.append({
                "question_id": item["question_id"],
                "answer": answer
            })
        answer_file.append({
            "question_id": item["question_id"],
            "answer": answer
        })
    with open(os.path.join(env.PATH_TO_SAVE_JSON, f"wrong_ans.json"), "w") as f:
        json.dump(wrong_ans, f, ensure_ascii=False)
    with open(os.path.join(env.PATH_TO_SAVE_JSON, f"model_answer_top{len(truefalse_prompts)}.json"), "w") as f:
        json.dump(answer_file, f, ensure_ascii=False)
    with open(os.path.join(env.PATH_TO_SAVE_JSON, f"submit_top{len(truefalse_prompts)}.json"), "w") as f:
        json.dump(submit_file, f, ensure_ascii=False)
    correct_prompts.update(total_questions)
    return correct_prompts


def main(model_name, question_data_path, articles_data_path, path_prompts, compare, language):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=env.PATH_TO_CACHE)
    # model = AutoModel.from_pretrained(
    #     model_name, cache_dir=env.PATH_TO_CACHE, device_map="auto")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, cache_dir=env.PATH_TO_CACHE, device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, cache_dir=env.PATH_TO_CACHE, device_map="auto")

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
                                 essay_prompts, options_prompts, truefalse_prompts, compare, language)
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d%H%M%S")
        with open(os.path.join(env.PATH_TO_SAVE_JSON, f"{model_name.replace(r'/','')}_{time_string}.json"), "w") as f:
            json.dump(accuracy, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default="bigscience/mt0-xxl")
    parser.add_argument('--questions', type=str,
                        default="data/raw/private_test.json")
    parser.add_argument('--articles', type=str,
                        default="data/training/all_articles_2023.json")
    parser.add_argument('--prompts', type=str,
                        default="prompts/prompts_top10.json")
    parser.add_argument('--compare', type=bool,
                        default=True)
    parser.add_argument('--language', type=str,
                        default="en")

    opts = parser.parse_args()
    # path_prompts = ["prompts/prompts_vn_top5.json",
    #                 "prompts/prompts_vn_top10.json",
    #                 "prompts/prompts_vn_top15.json"]
    path_prompts = ["prompts/prompt_vn_top3.json"]
    main(opts.model, opts.questions, opts.articles,
         path_prompts, False, LanguageType.VIETNAMESE)

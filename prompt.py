import argparse
import datetime
import json
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
from tqdm import tqdm
import torch


class AnswerType:
    CORRECT_VI = "yes"
    FALSE_VI = "no"


def generate_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=2048).input_ids.to("cuda")

    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=100,
                                 return_dict_in_generate=True, output_scores=True, temperature=1)

    return tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)


def determine_answer(prediction):
    prediction = prediction.lower()
    true_words = {'yes', "đúng", 'luôn đúng',
                  'luôn luôn', 'đảm bảo', 'Khẳng định'}
    label_true = AnswerType.CORRECT_VI
    label_false = AnswerType.FALSE_VI
    if any(word in prediction for word in true_words):
        return label_true
    else:
        return label_false


def read_json_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        prompts_data = json.load(file)
    return prompts_data


def prompt_tuning(model, tokenizer, data_train, articles, truefalse_prompts):
    total_questions = {"True/False": len(data_train)}
    correct_prompts = defaultdict()
    answer_file = defaultdict()
    for idx, item in tqdm(enumerate(data_train)):
        question = item['statement']

        relevant_articles = item.get('legal_passages', [])
        context = " ".join(
            articles[f"{ra['law_id']}@{ra['article_id']}"] for ra in relevant_articles)

        for idx, prompt in enumerate(truefalse_prompts):
            if f"prompt_{idx+1}" not in correct_prompts.keys():
                correct_prompts[f"prompt_{idx+1}"] = 0
                answer_file[f"prompt_{idx+1}"] = []
            text_prompt = prompt.format(
                premise=context, hypothesis=question)

            model_answer = generate_text(
                text_prompt, model=model, tokenizer=tokenizer)

            answer = determine_answer(model_answer)

            answer_file[f"prompt_{idx+1}"].append({
                "example_id": item["example_id"],
                "answer": answer
            })

    with open(os.path.join(f"answer_{len(truefalse_prompts)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_.json"), "w") as f:
        json.dump(answer_file, f, ensure_ascii=False)

    correct_prompts.update(total_questions)
    return correct_prompts


def main(model_name, question_data_path, articles_data_path, path_prompts, path_to_cache):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=path_to_cache)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, cache_dir=path_to_cache, device_map="auto")

    with open(articles_data_path, 'r') as file:
        articles = json.load(file)

    with open(question_data_path, 'r') as file:
        data_train = json.load(file)

    prompts = read_json_file(path_prompts)
    truefalse_prompts = prompts["truefalse"]
    accuracy = prompt_tuning(
        model, tokenizer, data_train, articles, truefalse_prompts)
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    with open(os.path.join(f"{model_name.replace(r'/','')}_{time_string}.json"), "w") as f:
        json.dump(accuracy, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--questions', type=str)
    parser.add_argument('--articles', type=str)
    parser.add_argument('--prompts', type=str)
    parser.add_argument('--cache', type=str)

    opts = parser.parse_args()

    main(opts.model, opts.questions, opts.articles, opts.prompts,  opts.cache)

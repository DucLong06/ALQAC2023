import io
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import my_env
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_text(input_text, model, tokenizer):
    inputs = tokenizer(input_text, return_tensors="pt",
                       max_length=2048, truncation=True).input_ids.to("cuda")
    outputs = model.generate(inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0]).replace("<pad> ", "").replace("</s>", "")


def determine_answer(prediction):
    prediction = prediction.lower()
    if prediction in ['true', 'correct', 'accurate', 'right', 'yes', 'affirmative']:
        return "True"
    elif prediction in ['false', 'incorrect', 'inaccurate', 'wrong', 'no', 'negative', 'not true']:
        return "False"
    else:
        return prediction


def read_file(file_path):
    with io.open(file_path, 'r', encoding='UTF-8') as file:
        return file.read().strip().split('\n')


def tune(model, tokenizer, data_train, articles, prompts_essay, prompts_options, prompts_truefalse):
    correct_predictions = 0

    total_questions = defaultdict()

    for idx, item in enumerate(data_train):
        question = item['text']
        question_type = item["question_type"]

        if question_type == "Question":
            question_prompt_list = prompts_options
            total_questions["Question"] += 1
        elif question_type == "True/False":
            question_prompt_list = prompts_truefalse
            total_questions["True/False"] += 1
        else:
            question_prompt_list = prompts_essay
            total_questions["Essay"] += 1

        relevant_articles = item.get('relevant_articles', [])
        context = " ".join(
            articles[f"{ra['law_id']}@{ra['article_id']}"] for ra in relevant_articles)

        for prompt in question_prompt_list:
            if "choices" in item:
                text_prompt = prompt.format(
                    context=context, querry=question, choices=item["choices"])
            else:
                text_prompt = prompt.format(context=context, querry=question)

            model_answer = generate_text(
                text_prompt, model=model, tokenizer=tokenizer)
            model_answer = determine_answer(model_answer)
            ground_truth_answer = determine_answer(item['answer'])

            if model_answer == ground_truth_answer:
                correct_predictions += 1
                correct_predictions_by_question[question_type] += 1
                correct_predictions_prompt += 1

            total_questions_prompt += 1

        accuracy_by_prompt[question_type] = correct_predictions_prompt / \
            total_questions_prompt

    accuracy = correct_predictions / len(data_train)
    accuracy_by_question = {
        "Question": correct_predictions_by_question["Question"] / total_questions["Question"],
        "True/False": correct_predictions_by_question["True/False"] / total_questions["True/False"],
        "Essay": correct_predictions_by_question["Study"] / total_questions["Study"]
    }

    return accuracy, accuracy_by_question, accuracy_by_prompt


def main(question_data_path, model, tokenizer):
    prompts_essay = read_file("prompts/essay")
    prompts_options = read_file("prompts/options")
    prompts_truefalse = read_file("prompts/truefalse")

    with open("data/training/gg_all_articles_2023.json", 'r') as file:
        articles = json.load(file)

    with open(question_data_path, 'r') as file:
        data_train = json.load(file)

    accuracy, accuracy_by_question, accuracy_by_prompt = tune(
        model, tokenizer, data_train, articles, prompts_essay, prompts_options, prompts_truefalse)

    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"Accuracy by Question Type:")
    for question_type, acc in accuracy_by_question.items():
        print(f"  {question_type}: {acc:.2f}")

    print(f"Accuracy by Prompt Type:")
    for prompt_type, acc in accuracy_by_prompt.items():
        print(f"  {prompt_type}: {acc:.2f}")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-xl", cache_dir=my_env.PATH_TO_CACHE)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-xl", cache_dir=my_env.PATH_TO_CACHE, device_map="auto")
    question_data_path = "data/training/gg_question_train.json"

    main(question_data_path, model=model, tokenizer=tokenizer)

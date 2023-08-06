import os
import json
import datetime
from collections import defaultdict
from tqdm import tqdm
from googletrans import Translator
from LLM import LanguageModel, LanguageType, AnswerType
from ....my_env import env


class QuestionType:
    def __init__(self, language):
        self.language = language
        self.essay_prompts = []
        self.options_prompts = []
        self.truefalse_prompts = []

    def load_prompts(self, essay_prompts, options_prompts, truefalse_prompts):
        self.essay_prompts = essay_prompts
        self.options_prompts = options_prompts
        self.truefalse_prompts = truefalse_prompts


class QuestionPromptTuner:
    def __init__(self, model_name, question_data_path, articles_data_path, language, path_prompts, compare):
        self.model_name = model_name
        self.question_data_path = question_data_path
        self.articles_data_path = articles_data_path
        self.language = language
        self.path_prompts = path_prompts
        self.compare = compare
        self.model = LanguageModel(model_name, language)
        self.translator = Translator()

    def read_json_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            prompts_data = json.load(file)
        return prompts_data

    def _get_total_questions(language):
        if language == LanguageType.ENGLISH:
            return {"Question": 0, "True/False": 0, "Essay": 0}
        else:
            return {"Trắc nghiệm": 0, "Đúng/Sai": 0, "Tự luận": 0}

    def prompt_tuning(self, model, tokenizer, data_train, articles, essay_prompts, options_prompts, truefalse_prompts, compare, language):
        total_questions = self._get_total_questions(language)
        wrong_ans = []
        submit_file = []
        translator = Translator()
        correct_prompts = defaultdict()
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

                model_answer, probabilities = self.model.generate_text(
                    text_prompt, model=model, tokenizer=tokenizer)

                model_answer = self.model.determine_answer(
                    model_answer, language)

                prompt_answers.append(model_answer)
                if compare:
                    ground_truth_answer = self.model.determine_answer(
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
            if compare:
                if final_answer == ground_truth_answer:
                    if f"majority_vote" not in correct_prompts.keys():
                        correct_prompts[f"majority_vote"] = 0
                    correct_prompts[f"majority_vote"] += 1

            else:
                if final_answer == AnswerType.TRUE:
                    answer = AnswerType.CORRECT_VI
                elif final_answer == AnswerType.FALSE:
                    answer = AnswerType.FALSE_VI
                elif len(final_answer) < 2:
                    answer = final_answer.upper()
                elif language == LanguageType.ENGLISH:
                    answer = translator.translate(
                        final_answer, src='en', dest='vi').text
                submit_file.append({
                    "question_id": item["question_id"],
                    "answer": answer
                })

        with open(os.path.join(env.PATH_TO_SAVE_JSON, f"wrong_ans.json"), "w") as f:
            json.dump(wrong_ans, f, ensure_ascii=False)
        with open(os.path.join(env.PATH_TO_SAVE_JSON, f"submit_top{len(truefalse_prompts)}.json"), "w") as f:
            json.dump(submit_file, f, ensure_ascii=False)
        correct_prompts.update(total_questions)
        return correct_prompts

    def main(self):
        self.model.load_model(self.model_name)

        with open(self.articles_data_path, "r") as file:
            articles = json.load(file)

        with open(self.question_data_path, "r") as file:
            data_train = json.load(file)

        for path in self.path_prompts:
            prompts = self.read_json_file(path)
            essay_prompts = prompts["essay"]
            options_prompts = prompts["options"]
            truefalse_prompts = prompts["truefalse"]
            accuracy = self.prompt_tuning(
                data_train, articles, essay_prompts, options_prompts, truefalse_prompts
            )
            now = datetime.datetime.now()
            time_string = now.strftime("%Y%m%d%H%M%S")
            with open(
                    os.path.join(env.PATH_TO_SAVE_JSON, f"{self.model_name.replace(r'/','')}_{time_string}.json"), "w",) as f:
                json.dump(accuracy, f, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bigscience/mt0-base")
    parser.add_argument(
        "--questions", type=str, default="data/raw/train.json"
    )
    parser.add_argument(
        "--articles", type=str, default="data/training/all_articles_2023.json"
    )
    parser.add_argument(
        "--prompts", type=str, default="prompts/prompts_top10.json"
    )
    parser.add_argument("--compare", type=bool, default=True)
    parser.add_argument("--language", type=str, default="en")

    opts = parser.parse_args()
    path_prompts = ["prompts/prompts_vn_top1.json"]
    tuner = QuestionPromptTuner(
        opts.model, opts.questions, opts.articles, path_prompts, opts.compare, opts.language
    )
    tuner.main()

import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_text(input_text):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, max_length=100, num_beams=5, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


with open("prompt.json", 'r') as file:
    prompts = json.load(file)

with open("data/training/gg_all_articles_2023.json", 'r') as file:
    articles = json.load(file)

with open("data/training/gg_question_train.json", 'r') as file:
    data_train = json.load(file)

for item in data_train:
    question = item['text']
    relevant_articles = item['relevant_articles']
    id_context = []
    context = ""
    for relevant_article in relevant_articles:
        id_context.append(
            f"{relevant_article['law_id']}@{relevant_article['article_id']}")
    for id in id_context:
        context += " "+articles[id]
    for prompt in prompts.values():
        text_prompt = prompt.replace(
            "{{context}}", context).replace(
            "{{querry}}", question)
        print(generate_text(text_prompt))
    break

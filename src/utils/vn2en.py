from collections import defaultdict
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


def _translate_using_google(text):
    translator = Translator()
    sentences = text.split("\n")
    for sentence in sentences:
        if sentence.strip():
            yield translator.translate(text, src='vi', dest='en').text


def _translate_using_transformers(text, model, tokenizer):
    sentences = text.split("\n")
    for sentence in sentences:
        if sentence.strip():
            inputs = f"vi: {sentence.strip()}"
            encoded_inputs = tokenizer(inputs, return_tensors="pt")
            outputs = model.generate(
                encoded_inputs.input_ids.to('cpu'), max_length=512)
            translated_sentence = tokenizer.batch_decode(
                outputs, skip_special_tokens=True)[0]
            yield translated_sentence.replace("en: ", "")


def translate_text(data, method='transformers'):
    if method == 'transformers':
        model_name = "VietAI/envit5-translation"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def _translate_sentence(text):
            return _translate_using_transformers(text, model=model, tokenizer=tokenizer)

    elif method == 'google':
        def _translate_sentence(text):
            return _translate_using_google(text)

    else:
        raise ValueError(
            "Invalid translation method. Please choose 'transformers' or 'google'.")

    translated_data = []
    for idx, item in enumerate(data):
        print(f"Number: {idx+1}/{len(data)}")
        translated_articles = []
        # if isinstance(item, dict):
        if "articles" in item.keys():
            for article in tqdm(item["articles"]):
                translated_text = _translate_sentence(article["text"])
                translated_article = {
                    "text": " ".join(translated_text),
                    "id": article["id"]
                }
                translated_articles.append(translated_article)
            translated_item = {
                "id": " ".join(_translate_sentence(item["id"])),
                "articles": translated_articles
            }
        else:
            for article in tqdm(item["relevant_articles"]):
                # translated_text = _translate_sentence(article["law_id"])
                translated_article = {
                    "law_id": article["law_id"],
                    "article_id": article["article_id"]
                }
                translated_articles.append(translated_article)
            if "choices" in item.keys():
                translated_choices = defaultdict()
                for ans, choice in item["choices"].items():
                    translated_choices[ans] = " ".join(
                        _translate_sentence(choice))
                translated_item = {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "text": " ".join(_translate_sentence(item["text"])),
                    "relevant_articles": translated_articles,
                    "answer": " ".join(_translate_sentence(item["answer"])),
                    "choices": translated_choices
                }
            else:
                translated_item = {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "text": " ".join(_translate_sentence(item["text"])),
                    "relevant_articles": translated_articles,
                    "answer": " ".join(_translate_sentence(item["answer"]))
                }

        translated_data.append(translated_item)

    return translated_data

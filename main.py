import my_env
import json
from tqdm import tqdm
from src.utils.vn2en import translate_text
from googletrans import Translator


def translate_json(json_data, translator):
    if isinstance(json_data, list):
        translated_data = []
        for item in json_data:
            translated_item = translate_json(item, translator)
            translated_data.append(translated_item)
        return translated_data
    elif isinstance(json_data, dict):
        translated_data = {}
        for key, value in json_data.items():
            if isinstance(value, str):
                translated_value = translator.translate(
                    value, src='vi', dest='en').text
                translated_data[key] = translated_value
            elif isinstance(value, (dict, list)):
                translated_data[key] = translate_json(value, translator)
            else:
                translated_data[key] = value
        return translated_data
    else:
        return json_data


if __name__ == "__main__":
    with open("data/raw/train.json", 'r') as file:
        data = json.load(file)
    # translator = Translator()
    # translated_data = translate_json(data, translator)
    # with open("ggapi_train.json", "w") as f:
    #     json.dump(translated_data, f, ensure_ascii=False, indent=4)
    # print("Translation completed and data saved to the JSON file.")
    translated_data = translate_text(data, method='transformers')

    with open("vit5_train.json", 'w') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

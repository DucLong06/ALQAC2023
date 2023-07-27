import my_env
import json
from tqdm import tqdm
from src.utils.vn2en import translate_text

if __name__ == "__main__":
    with open("data/raw/private_test.json", 'r') as file:
        data = json.load(file)

    translated_data = translate_text(data, method='transformers')

    with open("vit5_private_test.json", 'w') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

    print("Translation completed and data saved to the JSON file.")

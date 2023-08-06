import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ....my_env import env


class LanguageType:
    ENGLISH = "en"
    VIETNAMESE = "vie"


class AnswerType:
    TRUE = "true"
    FALSE = "false"
    CORRECT_VI = "đúng"
    FALSE_VI = "sai"


class LanguageModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self, model_name):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=env.PATH_TO_CACHE, device_map="auto"
        )
        self.model.to(self.device)

    def generate_text(self, input_text, max_new_tokens=100, temperature=1):
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=2048, truncation=True
        ).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_scores=True, temperature=temperature
            )
         # transition_scores = model.compute_transition_scores(
        #     outputs.sequences, outputs.scores, normalize_logits=True
        # )s
        # probabilities = torch.exp(transition_scores.sum(axis=1))
        probabilities = 0  # Note: the probabilities calculation seems to be commented out

        return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), probabilities

    def determine_answer(self, prediction, lang):
        prediction = prediction.lower()
        cleaned_string = re.sub(r"[^a-zA-Z]", "", prediction)
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
        if any(word in cleaned_string for word in true_words):
            return label_true
        elif any(word in cleaned_string for word in false_words):
            return label_false
        else:
            return prediction

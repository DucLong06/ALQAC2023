import re
import string
from pyvi import ViTokenizer, ViPosTagger
import torch
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def _clean_text(text: str) -> str:
    """
    Clean the input text by removing numbers, special characters, HTML tags, and extra spaces.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.

    """

    # remove number
    # text = re.sub(r'[0-9]', ' ', text)
    # remove special
    text = re.sub(
        r'[?|$|.,…|!|:;\\*\-+=^/\'\"\”@`~#%&\]\[\{\}–—­­­­­\()]', r' ', text)
    # remove tag HTML
    text = re.sub('<.*?>', '', text).strip()
    # remove spaces
    text = re.sub('(\s)+', r'\1', text)
    return text


def _normalize_text(text: str) -> str:
    """
    Normalize the input text by removing punctuation and converting it to lowercase.

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.

    """
    listpunctuation = string.punctuation.replace('_', '')
    for i in listpunctuation:
        text = text.replace(i, ' ')
    return text.lower()


def _remove_stopword(text: str) -> str:
    """
    Remove stopwords from the input text and return the processed text.

    Args:
        text (str): The input text from which stopwords need to be removed.

    Returns:
        str: The text after removing stopwords.

    """

    list_stopwords = open("/Users/longhoangduc/Documents/JAIST/ALQAC_2023/Src/Task1/stopwords.txt",
                          "r", encoding='utf-8').read()
    words = text.split()
    pre_text = [word for word in words if word not in list_stopwords]
    return ' '.join(pre_text)


def word_segment(text: str, tags_filter=["V", "N", "P", "."]) -> str:
    """
    Segment the words in the text(using pyvi) and return the segmented text cleaned.

    Args:
        text (str): The input text to be word segmented.

    Returns:
        str: The text after word segmentation.

    """

    text = ViTokenizer.tokenize(text)
    text = _normalize_text(text)
    return text
    # pos = ViPosTagger.postagging(text)
    # new_words = [word for word, tag in zip(
    #     pos[0], pos[1]) if tag in tags_filter]

    # return " ".join(new_words)

import os
from dotenv import load_dotenv

load_dotenv()

root_dir = os.path.dirname(os.path.abspath(__file__))

PATH_TO_BIN_CORPUS_ALL = os.getenv(
    "PATH_TO_BIN_CORPUS_ALL", os.path.join(root_dir, 'data', 'training', 'embedding_data.pkl'))

PATH_TO_CORPUS_ALL = os.getenv(
    "PATH_TO_CORPUS_ALL", os.path.join('data', 'training', 'all_articles.json'))

PATH_TO_CORPUS_2023 = os.getenv(
    "PATH_TO_CORPUS_2023", os.path.join('data', 'training', 'all_articles_2023.json'))

PATH_TO_QUESTION_ALL = os.getenv(
    "PATH_TO_QUESTION_ALL", os.path.join('data', 'training', 'all_question_train.json'))

PATH_TO_PUBLIC_TRAIN = os.getenv(
    "PATH_TO_PUBLIC_TRAIN",  os.path.join(root_dir, 'data', 'raw', 'V1.1', 'train.json'))

PATH_TO_PUBLIC_TEST = os.getenv(
    "PATH_TO_PUBLIC_TEST",  os.path.join(root_dir, 'data', 'raw', 'V1.1', 'public_test.json'))

PATH_TO_SAVE_MODEL = os.getenv(
    "PATH_TO_SAVE_MODEL", os.path.join(root_dir, 'models'))

PATH_TO_MODEL_PARAFORMER = os.getenv(
    "PATH_TO_MODEL_PARAFORMER", os.path.join(root_dir, 'models', 'keepitreal-vietnamese-sbert_20230720174334.pth'))

ID_TELEGRAM = os.getenv("ID_TELEGRAM")
TOKEN_BOT = os.getenv("TOKEN_BOT")

list_base_model =[
    "keepitreal/vietnamese-sbert",
    "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    "khanhpd2/sbert_phobert_large_cosine_sim",
    "khanhpd2/sbert-vinai-phobert-large-mnr",
    "ThiennNguyen/vi-sbert-QA",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
]
# system
LOG = os.getenv("LOG", "log/LOG")

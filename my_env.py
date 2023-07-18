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
    "PATH_TO_QUESTION_ALL", os.path.join('data', 'training', 'all_question.json'))

PATH_TO_PUBLIC_TRAIN = os.getenv(
    "PATH_TO_PUBLIC_TRAIN",  os.path.join(root_dir, 'data', 'raw', 'V1.1', 'train.json'))

PATH_TO_PUBLIC_TEST = os.getenv(
    "PATH_TO_PUBLIC_TEST",  os.path.join(root_dir, 'data', 'raw', 'V1.1', 'public_test.json'))

PATH_TO_SAVE_MODEL = os.getenv(
    "PATH_TO_SAVE_MODEL", os.path.join(root_dir, 'models'))

PATH_TO_MODEL_PARAFORMER = os.getenv(
    "PATH_TO_MODEL_PARAFORMER", os.path.join(root_dir, 'models', 'paraformer.pth'))

ID_TELEGRAM = os.getenv("ID_TELEGRAM")
TOKEN_BOT = os.getenv("TOKEN_BOT")

# system

LOG = os.getenv("LOG", "log/LOG")

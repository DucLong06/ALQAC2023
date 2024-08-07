import os
from dotenv import load_dotenv

load_dotenv()


class env():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    PATH_TO_CACHE = os.getenv(
        "PATH_TO_CACHE", os.path.join(root_dir, 'cache'))

    PATH_TO_BIN_CORPUS_ALL = os.getenv(
        "PATH_TO_BIN_CORPUS_ALL", os.path.join(root_dir, 'data', 'training', 'embedding_data.pkl'))

    PATH_TO_CORPUS = os.getenv(
        "PATH_TO_CORPUS", os.path.join('data', 'raw', 'law.json'))

    PATH_TO_CORPUS_ALL = os.getenv(
        "PATH_TO_CORPUS_ALL", os.path.join('data', 'training', 'all_articles.json'))

    PATH_TO_CORPUS_2023 = os.getenv(
        "PATH_TO_CORPUS_2023", os.path.join('data', 'training', 'all_articles_2023.json'))

    PATH_TO_QUESTION_ALL = os.getenv(
        "PATH_TO_QUESTION_ALL", os.path.join('data', 'training', 'all_question_train.json'))

    PATH_TO_QUESTION_F = os.getenv(
        "PATH_TO_QUESTION_F", os.path.join('data', 'training', 'all_question_train_f.json'))

    PATH_TO_PUBLIC_TRAIN = os.getenv(
        "PATH_TO_PUBLIC_TRAIN",  os.path.join(root_dir, 'data', 'raw', 'train.json'))

    PATH_TO_PUBLIC_TEST = os.getenv(
        "PATH_TO_PUBLIC_TEST",  os.path.join(root_dir, 'data', 'raw', 'public_test.json'))

    PATH_TO_SAVE_MODEL = os.getenv(
        "PATH_TO_SAVE_MODEL", os.path.join(root_dir, 'models'))

    PATH_TO_SAVE_JSON = os.getenv(
        "PATH_TO_SAVE_JSON", os.path.join(root_dir, 'data', 'result'))

    PATH_TO_MODEL_PARAFORMER = os.getenv(
        "PATH_TO_MODEL_PARAFORMER", os.path.join(root_dir, 'models', 'keepitreal-vietnamese-sbert_20230720174334.pth'))

    ID_TELEGRAM = os.getenv("ID_TELEGRAM")
    TOKEN_BOT = os.getenv("TOKEN_BOT")

    list_base_model = [
        "keepitreal/vietnamese-sbert",
        "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "khanhpd2/sbert_phobert_large_cosine_sim",
        "hmthanh/VietnamLegalText-SBERT"
    ]

    dict_bast_model = {
        "keepitreal": "keepitreal/vietnamese-sbert",
        "vovanphuc": "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        "sentence-transformers": "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "khanhpd2": "khanhpd2/sbert_phobert_large_cosine_sim",
        "hmthanh": "hmthanh/VietnamLegalText-SBERT"
    }

    TRUE_WORDS = {'true', 'correct', 'accurate', 'right',
                  'yes', 'affirmative', 'guaranteed', 'always' }
    FALSE_WORDS = {'false', 'incorrect', 'inaccurate', 'wrong',
                   'no', 'negative', 'not true', 'never', 'impossible'}

    TRUE_WORDS_VN = {'yes', "đúng",'luôn đúng','luôn luôn','đảm bảo','Khẳng định'}
    FALSE_WORDS_VN = {'no','sai','không thể','không bao giờ đúng','không bao giờ','không','không chính xác','không thể nào'}

    # system
    LOG = os.getenv("LOG", "logs/LOG")

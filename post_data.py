def convert_ID(id: str) -> dict:
    law_id, article_id = id.split("@")
    return {
        "law_id": law_id.strip(),
        "article_id": article_id.strip()
    }


def generate_json_submit():
    relevant_article = {
        "law_id": "",
        "article_id": ""
    }
    data = {
        "question_id": "",
        "question_type": "",
        "text": "",
        "relevant_articles": list(relevant_article)
    }

import datetime
from telegram import Bot
from tabulate import tabulate


def send_telegram_message(model_name, model_parameter, data_name, alpha, top_k_bm25, accuracy, precision, recall, f2, note):
    token, chat_id = """"""
    bot = Bot(token=token)

    current_time = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    headers = ["Date", "Model", "Model Parameter", "Data", "Alpha",
               "Top K BM25", "Accuracy", "Precision", "Recall", "F2", "Note"]
    data = [
        [current_time, model_name, model_parameter, data_name, alpha, top_k_bm25,
            f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f2:.4f}", note]
    ]
    table = tabulate(data, headers=headers, tablefmt="grid")

    message = f"<b>Training Results:</b>\n\n"
    message += f"{table}"

    bot.send_message(chat_id=chat_id, text=message, parse_mode='HTML')

    print("Sent Telegram message successfully.")

import datetime
from telegram import Bot
import my_env


async def send_telegram_message(model_name="", model_parameter="", data_name="", alpha="", top_k_bm25="", accuracy="", precision="", recall="", f2="", note=""):
    bot = Bot(token=my_env.TOKEN_BOT)
    current_time = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    # Format the message as key-value pairs
    message = f"Results:\n\n"
    message += f"Date: {current_time}\n"
    message += f"Model: {model_name}\n"
    message += f"Model Parameter: {model_parameter}\n"
    message += f"Data: {data_name}\n"
    message += f"Alpha: {alpha}\n"
    message += f"Top K BM25: {top_k_bm25}\n"
    message += f"Accuracy: {accuracy}\n"
    message += f"Precision: {precision}\n"
    message += f"Recall: {recall}\n"
    message += f"F2: {f2}\n"
    message += f"Note: {note}"

    await bot.send_message(chat_id=my_env.ID_TELEGRAM,text=message)

    print("Sent Telegram message successfully.")

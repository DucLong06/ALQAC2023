import datetime
from telegram import Bot
import my_env


async def send_telegram_message(model_name="", model_parameter="", data_name="", alpha="", top_k_bm25="", accuracy="", precision="", recall="", f2="", note=""):
    bot = Bot(token=my_env.TOKEN_BOT)
    current_time = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    # Format the message as key-value pairs
    message = f"<b>Training Results:</b>\n\n"
    message += f"<b>Date:</b> {current_time}\n"
    message += f"<b>Model:</b> {model_name}\n"
    message += f"<b>Model Parameter:</b> {model_parameter}\n"
    message += f"<b>Data:</b> {data_name}\n"
    message += f"<b>Alpha:</b> {alpha}\n"
    message += f"<b>Top K BM25:</b> {top_k_bm25}\n"
    message += f"<b>Accuracy:</b> {accuracy}\n"
    message += f"<b>Precision:</b> {precision}\n"
    message += f"<b>Recall:</b> {recall}\n"
    message += f"<b>F2:</b> {f2}\n"
    message += f"<b>Note:</b> {note}"

    await bot.send_message(chat_id=my_env.ID_TELEGRAM,
                           text=message, parse_mode='HTML')

    print("Sent Telegram message successfully.")


import asyncio
import datetime
from telegram import Bot
import my_env
import html


async def send_telegram_message(model_name="", model_parameter="", data_name="", alpha="", top_k_bm25="", accuracy="", precision="", recall="", f2="", note=""):
    bot = Bot(token=my_env.TOKEN_BOT)
    current_time = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')

    message = f"<b>Results:</b>\n\n"
    message += f"<b>Date:</b> {html.escape(str(current_time))}\n"
    message += f"<b>Model:</b> {html.escape(str(model_name))}\n"
    message += f"<b>Model Parameter:</b> {html.escape(str(model_parameter))}\n"
    message += f"<b>Data:</b> {html.escape(str(data_name))}\n"
    message += f"<b>Alpha:</b> {html.escape(str(alpha))}\n"
    message += f"<b>Top K BM25:</b> {html.escape(str(top_k_bm25))}\n"
    message += f"<b>Accuracy:</b> {html.escape(str(accuracy))}\n"
    message += f"<b>Precision:</b> {html.escape(str(precision))}\n"
    message += f"<b>Recall:</b> {html.escape(str(recall))}\n"
    message += f"<b>F2:</b> {html.escape(str(f2))}\n"
    message += f"<b>Note:</b> {html.escape(str(note))}"

    await bot.send_message(chat_id=my_env.ID_TELEGRAM, text=message, parse_mode='HTML')

    print("Sent Telegram message successfully.")



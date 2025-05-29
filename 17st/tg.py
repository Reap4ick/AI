import requests
import re
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime

TOKEN = "7874683003:AAFVUDU4qsqBwvSTcPVX7dd30Rmk3rJGb-c"

def log_interaction(user_input, bot_response):
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        f.write(f"🧑 Користувач: {user_input}\n")
        f.write(f"🤖 Бот: {bot_response}\n")

def query_llama(prompt: str) -> str:
    payload = {
        # "model": "deepseek-r1:14b",
        "model": "deepseek-r1:1.5b",
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "⚠️ Модель не відповіла.")
        else:
            return f"❌ Помилка API: {response.status_code}"
    except requests.RequestException as e:
        return f"❌ Помилка підключення до Ollama API: {str(e)}"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Напиши щось, і я відповім за допомогою нейромережі 🧠",
    )
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    waiting_msg = await update.message.reply_text("🧠 Думаю...")

    raw_reply = query_llama(user_message)
    await waiting_msg.delete()
    await update.message.reply_text(raw_reply , reply_markup=ReplyKeyboardRemove())
    log_interaction(user_message, raw_reply )

def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("✅ Бот запущений...")
    app.run_polling()

if __name__ == "__main__":
    main()

import asyncio
import logging
import sqlite3
import os
import requests
import json
from gtts import gTTS
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from dotenv import load_dotenv

# استدعاء مكتبة Groq
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
API_TOKEN = os.getenv('API_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DB_NAME = 'ielts_bot.db'

logging.basicConfig(level=logging.INFO)

if not API_TOKEN or not GROQ_API_KEY:
    logging.error("API_TOKEN or GROQ_API_KEY is not set. Please check your .env file.")
    exit(1)

# Configure the Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# ---------------------------------------------------------
# Database Functions
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            words_per_day INTEGER DEFAULT 5
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE words ADD COLUMN example_sentence_en TEXT")
        cursor.execute("ALTER TABLE words ADD COLUMN example_sentence_ar TEXT")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()

def get_user_settings(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT words_per_day FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 5

def set_user_settings(user_id, count):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (user_id, words_per_day) VALUES (?, ?)", (user_id, count))
    conn.commit()
    conn.close()

def get_daily_words(limit):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, word, english_def, arabic_meaning, example_sentence_en, example_sentence_ar 
        FROM words WHERE status = 'new' LIMIT ?
    """, (limit,))
    words = cursor.fetchall()
    conn.close()
    return words

def update_word_details(word_id, arabic, example_en, example_ar):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE words 
        SET arabic_meaning = ?, example_sentence_en = ?, example_sentence_ar = ?, status = 'learning'
        WHERE id = ?
    """, (arabic, example_en, example_ar, word_id))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# Translation & API Logic via Groq (Llama 3 70B)
# ---------------------------------------------------------
def fetch_and_translate_sync(word, english_def):
    # 1. Fetch Example from API
    example_sentence_en = "No example available in dictionary."
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            for meaning in data[0].get('meanings', []):
                for defn in meaning.get('definitions', []):
                    if 'example' in defn:
                        example_sentence_en = defn['example']
                        break
                if example_sentence_en != "No example available in dictionary.":
                    break
    except Exception as e:
        logging.error(f"Dictionary API error for '{word}': {e}")

    # 2. Translate using Groq API
    try:
        prompt = f"""
        أنت أستاذ لغة إنجليزية محترف متخصص في مساعدة الطلاب لاجتياز اختبار IELTS.
        
        الكلمة الإنجليزية: {word}
        التعريف الإنجليزي: {english_def}
        المثال الإنجليزي: {example_sentence_en}

        المطلوب:
        1. ترجمة الكلمة إلى اللغة العربية بأدق معنى أكاديمي لها.
        2. ترجمة المثال الإنجليزي إلى لغة عربية فصحى، مفهومة، واحترافية.
        
        يجب أن يكون المخرج بصيغة JSON حصراً بهذا الشكل:
        {{
            "arabic_meaning": "ترجمة الكلمة هنا",
            "example_ar": "ترجمة المثال هنا"
        }}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a translation API that outputs valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
        )
        
        # Parse JSON
        result_text = chat_completion.choices[0].message.content
        result = json.loads(result_text)
        
        arabic_meaning = result.get('arabic_meaning', 'تعذر استخراج الترجمة')
        example_sentence_ar = result.get('example_ar', 'تعذر استخراج المثال')

    except Exception as e:
        logging.error(f"Groq Translation error for '{word}': {e}")
        arabic_meaning = "خطأ في الترجمة"
        example_sentence_ar = "خطأ في الترجمة"

    return arabic_meaning, example_sentence_en, example_sentence_ar


# ---------------------------------------------------------
# Audio Generation
# ---------------------------------------------------------
async def generate_and_send_audio(chat_id, word):
    try:
        tts = gTTS(text=word, lang='en', slow=False)
        filename = f"temp_audio_{word}.mp3"
        tts.save(filename)
        
        voice_file = FSInputFile(filename)
        await bot.send_voice(chat_id=chat_id, voice=voice_file)
        
        os.remove(filename) 
    except Exception as e:
        logging.error(f"Error generating audio for '{word}': {e}")


# ---------------------------------------------------------
# Bot Handlers & Commands
# ---------------------------------------------------------
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    set_user_settings(user_id, 5) 
    
    welcome_text = (
        "Welcome to the IELTS Vocabulary System! 🎓\n"
        "Default setting: 5 words per day.\n\n"
        "Commands:\n"
        "/study - Get your words for today\n"
        "/set_words <number> - Change your daily word count"
    )
    await message.answer(welcome_text)

@dp.message(Command("set_words"))
async def cmd_set_words(message: types.Message, command: CommandObject):
    if not command.args or not command.args.isdigit():
        await message.answer("Invalid input. Please use format: /set_words 10")
        return
    
    new_limit = int(command.args)
    set_user_settings(message.from_user.id, new_limit)
    await message.answer(f"Settings updated. Your daily limit is now {new_limit} words.")

@dp.message(Command("study"))
async def cmd_study(message: types.Message):
    user_id = message.from_user.id
    limit = get_user_settings(user_id)
    
    words = get_daily_words(limit)
    
    if not words:
        await message.answer("No new words available in the database. You have finished them all!")
        return

    await message.answer(f"Fetching your {len(words)} words for today... The AI is preparing professional translations!")

    for w in words:
        word_id = w[0]
        word_text = w[1]
        english_def = w[2]
        arabic_meaning = w[3]
        example_en = w[4]
        example_ar = w[5]
        
        # Translate if not already translated or if previous attempt failed
        if not arabic_meaning or not example_en or not example_ar or arabic_meaning == "خطأ في الترجمة":
            arabic_meaning, example_en, example_ar = await asyncio.to_thread(fetch_and_translate_sync, word_text, english_def)
            update_word_details(word_id, arabic_meaning, example_en, example_ar)
        
        # Format the final message
        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic: {arabic_meaning}\n"
            f"Definition: {english_def}\n\n"
            f"Example (EN): {example_en}\n"
            f"Example (AR): {example_ar}"
        )
        await message.answer(msg_text)
        
        # Send audio
        await generate_and_send_audio(user_id, word_text)
        await asyncio.sleep(1.0)


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
async def main():
    init_db()
    logging.info("Starting bot polling...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped by user.")

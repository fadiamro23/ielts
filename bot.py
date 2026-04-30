import asyncio
import logging
import sqlite3
import os
import requests
from gtts import gTTS
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Configuration
API_TOKEN = os.getenv('API_TOKEN')
DB_NAME = 'ielts_bot.db'

# Using the 600M model to avoid the CUDA Out of Memory error you experienced.
# If you free up enough GPU memory (approx 4GB+), you can change this back to "facebook/nllb-200-1.3B".
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

logging.basicConfig(level=logging.INFO)

if not API_TOKEN:
    logging.error("API_TOKEN is not set. Please check your .env file.")
    exit(1)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# ---------------------------------------------------------
# Initialize NLLB Translation Pipeline Globally
# This loads the model into GPU/RAM only once when starting
# ---------------------------------------------------------
logging.info(f"Loading translation model: {NLLB_MODEL}...")
try:
    translator_pipe = pipeline(
        "translation",
        model=NLLB_MODEL,
        src_lang="eng_Latn",
        tgt_lang="arb_Arab",
        device=0  # Uses GPU 0. Change to -1 for CPU if GPU is completely full.
    )
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load translation model: {e}")
    exit(1)


# ---------------------------------------------------------
# Database Functions
# ---------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Ensure users table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            words_per_day INTEGER DEFAULT 5
        )
    ''')
    
    # Auto-migrate database to add new columns for English and Arabic examples
    # if they don't exist already.
    try:
        cursor.execute("ALTER TABLE words ADD COLUMN example_sentence_en TEXT")
        cursor.execute("ALTER TABLE words ADD COLUMN example_sentence_ar TEXT")
    except sqlite3.OperationalError:
        # Columns already exist, which is fine
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
    # Fetch words that haven't been studied yet
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
    # Update word with translations and mark status as 'learning'
    cursor.execute("""
        UPDATE words 
        SET arabic_meaning = ?, example_sentence_en = ?, example_sentence_ar = ?, status = 'learning'
        WHERE id = ?
    """, (arabic, example_en, example_ar, word_id))
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# Translation & API Logic
# ---------------------------------------------------------
def fetch_and_translate_sync(word):
    """
    This function blocks the thread because of the model pipeline.
    It fetches the English example via API, then uses NLLB to translate the word and the example.
    """
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

    # 2. Translate using local NLLB model
    try:
        # Translate the main word
        arabic_meaning = translator_pipe(word)[0]['translation_text']
        
        # Translate the example sentence if it exists
        if example_sentence_en != "No example available in dictionary.":
            example_sentence_ar = translator_pipe(example_sentence_en)[0]['translation_text']
        else:
            example_sentence_ar = "لا يوجد مثال متوفر لترجمته."
            
    except Exception as e:
        logging.error(f"NLLB Translation error for '{word}': {e}")
        arabic_meaning = "Translation error"
        example_sentence_ar = "Translation error"

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
        
        os.remove(filename)  # Clean up after sending
    except Exception as e:
        logging.error(f"Error generating audio for '{word}': {e}")


# ---------------------------------------------------------
# Bot Handlers & Commands
# ---------------------------------------------------------
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    set_user_settings(user_id, 5) # Default limit
    
    welcome_text = (
        "Welcome to the IELTS Vocabulary System.\n"
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

    await message.answer(f"Fetching your {len(words)} words for today... Please wait while the AI translates them.")

    for w in words:
        word_id = w[0]
        word_text = w[1]
        english_def = w[2]
        arabic_meaning = w[3]
        example_en = w[4]
        example_ar = w[5]
        
        # If translation or examples are missing, process them in a background thread
        if not arabic_meaning or not example_en or not example_ar:
            # asyncio.to_thread prevents the bot from freezing while the heavy AI model is working
            arabic_meaning, example_en, example_ar = await asyncio.to_thread(fetch_and_translate_sync, word_text)
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
        
        # Generate and send pronunciation
        await generate_and_send_audio(user_id, word_text)
        
        # Short pause to respect Telegram API limits
        await asyncio.sleep(1.5)


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

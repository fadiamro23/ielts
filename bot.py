import asyncio
import logging
import sqlite3
import os
import requests
from gtts import gTTS
from deep_translator import GoogleTranslator
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_TOKEN = os.getenv('API_TOKEN')
DB_NAME = 'ielts_bot.db'

logging.basicConfig(level=logging.INFO)

# Check if token is loaded properly
if not API_TOKEN:
    logging.error("API_TOKEN is not set. Please check your .env file.")
    exit(1)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Fetch user settings
def get_user_settings(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT words_per_day FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 5

# Update user settings
def set_user_settings(user_id, count):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO users (user_id, words_per_day) VALUES (?, ?)", (user_id, count))
    conn.commit()
    conn.close()

# Fetch daily words from the database
def get_daily_words(limit):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, word, english_def, arabic_meaning, example_sentence FROM words WHERE status = 'new' LIMIT ?", (limit,))
    words = cursor.fetchall()
    conn.close()
    return words

# Update word with Arabic meaning and Example in database
def update_word_details(word_id, arabic, example):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE words SET arabic_meaning = ?, example_sentence = ? WHERE id = ?", (arabic, example, word_id))
    conn.commit()
    conn.close()

# Helper function to fetch translation and example dynamically
def fetch_translation_and_example(word):
    # 1. Translate to Arabic
    try:
        arabic_meaning = GoogleTranslator(source='en', target='ar').translate(word)
    except Exception as e:
        logging.error(f"Translation error for {word}: {e}")
        arabic_meaning = "Translation unavailable"

    # 2. Fetch Example from Free Dictionary API
    example_sentence = "No example available in dictionary."
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Parse JSON to find the first available example
            for meaning in data[0].get('meanings', []):
                for defn in meaning.get('definitions', []):
                    if 'example' in defn:
                        example_sentence = defn['example']
                        break
                if example_sentence != "No example available in dictionary.":
                    break
    except Exception as e:
        logging.error(f"Dictionary API error for {word}: {e}")

    return arabic_meaning, example_sentence

# Generate audio file using gTTS and send it
async def generate_and_send_audio(chat_id, word):
    try:
        tts = gTTS(text=word, lang='en', slow=False)
        filename = f"temp_audio_{word}.mp3"
        tts.save(filename)
        
        voice_file = FSInputFile(filename)
        await bot.send_voice(chat_id=chat_id, voice=voice_file)
        
        os.remove(filename)
    except Exception as e:
        logging.error(f"Error generating audio for {word}: {e}")

# Command: /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    set_user_settings(user_id, 5) 
    
    welcome_text = (
        "Welcome to the IELTS Vocabulary System.\n"
        "Default setting: 5 words per day.\n\n"
        "Commands:\n"
        "/study - Get your words for today\n"
        "/set_words <number> - Change daily word count"
    )
    await message.answer(welcome_text)

# Command: /set_words <number>
@dp.message(Command("set_words"))
async def cmd_set_words(message: types.Message, command: CommandObject):
    if not command.args or not command.args.isdigit():
        await message.answer("Invalid input. Please use format: /set_words 10")
        return
    
    new_limit = int(command.args)
    set_user_settings(message.from_user.id, new_limit)
    await message.answer(f"Settings updated. Your daily limit is now {new_limit} words.")

# Command: /study
@dp.message(Command("study"))
async def cmd_study(message: types.Message):
    user_id = message.from_user.id
    limit = get_user_settings(user_id)
    
    words = get_daily_words(limit)
    
    if not words:
        await message.answer("No new words available in the database.")
        return

    await message.answer(f"Fetching your {len(words)} words for today...")

    for w in words:
        word_id, word_text, english_def, arabic_meaning, example_sentence = w
        
        # If translation or example is missing in DB, fetch and update
        if not arabic_meaning or not example_sentence:
            arabic_meaning, example_sentence = fetch_translation_and_example(word_text)
            update_word_details(word_id, arabic_meaning, example_sentence)
        
        # Format the text message with all details
        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic: {arabic_meaning}\n"
            f"Definition: {english_def}\n"
            f"Example: {example_sentence}"
        )
        await message.answer(msg_text)
        
        # Generate and send pronunciation
        await generate_and_send_audio(user_id, word_text)
        
        # Pause briefly to prevent hitting Telegram API limits
        await asyncio.sleep(1.5)

# Main function
async def main():
    logging.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")

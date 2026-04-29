import asyncio
import logging
import sqlite3
import os
from gtts import gTTS
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile

# Configuration
API_TOKEN = '8755826845:AAF4BuprTKC451oYeXh4z8dQatr8c-7bA1s'
DB_NAME = 'ielts_bot.db'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Initialize Database and Tables
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Create users table to store custom words_per_day
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            words_per_day INTEGER DEFAULT 5
        )
    ''')
    conn.commit()
    conn.close()

# Fetch user settings (default to 5 if not found)
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
    cursor.execute("SELECT id, word, english_def FROM words WHERE status = 'new' LIMIT ?", (limit,))
    words = cursor.fetchall()
    conn.close()
    return words

# Generate audio file using gTTS and send it
async def generate_and_send_audio(chat_id, word):
    try:
        # Create audio
        tts = gTTS(text=word, lang='en', slow=False)
        filename = f"temp_audio_{word}.mp3"
        tts.save(filename)
        
        # Send voice message
        voice_file = FSInputFile(filename)
        await bot.send_voice(chat_id=chat_id, voice=voice_file)
        
        # Remove file after sending to free up space
        os.remove(filename)
    except Exception as e:
        logging.error(f"Error generating audio for {word}: {e}")

# Command: /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    set_user_settings(user_id, 5) # Set default limit
    
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
        word_id, word_text, english_def = w
        
        # Format the text message
        msg_text = f"Word: {word_text}\nDefinition: {english_def}"
        await message.answer(msg_text)
        
        # Generate and send pronunciation
        await generate_and_send_audio(user_id, word_text)
        
        # Pause briefly to prevent hitting Telegram API limits
        await asyncio.sleep(1.5)

# Main function
async def main():
    init_db()
    logging.info("Starting bot...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")
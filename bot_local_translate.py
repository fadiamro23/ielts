import asyncio
import logging
import os
import re
import sqlite3
import tempfile

import requests
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from dotenv import load_dotenv
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

# Load environment variables from .env file
load_dotenv()

# =========================
# Configuration
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_TOKEN = os.getenv("API_TOKEN")
DB_NAME = os.getenv("DB_NAME", os.path.join(BASE_DIR, "ielts_bot.db"))

# Choose translation backend:
# nllb   -> facebook/nllb-200-1.3B or facebook/nllb-200-distilled-600M
# madlad -> google/madlad400-3b-mt
TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "nllb").lower().strip()

if TRANSLATION_BACKEND == "madlad":
    DEFAULT_MODEL = "google/madlad400-3b-mt"
else:
    DEFAULT_MODEL = "facebook/nllb-200-1.3B"

TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", DEFAULT_MODEL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

if not API_TOKEN:
    logging.error("API_TOKEN is not set. Please check your .env file.")
    raise SystemExit(1)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# =========================
# Database helpers
# =========================
def get_db_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    """
    Makes your existing database compatible with this bot.
    Your uploaded DB has users(user_id, is_active, joined_date),
    while the original code expects users.words_per_day.
    This function safely adds the missing column.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            is_active BOOLEAN DEFAULT 1,
            joined_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            words_per_day INTEGER DEFAULT 5
        )
    """)

    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]

    if "words_per_day" not in columns:
        logging.info("Adding missing column users.words_per_day")
        cursor.execute("ALTER TABLE users ADD COLUMN words_per_day INTEGER DEFAULT 5")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            english_def TEXT,
            arabic_meaning TEXT,
            example_sentence TEXT,
            status TEXT DEFAULT 'new',
            learning_stage INTEGER DEFAULT 0,
            next_review_date DATE
        )
    """)

    conn.commit()
    conn.close()


def get_user_settings(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT words_per_day FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result and result[0] else 5


def set_user_settings(user_id, count):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
    exists = cursor.fetchone()

    if exists:
        cursor.execute(
            "UPDATE users SET words_per_day = ?, is_active = 1 WHERE user_id = ?",
            (count, user_id)
        )
    else:
        cursor.execute(
            "INSERT INTO users (user_id, words_per_day, is_active) VALUES (?, ?, 1)",
            (user_id, count)
        )

    conn.commit()
    conn.close()


def get_daily_words(limit):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, word, english_def, arabic_meaning, example_sentence
        FROM words
        WHERE status = 'new'
        LIMIT ?
        """,
        (limit,)
    )
    words = cursor.fetchall()
    conn.close()
    return words


def update_word_details(word_id, arabic, example):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE words SET arabic_meaning = ?, example_sentence = ? WHERE id = ?",
        (arabic, example, word_id)
    )
    conn.commit()
    conn.close()


# =========================
# Local translation engine
# =========================
class LocalTranslator:
    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        logging.info("Loading translation backend=%s model=%s device=%s", backend, model_name, self.device)

        # IMPORTANT:
        # Do not load the model first and then call .to("cuda") for large models.
        # On 24GB cards this can briefly duplicate weights in GPU memory and cause OOM.
        # device_map="auto" places weights directly and more safely.
        load_kwargs = {
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": False,
        }
        if self.device == "cuda":
            load_kwargs["device_map"] = {"": 0}

        if self.backend == "madlad":
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            if self.device != "cuda":
                self.model.to(self.device)

        self.model.eval()
        logging.info("Translation model loaded successfully.")

    def translate_en_to_ar(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        with torch.no_grad():
            if self.backend == "madlad":
                return self._translate_madlad(text)
            return self._translate_nllb(text)

    def _translate_nllb(self, text: str) -> str:
        model_device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(model_device)

        outputs = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("arb_Arab"),
            max_new_tokens=128,
            num_beams=4,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()

    def _translate_madlad(self, text: str) -> str:
        # MADLAD target-language tag for Arabic
        input_text = "<2ar> " + text

        model_device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(model_device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


translator = None


def fix_ielts_terms(text: str) -> str:
    """
    Small post-processing layer for IELTS vocabulary.
    Add more replacements as you notice repeated model mistakes.
    """
    replacements = {
        "إي إل تي إس": "الآيلتس",
        "آي إلتس": "الآيلتس",
        "IELTS": "الآيلتس",
        "درجة تقريري": "درجتي في قسم المحادثة",
        "درجة اللغة": "الدرجة",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text


def local_translate_en_to_ar(text: str) -> str:
    global translator
    if translator is None:
        translator = LocalTranslator(TRANSLATION_BACKEND, TRANSLATION_MODEL)

    translated = translator.translate_en_to_ar(text)
    return fix_ielts_terms(translated)


# =========================
# Dictionary / examples
# =========================
def fetch_example_from_dictionary(word):
    example_sentence = "No example available in dictionary."

    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()

            for meaning in data[0].get("meanings", []):
                for defn in meaning.get("definitions", []):
                    if "example" in defn:
                        example_sentence = defn["example"]
                        break

                if example_sentence != "No example available in dictionary.":
                    break

    except Exception as e:
        logging.error("Dictionary API error for %s: %s", word, e)

    return example_sentence


def fetch_translation_and_example(word):
    try:
        arabic_meaning = local_translate_en_to_ar(word)
    except Exception as e:
        logging.exception("Local translation error for %s: %s", word, e)
        arabic_meaning = "Translation unavailable"

    example_sentence = fetch_example_from_dictionary(word)
    return arabic_meaning, example_sentence


# =========================
# Audio
# =========================
def safe_filename(text):
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return text[:40] or "audio"


async def generate_and_send_audio(chat_id, word):
    filename = None

    try:
        tts = gTTS(text=word, lang="en", slow=False)

        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f"ielts_{safe_filename(word)}.mp3")

        tts.save(filename)

        voice_file = FSInputFile(filename)
        await bot.send_voice(chat_id=chat_id, voice=voice_file)

    except Exception as e:
        logging.error("Error generating audio for %s: %s", word, e)

    finally:
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
            except OSError:
                pass


# =========================
# Commands
# =========================
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    set_user_settings(user_id, 5)

    welcome_text = (
        "Welcome to the IELTS Vocabulary System.\n"
        "Default setting: 5 words per day.\n\n"
        "Commands:\n"
        "/study - Get your words for today\n"
        "/set_words <number> - Change daily word count\n"
        "/translate <text> - Translate English text to Arabic using the local AI model\n\n"
        f"Translation backend: {TRANSLATION_BACKEND}\n"
        f"Model: {TRANSLATION_MODEL}"
    )

    await message.answer(welcome_text)


@dp.message(Command("set_words"))
async def cmd_set_words(message: types.Message, command: CommandObject):
    if not command.args or not command.args.isdigit():
        await message.answer("Invalid input. Please use format: /set_words 10")
        return

    new_limit = int(command.args)

    if new_limit < 1 or new_limit > 50:
        await message.answer("Please choose a number between 1 and 50.")
        return

    set_user_settings(message.from_user.id, new_limit)
    await message.answer(f"Settings updated. Your daily limit is now {new_limit} words.")


@dp.message(Command("translate"))
async def cmd_translate(message: types.Message, command: CommandObject):
    if not command.args:
        await message.answer("Please use: /translate I want to improve my IELTS speaking band score.")
        return

    await message.answer("Translating with the local model...")

    try:
        translated = await asyncio.to_thread(local_translate_en_to_ar, command.args)
        await message.answer(translated)
    except Exception as e:
        logging.exception("Translate command error: %s", e)
        await message.answer("Translation failed. Please check the server logs.")


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

        if not arabic_meaning or not example_sentence:
            arabic_meaning, example_sentence = await asyncio.to_thread(
                fetch_translation_and_example,
                word_text
            )
            update_word_details(word_id, arabic_meaning, example_sentence)

        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic: {arabic_meaning}\n"
            f"Definition: {english_def or 'No definition available.'}\n"
            f"Example: {example_sentence or 'No example available.'}"
        )

        await message.answer(msg_text)
        await generate_and_send_audio(user_id, word_text)
        await asyncio.sleep(1.5)


# =========================
# Main
# =========================
async def main():
    init_db()

    logging.info("Starting bot...")
    logging.info("DB: %s", DB_NAME)
    logging.info("Translation backend: %s", TRANSLATION_BACKEND)
    logging.info("Translation model: %s", TRANSLATION_MODEL)

    # Load the model at startup, not after the first user message.
    # This makes the first Telegram response faster after startup is complete.
    await asyncio.to_thread(local_translate_en_to_ar, "test")

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")

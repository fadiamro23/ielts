# bot_local_translate.py
# IELTS Vocabulary Telegram Bot with local translation backend.
# Supports:
#   - NLLB:   facebook/nllb-200-1.3B or facebook/nllb-200-distilled-600M
#   - MADLAD: google/madlad400-3b-mt
#
# Recommended run:
#   export TRANSLATION_BACKEND=nllb
#   export TRANSLATION_MODEL=facebook/nllb-200-1.3B
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   CUDA_VISIBLE_DEVICES=2 python bot_local_translate.py

import os

# Prevent Transformers from trying to load TensorFlow / vision extras.
# These must be set before importing transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_FLAX", "0")

import asyncio
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, CommandObject
from aiogram.types import FSInputFile
from dotenv import load_dotenv
from gtts import gTTS

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------

API_TOKEN = os.getenv("API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = os.getenv("DB_NAME", str(BASE_DIR / "ielts_bot.db"))

TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "nllb").strip().lower()
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-1.3B").strip()

# Translation language settings
NLLB_SOURCE_LANG = os.getenv("NLLB_SOURCE_LANG", "eng_Latn")
NLLB_TARGET_LANG = os.getenv("NLLB_TARGET_LANG", "arb_Arab")
MADLAD_TARGET_TAG = os.getenv("MADLAD_TARGET_TAG", "<2ar>")

# Generation settings
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

# Optional: set to 1 if you want to skip preloading at startup
LAZY_LOAD_TRANSLATOR = os.getenv("LAZY_LOAD_TRANSLATOR", "0").strip() == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

if not API_TOKEN:
    logging.error("API_TOKEN is not set. Please check your .env file.")
    raise SystemExit(1)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

_TRANSLATOR = None
_TRANSLATOR_LOCK = asyncio.Lock()


# -----------------------------
# Database helpers
# -----------------------------

def get_connection():
    return sqlite3.connect(DB_NAME)


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def init_db():
    """
    Makes the database compatible with this bot without deleting existing data.
    It creates missing tables and adds words_per_day if missing.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            words_per_day INTEGER DEFAULT 5
        )
        """
    )

    if not column_exists(cursor, "users", "words_per_day"):
        cursor.execute("ALTER TABLE users ADD COLUMN words_per_day INTEGER DEFAULT 5")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            english_def TEXT,
            arabic_meaning TEXT,
            example_sentence TEXT,
            status TEXT DEFAULT 'new'
        )
        """
    )

    # Safety upgrades for older DB versions
    for col, ddl in {
        "english_def": "ALTER TABLE words ADD COLUMN english_def TEXT",
        "arabic_meaning": "ALTER TABLE words ADD COLUMN arabic_meaning TEXT",
        "example_sentence": "ALTER TABLE words ADD COLUMN example_sentence TEXT",
        "status": "ALTER TABLE words ADD COLUMN status TEXT DEFAULT 'new'",
    }.items():
        if not column_exists(cursor, "words", col):
            cursor.execute(ddl)

    conn.commit()
    conn.close()


def get_user_settings(user_id: int) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT words_per_day FROM users WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return int(result[0]) if result and result[0] else 5


def set_user_settings(user_id: int, count: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO users (user_id, words_per_day)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET words_per_day = excluded.words_per_day
        """,
        (user_id, count),
    )
    conn.commit()
    conn.close()


def get_daily_words(limit: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, word, english_def, arabic_meaning, example_sentence
        FROM words
        WHERE COALESCE(status, 'new') = 'new'
        LIMIT ?
        """,
        (limit,),
    )
    words = cursor.fetchall()
    conn.close()
    return words


def update_word_details(word_id: int, arabic: str, example: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE words
        SET arabic_meaning = ?, example_sentence = ?
        WHERE id = ?
        """,
        (arabic, example, word_id),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Local translation backend
# -----------------------------

class LocalTranslator:
    """
    Local translator.

    Important memory decision:
    - Do NOT use device_map here for NLLB 1.3B on 24GB GPUs with your setup.
    - Do NOT pass dtype=... because transformers 4.46.3 needs torch_dtype=...
    - Load on CPU first, convert to FP16, then move once to CUDA.
      This avoids the repeated OOM you saw during accelerate/device_map loading.
    """

    def __init__(self, backend: str, model_name: str):
        self.backend = backend.strip().lower()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(
            "Loading translation backend=%s model=%s device=%s",
            self.backend,
            self.model_name,
            self.device,
        )

        if self.backend not in {"nllb", "madlad"}:
            raise ValueError("TRANSLATION_BACKEND must be either 'nllb' or 'madlad'.")

        # Load on CPU first.
        # Keep use_safetensors=False for NLLB because your logs showed problematic
        # safetensors/PR resolution attempts.
        load_kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": False,
        }

        if self.backend == "nllb":
            load_kwargs["use_safetensors"] = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        else:
            # Lazy import so NLLB users do not depend on T5 classes at startup.
            from transformers import T5ForConditionalGeneration, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        # Move once to GPU after loading.
        if self.device == "cuda":
            logging.info("Moving model to CUDA in FP16...")
            self.model = self.model.half().to(self.device)
        else:
            logging.warning("CUDA is not available. Running translation on CPU.")
            self.model = self.model.to(self.device)

        self.model.eval()
        logging.info("Translation model loaded successfully.")

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def translate_en_to_ar(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self.backend == "madlad":
            return self._translate_madlad_to_ar(text)

        return self._translate_nllb_to_ar(text)

    def _translate_nllb_to_ar(self, text: str) -> str:
        # NLLB language codes
        self.tokenizer.src_lang = NLLB_SOURCE_LANG

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        inputs = self._to_device(inputs)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(NLLB_TARGET_LANG)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )

        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return fix_ielts_terms(translated.strip())

    def _translate_madlad_to_ar(self, text: str) -> str:
        # MADLAD uses target tag prefix, e.g. <2ar>
        input_text = f"{MADLAD_TARGET_TAG} {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        inputs = self._to_device(inputs)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )

        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fix_ielts_terms(translated.strip())


def fix_ielts_terms(text: str) -> str:
    """
    Small post-processing layer for common IELTS translation issues.
    Keep this conservative so it does not damage normal Arabic.
    """
    replacements = {
        "إي إل تي إس": "الآيلتس",
        "إيلتس": "الآيلتس",
        "IELTS": "الآيلتس",
        "درجة تقريري": "درجتي في قسم المحادثة",
        "درجة اللغة الآيلتس": "درجتي في اختبار الآيلتس",
        "درجة اللغة في الآيلتس": "درجتي في اختبار الآيلتس",
        "درجة التحدث": "درجة المحادثة",
        "درجة الكلام": "درجة المحادثة",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    return text


async def get_translator() -> LocalTranslator:
    global _TRANSLATOR

    if _TRANSLATOR is not None:
        return _TRANSLATOR

    async with _TRANSLATOR_LOCK:
        if _TRANSLATOR is None:
            _TRANSLATOR = await asyncio.to_thread(
                LocalTranslator,
                TRANSLATION_BACKEND,
                TRANSLATION_MODEL,
            )
        return _TRANSLATOR


async def local_translate_en_to_ar(text: str) -> str:
    translator = await get_translator()
    return await asyncio.to_thread(translator.translate_en_to_ar, text)


# -----------------------------
# External helpers
# -----------------------------

def fetch_dictionary_example(word: str) -> str:
    example_sentence = "No example available in dictionary."

    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=8)

        if response.status_code == 200:
            data = response.json()

            if isinstance(data, list) and data:
                for meaning in data[0].get("meanings", []):
                    for definition in meaning.get("definitions", []):
                        example = definition.get("example")
                        if example:
                            return example

    except Exception as e:
        logging.error("Dictionary API error for %s: %s", word, e)

    return example_sentence


async def fetch_translation_and_example(word: str) -> Tuple[str, str]:
    """
    Replaces GoogleTranslator with the local model.
    """
    try:
        arabic_meaning = await local_translate_en_to_ar(word)
    except Exception as e:
        logging.exception("Local translation error for %s: %s", word, e)
        arabic_meaning = "Translation unavailable"

    example_sentence = await asyncio.to_thread(fetch_dictionary_example, word)

    return arabic_meaning, example_sentence


async def generate_and_send_audio(chat_id: int, word: str):
    """
    Generate English pronunciation using gTTS.
    This still uses Google's TTS service online. It is separate from translation.
    """
    filename = None

    try:
        safe_word = "".join(c for c in word if c.isalnum() or c in ("-", "_"))[:80]
        if not safe_word:
            safe_word = "word"

        with tempfile.NamedTemporaryFile(
            suffix=f"_{safe_word}.mp3",
            delete=False,
            dir=str(BASE_DIR),
        ) as tmp:
            filename = tmp.name

        tts = gTTS(text=word, lang="en", slow=False)
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


# -----------------------------
# Bot commands
# -----------------------------

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
        "/translate <text> - Translate English text to Arabic locally"
    )
    await message.answer(welcome_text)


@dp.message(Command("set_words"))
async def cmd_set_words(message: types.Message, command: CommandObject):
    if not command.args or not command.args.strip().isdigit():
        await message.answer("Invalid input. Please use format: /set_words 10")
        return

    new_limit = int(command.args.strip())

    if new_limit < 1 or new_limit > 100:
        await message.answer("Please choose a number between 1 and 100.")
        return

    set_user_settings(message.from_user.id, new_limit)
    await message.answer(f"Settings updated. Your daily limit is now {new_limit} words.")


@dp.message(Command("translate"))
async def cmd_translate(message: types.Message, command: CommandObject):
    if not command.args or not command.args.strip():
        await message.answer("Please use format:\n/translate I want to improve my IELTS speaking band score.")
        return

    text = command.args.strip()

    await message.answer("Translating locally...")

    try:
        translated = await local_translate_en_to_ar(text)
        await message.answer(f"Arabic:\n{translated}")
    except Exception as e:
        logging.exception("Translation command failed: %s", e)
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

    for row in words:
        word_id, word_text, english_def, arabic_meaning, example_sentence = row

        if not word_text:
            continue

        if not arabic_meaning or not example_sentence:
            arabic_meaning, example_sentence = await fetch_translation_and_example(word_text)
            update_word_details(word_id, arabic_meaning, example_sentence)

        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic: {arabic_meaning or 'Translation unavailable'}\n"
            f"Definition: {english_def or 'No definition available.'}\n"
            f"Example: {example_sentence or 'No example available.'}"
        )

        await message.answer(msg_text)

        await generate_and_send_audio(user_id, word_text)

        # Pause briefly to prevent Telegram API limits
        await asyncio.sleep(1.5)


@dp.message()
async def fallback_message(message: types.Message):
    await message.answer(
        "Use /study for daily IELTS words or /translate <text> for local English-to-Arabic translation."
    )


# -----------------------------
# Main
# -----------------------------

async def main():
    logging.info("Starting bot...")
    logging.info("DB: %s", DB_NAME)
    logging.info("Translation backend: %s", TRANSLATION_BACKEND)
    logging.info("Translation model: %s", TRANSLATION_MODEL)

    init_db()

    if not LAZY_LOAD_TRANSLATOR:
        # Preload translator before polling, so first user request is not delayed.
        # If you want faster bot startup, run with:
        #   export LAZY_LOAD_TRANSLATOR=1
        await get_translator()

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")

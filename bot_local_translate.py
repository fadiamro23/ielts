# bot_local_translate.py
# IELTS Vocabulary Telegram Bot with local English -> Arabic translation.
#
# What this version does:
# - Translates IELTS word to Arabic using local NLLB/MADLAD.
# - Translates the English definition to Arabic.
# - Gets/generates an English example sentence.
# - Translates the example sentence to Arabic.
# - Shows all of that in /study.
#
# Recommended run:
#   export TRANSLATION_BACKEND=nllb
#   export TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   CUDA_VISIBLE_DEVICES=1 python bot_local_translate.py
#
# For stronger model if GPU is free:
#   export TRANSLATION_MODEL=facebook/nllb-200-1.3B

import os

# Prevent Transformers from trying to load TensorFlow / Flax.
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

# -----------------------------
# Environment / config
# -----------------------------

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = os.getenv("DB_NAME", str(BASE_DIR / "ielts_bot.db"))

TRANSLATION_BACKEND = os.getenv("TRANSLATION_BACKEND", "nllb").strip().lower()
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M").strip()

NLLB_SOURCE_LANG = os.getenv("NLLB_SOURCE_LANG", "eng_Latn")
NLLB_TARGET_LANG = os.getenv("NLLB_TARGET_LANG", "arb_Arab")
MADLAD_TARGET_TAG = os.getenv("MADLAD_TARGET_TAG", "<2ar>")

MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "512"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

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


def add_column_if_missing(cursor: sqlite3.Cursor, table_name: str, column_name: str, ddl: str):
    if not column_exists(cursor, table_name, column_name):
        logging.info("Adding missing column %s.%s", table_name, column_name)
        cursor.execute(ddl)


def init_db():
    """
    Upgrades your existing DB safely without deleting data.
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

    add_column_if_missing(
        cursor,
        "users",
        "words_per_day",
        "ALTER TABLE users ADD COLUMN words_per_day INTEGER DEFAULT 5",
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            english_def TEXT,
            english_def_arabic TEXT,
            arabic_meaning TEXT,
            example_sentence TEXT,
            example_sentence_arabic TEXT,
            status TEXT DEFAULT 'new'
        )
        """
    )

    # Upgrade older DB structure
    add_column_if_missing(cursor, "words", "english_def", "ALTER TABLE words ADD COLUMN english_def TEXT")
    add_column_if_missing(cursor, "words", "english_def_arabic", "ALTER TABLE words ADD COLUMN english_def_arabic TEXT")
    add_column_if_missing(cursor, "words", "arabic_meaning", "ALTER TABLE words ADD COLUMN arabic_meaning TEXT")
    add_column_if_missing(cursor, "words", "example_sentence", "ALTER TABLE words ADD COLUMN example_sentence TEXT")
    add_column_if_missing(cursor, "words", "example_sentence_arabic", "ALTER TABLE words ADD COLUMN example_sentence_arabic TEXT")
    add_column_if_missing(cursor, "words", "status", "ALTER TABLE words ADD COLUMN status TEXT DEFAULT 'new'")

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
        SELECT
            id,
            word,
            english_def,
            english_def_arabic,
            arabic_meaning,
            example_sentence,
            example_sentence_arabic
        FROM words
        WHERE COALESCE(status, 'new') = 'new'
        LIMIT ?
        """,
        (limit,),
    )
    words = cursor.fetchall()
    conn.close()
    return words


def update_word_learning_details(
    word_id: int,
    arabic_meaning: str,
    english_def: str,
    english_def_arabic: str,
    example_sentence: str,
    example_sentence_arabic: str,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE words
        SET
            arabic_meaning = ?,
            english_def = ?,
            english_def_arabic = ?,
            example_sentence = ?,
            example_sentence_arabic = ?
        WHERE id = ?
        """,
        (
            arabic_meaning,
            english_def,
            english_def_arabic,
            example_sentence,
            example_sentence_arabic,
            word_id,
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Local translation backend
# -----------------------------

class LocalTranslator:
    """
    Local translator for English -> Arabic.

    Notes for your server:
    - Use CUDA_VISIBLE_DEVICES to choose a FREE GPU, e.g. CUDA_VISIBLE_DEVICES=1.
    - With transformers 4.46.3 use torch_dtype, not dtype.
    - This code avoids device_map because it caused OOM in your environment.
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

        load_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": False,
        }

        if self.backend == "nllb":
            # This avoids the problematic safetensors PR path seen in your logs.
            load_kwargs["use_safetensors"] = False
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        else:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_kwargs,
            )

        if self.device == "cuda":
            first_param = next(self.model.parameters())
            logging.info("Model dtype before CUDA move: %s", first_param.dtype)
            logging.info("Moving model to CUDA...")
            self.model = self.model.to(self.device)
        else:
            logging.warning("CUDA is not available. Running translation on CPU.")
            self.model = self.model.to(self.device)

        self.model.eval()
        logging.info("Translation model loaded successfully.")

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def translate_en_to_ar(self, text: str) -> str:
        text = clean_text(text)
        if not text:
            return ""

        if self.backend == "madlad":
            return self._translate_madlad_to_ar(text)

        return self._translate_nllb_to_ar(text)

    def _translate_nllb_to_ar(self, text: str) -> str:
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


def clean_text(text: Optional[str]) -> str:
    return " ".join((text or "").strip().split())


def fix_ielts_terms(text: str) -> str:
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
# Dictionary / examples
# -----------------------------

def fetch_dictionary_definition_and_example(word: str) -> Tuple[str, str]:
    """
    Returns:
      english_definition, english_example

    Priority:
    1. dictionaryapi.dev definition/example
    2. fallback IELTS-style example
    """
    fallback_definition = ""
    fallback_example = make_fallback_english_example(word)

    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=8)

        if response.status_code != 200:
            return fallback_definition, fallback_example

        data = response.json()
        if not isinstance(data, list) or not data:
            return fallback_definition, fallback_example

        best_definition = ""
        best_example = ""

        for meaning in data[0].get("meanings", []):
            for definition_obj in meaning.get("definitions", []):
                definition_text = clean_text(definition_obj.get("definition"))
                example_text = clean_text(definition_obj.get("example"))

                if definition_text and not best_definition:
                    best_definition = definition_text

                if example_text and not best_example:
                    best_example = example_text

                if best_definition and best_example:
                    return best_definition, best_example

        return best_definition or fallback_definition, best_example or fallback_example

    except Exception as e:
        logging.error("Dictionary API error for %s: %s", word, e)
        return fallback_definition, fallback_example


def make_fallback_english_example(word: str) -> str:
    word = clean_text(word)
    if not word:
        return "This word is commonly used in academic English."

    # A simple safe fallback when the dictionary API has no example.
    return f"The student learned how to use the word '{word}' correctly in an IELTS sentence."


async def build_learning_card(
    word: str,
    english_def_from_db: Optional[str],
    arabic_meaning_from_db: Optional[str],
    english_def_arabic_from_db: Optional[str],
    example_from_db: Optional[str],
    example_arabic_from_db: Optional[str],
) -> Tuple[str, str, str, str, str]:
    """
    Builds:
      arabic_meaning,
      english_definition,
      arabic_definition,
      english_example,
      arabic_example
    """
    word = clean_text(word)

    english_definition = clean_text(english_def_from_db)
    english_example = clean_text(example_from_db)

    if not english_definition or not english_example:
        fetched_definition, fetched_example = await asyncio.to_thread(
            fetch_dictionary_definition_and_example,
            word,
        )

        if not english_definition and fetched_definition:
            english_definition = fetched_definition

        if not english_example and fetched_example:
            english_example = fetched_example

    if not english_definition:
        english_definition = "No English definition available."

    if not english_example:
        english_example = make_fallback_english_example(word)

    arabic_meaning = clean_text(arabic_meaning_from_db)
    arabic_definition = clean_text(english_def_arabic_from_db)
    arabic_example = clean_text(example_arabic_from_db)

    translation_tasks = []

    if not arabic_meaning:
        translation_tasks.append(("word", local_translate_en_to_ar(word)))

    if not arabic_definition and english_definition != "No English definition available.":
        translation_tasks.append(("definition", local_translate_en_to_ar(english_definition)))

    if not arabic_example:
        translation_tasks.append(("example", local_translate_en_to_ar(english_example)))

    if translation_tasks:
        results = await asyncio.gather(
            *(task for _, task in translation_tasks),
            return_exceptions=True,
        )

        for (name, _), result in zip(translation_tasks, results):
            if isinstance(result, Exception):
                logging.error("Translation failed for %s/%s: %s", name, word, result)
                translated_text = "Translation unavailable"
            else:
                translated_text = clean_text(str(result))

            if name == "word":
                arabic_meaning = translated_text
            elif name == "definition":
                arabic_definition = translated_text
            elif name == "example":
                arabic_example = translated_text

    if not arabic_meaning:
        arabic_meaning = "Translation unavailable"

    if not arabic_definition:
        arabic_definition = "Translation unavailable"

    if not arabic_example:
        arabic_example = "Translation unavailable"

    return (
        arabic_meaning,
        english_definition,
        arabic_definition,
        english_example,
        arabic_example,
    )


# -----------------------------
# Audio
# -----------------------------

async def generate_and_send_audio(chat_id: int, word: str):
    """
    Generate English pronunciation using gTTS.
    Translation is local, but gTTS itself is still an online Google TTS service.
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
        "/study - Get your IELTS words for today\n"
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

    await message.answer(f"Fetching your {len(words)} IELTS words for today...")

    for row in words:
        (
            word_id,
            word_text,
            english_def,
            english_def_arabic,
            arabic_meaning,
            example_sentence,
            example_sentence_arabic,
        ) = row

        word_text = clean_text(word_text)
        if not word_text:
            continue

        (
            arabic_meaning,
            english_def,
            english_def_arabic,
            example_sentence,
            example_sentence_arabic,
        ) = await build_learning_card(
            word=word_text,
            english_def_from_db=english_def,
            arabic_meaning_from_db=arabic_meaning,
            english_def_arabic_from_db=english_def_arabic,
            example_from_db=example_sentence,
            example_arabic_from_db=example_sentence_arabic,
        )

        update_word_learning_details(
            word_id=word_id,
            arabic_meaning=arabic_meaning,
            english_def=english_def,
            english_def_arabic=english_def_arabic,
            example_sentence=example_sentence,
            example_sentence_arabic=example_sentence_arabic,
        )

        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic Meaning: {arabic_meaning}\n\n"
            f"English Definition:\n{english_def}\n\n"
            f"Arabic Definition:\n{english_def_arabic}\n\n"
            f"English Example:\n{example_sentence}\n\n"
            f"Arabic Example:\n{example_sentence_arabic}"
        )

        await message.answer(msg_text)
        await generate_and_send_audio(user_id, word_text)

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
        await get_translator()

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")

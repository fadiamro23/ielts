# bot_local_translate.py
# IELTS Vocabulary Telegram Bot with local English -> Arabic translation.
#
# Features:
# - Uses local NLLB or MADLAD translation instead of GoogleTranslator.
# - /study shows:
#     Word
#     Arabic Meaning
#     English Definition
#     Arabic Definition
#     IELTS English Example
#     Arabic Example
# - Uses IELTS-friendly examples instead of random/unsuitable dictionary examples when available.
# - Translates word, definition, and example with the local model.
# - Avoids saving "Translation unavailable" into the database.
# - Adds missing DB columns safely without deleting old data.
#
# Recommended run on a FREE GPU:
#   export TRANSLATION_BACKEND=nllb
#   export TRANSLATION_MODEL=facebook/nllb-200-distilled-600M
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   CUDA_VISIBLE_DEVICES=1 python bot_local_translate.py
#
# Stronger model, if the GPU is free:
#   export TRANSLATION_MODEL=facebook/nllb-200-1.3B
#   CUDA_VISIBLE_DEVICES=1 python bot_local_translate.py

import os

# Disable TF/Flax loading through transformers.
# Must be set before importing transformers.
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
# Config
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
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "4"))

# If set to 1, model loads only when first translation is requested.
LAZY_LOAD_TRANSLATOR = os.getenv("LAZY_LOAD_TRANSLATOR", "0").strip() == "1"

# If set to 1, /study will prefer the dictionary example if it exists.
# Default 0 means IELTS-friendly examples are preferred.
USE_DICTIONARY_EXAMPLES = os.getenv("USE_DICTIONARY_EXAMPLES", "0").strip() == "1"

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
        logging.info("Adding missing DB column: %s.%s", table_name, column_name)
        cursor.execute(ddl)


def init_db():
    """
    Safely creates/upgrades the database schema without deleting existing data.
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

    add_column_if_missing(cursor, "words", "english_def", "ALTER TABLE words ADD COLUMN english_def TEXT")
    add_column_if_missing(cursor, "words", "english_def_arabic", "ALTER TABLE words ADD COLUMN english_def_arabic TEXT")
    add_column_if_missing(cursor, "words", "arabic_meaning", "ALTER TABLE words ADD COLUMN arabic_meaning TEXT")
    add_column_if_missing(cursor, "words", "example_sentence", "ALTER TABLE words ADD COLUMN example_sentence TEXT")
    add_column_if_missing(cursor, "words", "example_sentence_arabic", "ALTER TABLE words ADD COLUMN example_sentence_arabic TEXT")
    add_column_if_missing(cursor, "words", "status", "ALTER TABLE words ADD COLUMN status TEXT DEFAULT 'new'")

    conn.commit()
    conn.close()


def clean_failed_translations():
    """
    Removes old failed placeholders so the bot can retry translation.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE words
        SET example_sentence_arabic = NULL
        WHERE example_sentence_arabic = 'Translation unavailable'
        """
    )

    cursor.execute(
        """
        UPDATE words
        SET english_def_arabic = NULL
        WHERE english_def_arabic = 'Translation unavailable'
        """
    )

    cursor.execute(
        """
        UPDATE words
        SET arabic_meaning = NULL
        WHERE arabic_meaning = 'Translation unavailable'
        """
    )

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
    rows = cursor.fetchall()
    conn.close()
    return rows


def update_word_learning_details(
    word_id: int,
    arabic_meaning: str,
    english_def: str,
    english_def_arabic: str,
    example_sentence: str,
    example_sentence_arabic: str,
):
    """
    Saves successful generated/translated content.
    Does not save Translation unavailable placeholders.
    """
    def safe_db_value(value: Optional[str]) -> Optional[str]:
        value = clean_text(value)
        if not value:
            return None
        if value == "Translation unavailable":
            return None
        return value

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE words
        SET
            arabic_meaning = COALESCE(?, arabic_meaning),
            english_def = COALESCE(?, english_def),
            english_def_arabic = COALESCE(?, english_def_arabic),
            example_sentence = COALESCE(?, example_sentence),
            example_sentence_arabic = COALESCE(?, example_sentence_arabic)
        WHERE id = ?
        """,
        (
            safe_db_value(arabic_meaning),
            safe_db_value(english_def),
            safe_db_value(english_def_arabic),
            safe_db_value(example_sentence),
            safe_db_value(example_sentence_arabic),
            word_id,
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Text helpers
# -----------------------------

def clean_text(text: Optional[str]) -> str:
    return " ".join((text or "").strip().split())


def is_bad_translation(text: Optional[str]) -> bool:
    text = clean_text(text)
    if not text:
        return True
    if text == "Translation unavailable":
        return True
    if len(text) > 400:
        return True

    # Detect obvious repetition like: "تحمّل، تحمّل، تحمّل..."
    tokens = [t.strip("،,.;:!?()[]{}") for t in text.split() if t.strip()]
    if len(tokens) >= 8:
        unique_ratio = len(set(tokens)) / max(len(tokens), 1)
        if unique_ratio < 0.35:
            return True

    return False


def fix_ielts_terms(text: str) -> str:
    text = clean_text(text)

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

    return clean_text(text)


# -----------------------------
# IELTS examples and definition cleanup
# -----------------------------

IELTS_EXAMPLES = {
    "abandon": "Many students abandon their study plans when they do not see quick progress.",
    "abide": "Students must abide by the rules during the IELTS exam.",
    "ability": "The IELTS speaking test measures a candidate's ability to communicate clearly.",
    "able": "A good writer is able to explain ideas clearly and support them with examples.",
    "abroad": "Many young people choose to study abroad to improve their career opportunities.",
    "absence": "The absence of clear examples can weaken an IELTS essay.",
    "absolute": "There is no absolute answer to some social and educational issues.",
    "absorb": "Reading every day helps learners absorb new vocabulary naturally.",
    "abstract": "Some IELTS writing topics involve abstract ideas such as freedom or success.",
    "abundant": "Modern technology gives students abundant access to learning materials.",
    "academic": "Academic vocabulary is important for achieving a high IELTS writing score.",
    "accept": "Some people accept change easily, while others prefer familiar routines.",
    "access": "Access to quality education can improve opportunities for young people.",
    "accommodation": "Many international students need affordable accommodation near their university.",
    "accompany": "Strong examples should accompany every main idea in an IELTS essay.",
    "accomplish": "Students can accomplish their goals if they practise consistently.",
    "account": "Governments should take environmental costs into account when planning new projects.",
    "accurate": "Accurate grammar helps candidates express their ideas more clearly.",
    "achieve": "Many candidates work hard to achieve a higher band score.",
    "acknowledge": "It is important to acknowledge both sides of an argument in academic writing.",
    "acquire": "Learners can acquire new vocabulary through reading and regular practice.",
    "adapt": "Students need to adapt quickly when they study in a foreign country.",
    "adequate": "An adequate answer includes clear ideas, examples, and logical organization.",
    "advantage": "One advantage of online learning is that students can study from home.",
    "affect": "Technology can affect the way people communicate and learn.",
    "afford": "Some families cannot afford private education.",
    "alternative": "Public transport can be an effective alternative to private cars.",
    "analyze": "Candidates should analyze the question carefully before writing their essay.",
    "approach": "A balanced approach is often useful in IELTS writing task two.",
    "appropriate": "Formal language is more appropriate for academic essays.",
    "approximately": "The number of visitors increased by approximately 20 percent.",
    "argue": "Some people argue that children should start learning languages at an early age.",
    "assume": "It is wrong to assume that all students learn in the same way.",
    "benefit": "Regular exercise can benefit both physical and mental health.",
    "challenge": "Living abroad can be a challenge for students at first.",
    "community": "A strong community can support young people and reduce social problems.",
    "consequence": "One consequence of pollution is the decline of public health.",
    "consider": "Governments should consider the needs of both drivers and pedestrians.",
    "consistent": "Consistent practice is essential for IELTS improvement.",
    "consume": "People today consume more digital content than ever before.",
    "contribute": "Education can contribute to economic growth and social stability.",
    "decline": "The use of printed newspapers has declined in many countries.",
    "demonstrate": "A high-scoring essay should demonstrate clear reasoning.",
    "develop": "Students can develop fluency by speaking English every day.",
    "efficient": "Public transport should be efficient, affordable, and reliable.",
    "environment": "Protecting the environment should be a priority for every country.",
    "essential": "Clear organization is essential in IELTS writing.",
    "evidence": "Writers should support their opinions with evidence and examples.",
    "factor": "Cost is an important factor when students choose a university.",
    "flexible": "Online courses are flexible because students can study at any time.",
    "impact": "Social media has a major impact on communication.",
    "improve": "Daily practice can improve a candidate's speaking fluency.",
    "individual": "Every individual has a responsibility to protect the environment.",
    "influence": "Parents can strongly influence their children's study habits.",
    "maintain": "People should maintain a healthy balance between work and leisure.",
    "method": "Different students prefer different methods of learning vocabulary.",
    "occur": "Traffic problems often occur in large cities during rush hour.",
    "policy": "A clear government policy can help reduce air pollution.",
    "priority": "For many students, improving English is a top priority.",
    "process": "Learning a language is a gradual process.",
    "require": "Some jobs require excellent communication skills.",
    "resource": "The internet is a valuable resource for IELTS preparation.",
    "significant": "There has been a significant increase in online education.",
    "solution": "Investing in public transport is one possible solution to traffic congestion.",
    "strategy": "A clear strategy can help candidates manage their time in the IELTS test.",
    "sufficient": "Students need sufficient practice before taking the IELTS exam.",
    "traditional": "Traditional classrooms still play an important role in education.",
}


ARABIC_DEFINITION_OVERRIDES = {
    "abandon": "يفقد السيطرة أو الانضباط، أو يترك شيئًا أو شخصًا دون رعاية أو استمرار.",
    "abide": "يتحمّل شيئًا صعبًا أو يلتزم بقاعدة أو قرار.",
}


ARABIC_MEANING_OVERRIDES = {
    "abandon": "يتخلى عن / يترك",
    "abide": "يلتزم / يتحمّل",
}


def make_ielts_example(word: str) -> str:
    word_lower = clean_text(word).lower()
    if word_lower in IELTS_EXAMPLES:
        return IELTS_EXAMPLES[word_lower]

    return f"Students can improve their IELTS score by learning how to use the word '{word_lower}' correctly."


def improve_definition_for_translation(word: str, definition: str) -> str:
    """
    Some DB definitions are synonym lists like:
      endure; put up with; bear; tolerate
    NLLB may repeat them badly.
    This converts some short synonym lists into a clearer English sentence.
    """
    word_lower = clean_text(word).lower()
    definition = clean_text(definition)

    if word_lower == "abide" and ("endure" in definition or "tolerate" in definition):
        return "to tolerate something difficult, or to accept and follow a rule or decision"

    if word_lower == "abandon" and "lacking restraint" in definition:
        return "a lack of control or restraint, especially because of strong emotion or enthusiasm"

    # If definition is just semicolon-separated synonyms, make it more sentence-like.
    if ";" in definition and len(definition.split()) <= 12:
        return f"to mean: {definition}"

    return definition


# -----------------------------
# Local translator
# -----------------------------

class LocalTranslator:
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

    def translate_en_to_ar(self, text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        text = clean_text(text)
        if not text:
            return ""

        if self.backend == "madlad":
            return self._translate_madlad_to_ar(text, max_new_tokens=max_new_tokens)

        return self._translate_nllb_to_ar(text, max_new_tokens=max_new_tokens)

    def _translate_nllb_to_ar(self, text: str, max_new_tokens: int) -> str:
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
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True,
            )

        translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return fix_ielts_terms(translated)

    def _translate_madlad_to_ar(self, text: str, max_new_tokens: int) -> str:
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
                max_new_tokens=max_new_tokens,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                repetition_penalty=1.25,
                early_stopping=True,
            )

        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fix_ielts_terms(translated)


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


async def local_translate_en_to_ar(text: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    translator = await get_translator()
    return await asyncio.to_thread(translator.translate_en_to_ar, text, max_new_tokens)


async def safe_translate_en_to_ar(
    text: str,
    fallback: str = "",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    try:
        translated = await local_translate_en_to_ar(text, max_new_tokens=max_new_tokens)
        translated = clean_text(translated)

        if is_bad_translation(translated):
            return fallback

        return translated

    except Exception as e:
        logging.exception("Translation failed for text=%r error=%s", text, e)
        return fallback


# -----------------------------
# Dictionary
# -----------------------------

def fetch_dictionary_definition(word: str) -> str:
    """
    Returns the first English definition from dictionaryapi.dev.
    We do not blindly trust examples from the API because they can be unsuitable for IELTS.
    """
    word = clean_text(word)
    if not word:
        return ""

    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        response = requests.get(url, timeout=8)

        if response.status_code != 200:
            return ""

        data = response.json()
        if not isinstance(data, list) or not data:
            return ""

        for meaning in data[0].get("meanings", []):
            for definition_obj in meaning.get("definitions", []):
                definition = clean_text(definition_obj.get("definition"))
                if definition:
                    return definition

    except Exception as e:
        logging.error("Dictionary API error for %s: %s", word, e)

    return ""


async def build_learning_card(
    word: str,
    english_def_from_db: Optional[str],
    arabic_meaning_from_db: Optional[str],
    english_def_arabic_from_db: Optional[str],
    example_from_db: Optional[str],
    example_arabic_from_db: Optional[str],
) -> Tuple[str, str, str, str, str]:
    """
    Returns:
      arabic_meaning,
      english_definition,
      arabic_definition,
      english_example,
      arabic_example
    """
    word = clean_text(word)
    word_lower = word.lower()

    # 1. English definition
    english_definition = clean_text(english_def_from_db)

    if not english_definition:
        english_definition = await asyncio.to_thread(fetch_dictionary_definition, word)

    if not english_definition:
        english_definition = f"A useful academic English word for IELTS preparation."

    english_definition_for_translation = improve_definition_for_translation(word, english_definition)

    # 2. IELTS-friendly English example
    # Prefer our IELTS examples unless explicitly told to use dictionary/DB examples.
    if USE_DICTIONARY_EXAMPLES:
        english_example = clean_text(example_from_db) or make_ielts_example(word)
    else:
        english_example = make_ielts_example(word)

    # 3. Arabic meaning
    arabic_meaning = clean_text(arabic_meaning_from_db)
    if word_lower in ARABIC_MEANING_OVERRIDES:
        arabic_meaning = ARABIC_MEANING_OVERRIDES[word_lower]

    if is_bad_translation(arabic_meaning):
        arabic_meaning = await safe_translate_en_to_ar(
            word,
            fallback="",
            max_new_tokens=32,
        )

    # 4. Arabic definition
    arabic_definition = clean_text(english_def_arabic_from_db)
    if word_lower in ARABIC_DEFINITION_OVERRIDES:
        arabic_definition = ARABIC_DEFINITION_OVERRIDES[word_lower]

    if is_bad_translation(arabic_definition):
        arabic_definition = await safe_translate_en_to_ar(
            english_definition_for_translation,
            fallback="",
            max_new_tokens=96,
        )

    # 5. Arabic example
    arabic_example = clean_text(example_arabic_from_db)
    if is_bad_translation(arabic_example):
        arabic_example = await safe_translate_en_to_ar(
            english_example,
            fallback="",
            max_new_tokens=96,
        )

    # Display fallback only, do not save this text into DB.
    display_arabic_meaning = arabic_meaning or "Translation unavailable"
    display_arabic_definition = arabic_definition or "Translation unavailable"
    display_arabic_example = arabic_example or "Translation unavailable"

    return (
        display_arabic_meaning,
        english_definition,
        display_arabic_definition,
        english_example,
        display_arabic_example,
    )


# -----------------------------
# Audio
# -----------------------------

async def generate_and_send_audio(chat_id: int, word: str):
    """
    English pronunciation using gTTS.
    Note: translation is local, but gTTS itself is an online TTS service.
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

    translated = await safe_translate_en_to_ar(
        text,
        fallback="Translation unavailable",
        max_new_tokens=128,
    )

    await message.answer(f"Arabic:\n{translated}")


@dp.message(Command("study"))
async def cmd_study(message: types.Message):
    user_id = message.from_user.id
    limit = get_user_settings(user_id)

    rows = get_daily_words(limit)

    if not rows:
        await message.answer("No new words available in the database.")
        return

    await message.answer(f"Fetching your {len(rows)} IELTS words for today...")

    for row in rows:
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
            final_arabic_meaning,
            final_english_def,
            final_arabic_def,
            final_english_example,
            final_arabic_example,
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
            arabic_meaning=final_arabic_meaning,
            english_def=final_english_def,
            english_def_arabic=final_arabic_def,
            example_sentence=final_english_example,
            example_sentence_arabic=final_arabic_example,
        )

        msg_text = (
            f"Word: {word_text}\n"
            f"Arabic Meaning: {final_arabic_meaning}\n\n"
            f"English Definition:\n{final_english_def}\n\n"
            f"Arabic Definition:\n{final_arabic_def}\n\n"
            f"IELTS English Example:\n{final_english_example}\n\n"
            f"Arabic Example:\n{final_arabic_example}"
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
    clean_failed_translations()

    if not LAZY_LOAD_TRANSLATOR:
        await get_translator()

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")

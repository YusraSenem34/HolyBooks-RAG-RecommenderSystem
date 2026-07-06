# config.py — Central configuration for Sacred Texts RAG
# Edit settings here; all other files import from this file.

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ── Embedding model (runs locally, no API key needed) ─────────────────────────
EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"

# ── LLM (Groq — Llama 3.3 70B) ───────────────────────────────────────────────
GROQ_MODEL      = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3   # lower = more factual, higher = more creative
LLM_MAX_TOKENS  = 2048

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_PER_BOOK  = 3      # maximum passages fetched per book per query
DRIFT_THRESHOLD = 0.35   # cosine similarity below this → fresh retrieval
MIN_RELEVANCE   = 0.25   # minimum relevance score to include a passage

# Context expansion — fetch the previous and next verse alongside each match
# to give the LLM a complete thought rather than an isolated verse.
# Disabled for Gita — shlokas are self-contained philosophical units.
CONTEXT_EXPANSION = {
    "bible": True,    # narrative — neighbors complete the thought
    "quran": True,    # helpful for narrative ayahs
    "torah": True,    # narrative/legal — neighbors add context
    "gita":  False,   # self-contained shlokas — not needed
}

# ── Books ─────────────────────────────────────────────────────────────────────
# Each entry defines one "pocket" in the multi-pocket retrieval system.
#
# Column conventions:
#   col_text     → the column embedded into ChromaDB (English text only)
#   col_metadata → list of columns stored as metadata (never embedded)
#   col_skip     → columns ignored entirely
#
BOOKS = {
    "bible": {
        "name":         "King James Bible",
        "tradition":    "Christianity",
        "followers":    "2.4 billion",
        "color":        "#4a7eb5",
        "filename":     "bible_kjv_dataset.csv",
        "collection":   "bible",
        "col_text":     "text",
        "col_metadata": ["book", "chapter", "verse", "citation"],
        "col_skip":     [],
        # citation column is already formatted as "Genesis 1:1"
        "ref_template": "{citation}",
    },
    "quran": {
        "name":         "The Quran",
        "tradition":    "Islam",
        "followers":    "1.9 billion",
        "color":        "#4a9e6b",
        "filename":     "the_quran_dataset.csv",
        "collection":   "quran",
        "col_text":     "ayah_en",
        "col_metadata": ["surah_no", "surah_name_roman", "ayah_no_surah", "ayah_ar"],
        "col_skip":     [],
        "ref_template": "{surah_name_roman} {surah_no}:{ayah_no_surah}",
    },
    "torah": {
        "name":         "Tanakh (JPS 1917)",
        "tradition":    "Judaism",
        "followers":    "~15 million",
        "color":        "#c9a84c",
        "filename":     "tanakh.csv",
        "collection":   "torah",
        "col_text":     "Text_English",
        "col_metadata": ["Book", "Chapter", "Verse", "Text_Hebrew"],
        "col_skip":     [],
        "ref_template": "{Book} {Chapter}:{Verse}",
    },
    "gita": {
        "name":         "Bhagavad Gita",
        "tradition":    "Hinduism",
        "followers":    "1.2 billion",
        "color":        "#c45a3a",
        "filename":     "Bhagwad_Gita.csv",
        "collection":   "gita",
        "col_text":     "EngMeaning",
        "col_metadata": ["ID", "Chapter", "Verse", "WordMeaning"],
        "col_skip":     ["Shloka", "HinMeaning", "Transliteration"],
        "ref_template": "Gita {Chapter}:{Verse}",
    },
}

# ── Suggested themes shown as chips in the UI ─────────────────────────────────
SUGGESTED_THEMES = [
    "forgiveness",
    "the meaning of suffering",
    "compassion for others",
    "death and the afterlife",
    "inner peace and stillness",
    "justice and mercy",
    "duty and purpose",
    "dealing with betrayal",
    "love and devotion",
    "the nature of the soul",
]
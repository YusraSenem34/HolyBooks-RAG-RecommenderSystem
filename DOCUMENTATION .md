# Sacred Texts RAG — Living Documentation

---

## 📌 What This Project Does

A conversational theme explorer that lets users ask questions and receive
comparative answers drawn from four major religious texts:

| Book             | Tradition    | Followers | Source                                  |
|------------------|--------------|-----------|-----------------------------------------|
| King James Bible | Christianity | 2.4B      | Kaggle — phyred23/bibleverses           |
| The Quran        | Islam        | 1.9B      | Kaggle — imrankhan197/the-quran-dataset |
| Torah (JPS 1917) | Judaism      | ~15M      | GitHub — MarkBuffalo/gen-tanakh         |
| Bhagavad Gita    | Hinduism     | 1.2B      | GitHub — Aabi0207/Bhagavad-Gita-GPT     |

---

## 🏗️ Architecture

### The Multi-Pocket Retrieval Pattern
The system avoids bias toward longer books (e.g. Bible has ~50x more words
than the Gita) by searching each book's ChromaDB collection independently,
guaranteeing a fixed number of results per book before merging.

```
[User Question]
      │
      ├──▶ [Search Bible Collection]    → top-3 passages
      ├──▶ [Search Quran Collection]    → top-3 passages
      ├──▶ [Search Torah Collection]    → top-3 passages
      └──▶ [Search Gita Collection]     → top-3 passages
                        │
              [Bundle: 12 passages]
                        │
              [Llama 3.3 70B via Groq]
                        │
              [Comparative Response]
```

### Sticky Context (Multi-turn Conversation)
On follow-up questions, the system reuses the passages retrieved in the
previous turn unless topic drift is detected (cosine similarity < 0.35),
in which case it runs a fresh retrieval.

```
Turn 1: retrieve passages → store in session
Turn 2: detect drift?
  No  → reuse stored passages + conversation history → LLM
  Yes → fresh multi-pocket retrieval → LLM
```

---

## 📁 Project Structure

```
sacred_texts_rag/
│
├── 📄 DOCUMENTATION.md       ← you are here (living doc)
├── 📄 requirements.txt       ← all Python dependencies
├── 📄 .env.example           ← copy to .env and fill in your API key
├── 📄 config.py              ← all settings in one place
│
├── 📂 setup/                 ← ⚠️ ONE-TIME scripts — safe to delete after running
│  
│   ├── 02_build_index.py     ← loads, filters, embeds, stores in ChromaDB
│
├── 📂 data/                  ← dataset files (place manual downloads here)
│   ├── bible_kjv_dataset.csv ← manual download (Kaggle)
│   ├── the_quran_dataset.csv ← manual download (Kaggle)
│   ├── tanakh.csv            ← converted from JSON (MarkBuffalo/gen-tanakh)
│   └── Bhagwad_Gita.csv      ← manual download (GitHub)
│
├── 📂 chroma_db/             ← ChromaDB persistent storage (auto-created)
│
├── 📄 retriever.py           ← multi-pocket retrieval logic
├── 📄 synthesizer.py         ← Groq / Llama 3.3 70B synthesis logic
└── 📄 app.py                 ← Streamlit UI
```

---

## 📦 Dataset Sources

| Book | URL | Format | Download | Notes |
|------|-----|--------|----------|-------|
| **Bible KJV** | https://www.kaggle.com/datasets/phyred23/bibleverses | CSV | ✅ Manual | Cols: `book_name`, `chapter`, `verse`, `text` |
| **Quran** | https://www.kaggle.com/datasets/imrankhan197/the-quran-dataset | CSV | ✅ Manual | Embed `ayah_en`. Store `ayah_ar` + `surah_name_roman` as metadata |
| **Torah JPS 1917** | https://github.com/MarkBuffalo/gen-tanakh | JSON → CSV | ✅ Manual | Downloaded as JSON, converted to CSV manually. Cols: `Book`, `Chapter`, `Verse`, `Text_English`, `Text_Hebrew`. Embed `Text_English` only |
| **Bhagavad Gita** | https://github.com/Aabi0207/Bhagavad-Gita-GPT | CSV | ✅ Manual | 2,414 rows — filter to English only in index script |

### Torah download + conversion scenario
The MarkBuffalo/gen-tanakh dataset does not offer a direct CSV download.
I downloaded the raw JSON file from:
  https://raw.githubusercontent.com/MarkBuffalo/gen-tanakh/main/tanakh.json
...and converted it to CSV manually using pandas:

```python
import pandas as pd, json

with open("tanakh.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Structure: data is a list of verse objects with keys:
# book, chapter, verse, text (English), hebrew (Hebrew script)
df = pd.DataFrame(data)
df.to_csv("data/tanakh.csv", index=False, encoding="utf-8")
```

Save the result as: data/tanakh.csv


## ✅ All Datasets Confirmed — Column Reference

| Book | File | Embed column | Metadata columns | Skip columns | Reference format |
|------|------|-------------|-----------------|-------------|-----------------|
| **Bible KJV** | `bible_kjv_dataset.csv` | `text` | `book`, `chapter`, `verse`, `citation` | — | uses `citation` column directly (pre-formatted as `Genesis 1:1`) |
| **Quran** | `the_quran_dataset.csv` | `ayah_en` | `surah_no`, `surah_name_roman`, `ayah_no_surah`, `ayah_ar` | — | `Al-Baqarah 2:255` |
| **Torah** | `tanakh.csv` | `Text_English` | `Book`, `Chapter`, `Verse`, `Text_Hebrew` | — | `Genesis 1:1` |
| **Bhagavad Gita** | `Bhagwad_Gita.csv` | `EngMeaning` | `ID`, `Chapter`, `Verse`, `WordMeaning` | `Shloka`, `HinMeaning`, `Transliteration` | `Gita 2:47` |

> ⚠️ `WordMeaning` in the Gita can be 200–400 words per verse (word breakdown + commentary).
> Stored in ChromaDB metadata but displayed in a collapsible section in the UI — not inline.

> ⚠️ `Text_Hebrew` and `ayah_ar` appear garbled in Excel (Windows cp1252 vs UTF-8 mismatch)
> but read correctly in Python with `encoding="utf-8"`. No data loss.

All 4 files are in `data/`. Ready to write `02_build_index.py`.

---

## ⚙️ Setup Instructions (Run Once)

### 1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
```bash
cp .env.example .env
# Open .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

### 4. Place dataset files in data/
Manually place these files before running the setup scripts:
```
data/bible_kjv_dataset.csv  ← from Kaggle (phyred23/bibleverses)
data/the_quran_dataset.csv  ← from Kaggle (imrankhan197/the-quran-dataset)
data/tanakh.csv             ← JSON downloaded from MarkBuffalo/gen-tanakh, converted to CSV
data/Bhagwad_Gita.csv       ← from GitHub (Aabi0207/Bhagavad-Gita-GPT)
```

### 5. Validate downloads  ⚠️ ONE-TIME
```bash
python setup/01_download_texts.py
# Validates all 4 CSV files are present and have correct columns
# Exits with clear error messages if anything is missing
# The file has been deleted after validation.
```

### 6. Build the vector index  ⚠️ ONE-TIME
```bash
python setup/02_build_index.py
# Loads, filters, and chunks all 4 books
# Creates embeddings and stores in ChromaDB
# Takes ~15 minutes on first run
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## 🔧 Configuration Reference (config.py)

| Setting            | Default                   | Description                              |
|--------------------|---------------------------|------------------------------------------|
| `CHUNK_SIZE`       | verse (no chunking)       | Each verse = one ChromaDB document       |
| `TOP_K_PER_BOOK`   | 3                         | Passages retrieved per book per query    |
| `DRIFT_THRESHOLD`  | 0.35                      | Cosine similarity below = new retrieval  |
| `EMBEDDING_MODEL`  | multi-qa-mpnet-base-dot-v1| Sentence-transformers model              |
| `GROQ_MODEL`       | llama-3.3-70b-versatile   | LLM for synthesis                        |
| `CHROMA_PATH`      | ./chroma_db               | Where ChromaDB stores its files          |
| `DATA_PATH`        | ./data                    | Where CSV/JSON files are stored          |

---

## 💬 Example Questions

Works well:
- "What do these books say about forgiveness?"
- "How should we deal with betrayal?"
- "What is the purpose of suffering?"
- "How do these traditions view justice versus mercy?"

Follow-up examples:
- "Why does the Bible emphasize forgiveness more than the others?"
- "Can you compare just the Quran and the Gita on this?"
- "Go deeper on the Hindu perspective."

Out of scope (handled gracefully):
- Questions about books not in the dataset
- Requests to declare one religion "correct"
- Very specific verse lookups by address (e.g. "John 3:16")

---

## 🧩 Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| One ChromaDB collection per book | ✅ | Guarantees equal retrieval (multi-pocket) |
| Verse-level indexing | ✅ | Natural boundaries, accurate references |
| Sentence-transformers locally | ✅ | No API cost, fast, good quality |
| Groq for LLM | ✅ | Free tier, fastest inference for Llama |
| Sticky context on follow-ups | ✅ | Maintains comparative thread |
| Drift detection via cosine sim | ✅ | Avoids stale context on topic change |
| JPS 1917 for Torah | ✅ | Official Jewish translation, public domain |
| Torah scope: 16 books of Tanakh | ✅ | Keeps ~95% of active Jewish practice |
| Arabic stored as metadata | ✅ | Display only — not embedded |
| UTF-8 enforced on all reads | ✅ | Prevents Arabic/Sanskrit corruption |

---

## 🚧 Known Limitations (v1.4)

- Translations may not capture nuance of original languages
- No speaker attribution within books (who is speaking in a verse)

---


## 📝 Changelog

| Version | Changes |
|---------|---------|
| v1.0 | Initial build — Bible, Quran, Torah, Gita · ChromaDB · Groq Llama 3.3 70B · Streamlit UI |
| v1.1 | Torah downloader: retry logic + chapter-level resume cache |
| v1.2 | Switched all sources to verse-level CSV/JSON datasets. Torah → MarkBuffalo/gen-tanakh. Gita → Aabi0207/Bhagavad-Gita-GPT |
| v1.3 | Torah JSON→CSV conversion scenario added. All 4 files confirmed manually downloaded by me |
| v1.4 | Embed `English Translation` columns for all books. All 4 datasets ready — 02_build_index.py next |
| v1.5 | Use the original texts as metadata. For Gita, add `WordMeaning`  column too. All column/metadata decisions finalised |
| v1.6 | Index builder explanation added to docs. Post-indexing next steps documented |
| v1.7 | Bible CSV columns corrected: `book_name` → `book`. `citation` column added to metadata (pre-formatted reference) |
| v1.8 | Context expansion implemented. `seq_index` added to metadata in `02_build_index.py`. `CONTEXT_EXPANSION` per-book config added. Retriever fetches prev/next verse for Bible, Quran, Torah. Gita excluded (self-contained shlokas). `MIN_RELEVANCE` and `multi-qa-mpnet-base-dot-v1` added to config |
| v1.9 | Fixed ChromaDB distance metric: `hnsw:space: "ip"` (inner product / dot product) added to `create_collection`. Required because `multi-qa-mpnet-base-dot-v1` produces unnormalized vectors (norm ≈ 5.8, not 1.0) — cosine and dot product give different results for this model. Reindex required |

---

## 🧠 How the Index Builder Works (02_build_index.py)

Understanding what this script does is important before running it.

### The problem it solves
Your 4 CSV files are just rows of text — unsearchable by meaning.
If a user asks "what does the Gita say about inner peace?", you cannot
grep for that. The index builder transforms CSVs into a semantic vector
database that understands meaning, not just keywords.

### Step by step

**Step 1 — Read the CSVs**
All 4 files are loaded into pandas DataFrames.

**Step 2 — Pick the right columns (from config.py)**
For each book, config.py defines exactly:
- `col_text`     → the English column to embed (e.g. `ayah_en` for Quran)
- `col_metadata` → columns stored alongside for display (e.g. `Text_Hebrew`)
- `col_skip`     → columns ignored entirely (e.g. `Shloka`, `HinMeaning`)

config.py is the single source of truth — change filenames or columns
there and the index script picks it up automatically.

**Step 3 — Convert each verse into a vector (embedding)**
The embedding model (`all-MiniLM-L6-v2` didn’t work well so `multi-qa-mpnet-base-dot-v1` is used later.) reads each verse and converts
it into 768 (multi-qa-mpnet-base-dot-v1) numbers representing its meaning:

```
"For God so loved the world..."  →  [0.23, -0.14, 0.87, 0.02, ...]
                                      ↑ 786 numbers representing meaning
```

Verses about similar topics produce similar vectors — even with
completely different words. This is what makes semantic search possible.

**Step 4 — Store in ChromaDB (one collection per book)**
Each verse is stored as one entry with three parts:

```
ID       → "bible_0"
Document → "For God so loved the world..."      ← text
Vector   → [0.23, -0.14, 0.87, ...]             ← embedding
Metadata → { book_name: "John",
              chapter: "3",
              verse: "16",
              reference: "John 3:16" }           ← reference info
```

One separate ChromaDB collection per book (the multi-pocket design):
```
chroma_db/
  ├── bible     ← ~31,000 verses
  ├── quran     ←  ~6,200 verses
  ├── torah     ← ~23,000 verses
  └── gita      ←    ~700 verses
```

**Step 5 — At query time (when the app runs)**
1. User types "what does forgiveness mean?"
2. Query is converted to a vector using the same model
3. Each collection is searched separately → top 3 verses per book
4. 12 verses total bundled and sent to Llama 3.3 70B
5. LLM generates a structured comparative response

### One-line summary
> 02_build_index.py reads your CSVs, turns every verse into a vector,
> and saves everything into ChromaDB so the app can search by meaning —
> not just by keywords. config.py tells it which columns to use per book.

---

## 🚀 What Comes After Indexing

Once `python setup/02_build_index.py` completes successfully:

### 1. Verify the output
The script prints a verification table at the end:
```
✅ King James Bible          ~31,000 verses
✅ The Quran                  ~6,200 verses
✅ Tanakh (JPS 1917)         ~23,000 verses
✅ Bhagavad Gita                ~700 verses
```
It also runs a test query on each collection and prints the top result.
If any collection shows 0 verses or throws an error, fix it before proceeding.

### 2. Clean up setup/ (optional but recommended)
Both setup scripts are now done and safe to delete:
```bash
rm -rf setup/
```
The data/ folder and chroma_db/ folder must be kept — deleting them
means re-running the index builder from scratch.

### 3. Launch the app
```bash
streamlit run app.py
```
The app will open in your browser at http://localhost:8501

### 4. Test with a few queries
Try these to confirm each book is retrieving correctly:
- "What is the meaning of suffering?"   ← philosophical, all books should respond
- "How should we treat our enemies?"    ← ethical, strong results expected
- "What happens after death?"           ← theological, very different per tradition
- Ask a follow-up to test sticky context is working

### 5. Files to keep forever
```
config.py       ← settings — edit here if anything changes
retriever.py    ← multi-pocket search logic
synthesizer.py  ← Groq / Llama synthesis
app.py          ← Streamlit UI
data/           ← original CSVs (keep as backup)
chroma_db/      ← the vector index (never delete this)
DOCUMENTATION.md← this file
requirements.txt
.env
```


"""
setup/02_build_index.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  ONE-TIME SCRIPT — Run once after placing all 4 files
    in data/. Safe to delete after successful run.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

What this script does:
  1. Reads each CSV/JSON with UTF-8 encoding
  2. Validates required columns are present
  3. Drops rows where the embed column is empty
  4. Stores each verse as one ChromaDB document with full metadata
  5. Creates one ChromaDB collection per book (the multi-pocket design)

Expected files in data/:
  bible_kjv_dataset.csv   →  bible   collection
  the_quran_dataset.csv   →  quran   collection
  tanakh.csv              →  torah   collection
  Bhagwad_Gita.csv        →  gita    collection

Run:
  python setup/02_build_index.py
"""

import os
import sys
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, CHROMA_PATH, BOOKS, EMBEDDING_MODEL


# ── Tiny logger ───────────────────────────────────────────────────────────────

def ok(msg):   print(f"  ✅ {msg}")
def info(msg): print(f"  ℹ️  {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


# ── ChromaDB + embedding setup ────────────────────────────────────────────────

def get_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_ef():
    info(f"Loading embedding model: {EMBEDDING_MODEL}")
    info("First run downloads ~90 MB — subsequent runs are instant.")
    return SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


# ── CSV / file loader ─────────────────────────────────────────────────────────

def load_file(book_key: str, book_info: dict) -> pd.DataFrame | None:
    """
    Load a dataset file into a DataFrame.
    Handles CSV with UTF-8 encoding and replaces bad characters safely.
    """
    filename = book_info["filename"]
    filepath = os.path.join(DATA_PATH, filename)

    if not os.path.exists(filepath):
        err(f"File not found: data/{filename}")
        err(f"Place the file in the data/ folder and re-run.")
        return None

    ext = os.path.splitext(filename)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(
                filepath,
                encoding="utf-8",
                encoding_errors="replace",  # replaces unreadable chars with ?
                dtype=str,                  # read all as string — avoids pandas
            )                               # misinterpreting verse numbers as floats
        elif ext == ".json":
            df = pd.read_json(filepath, encoding="utf-8", dtype=str)
        else:
            err(f"Unsupported file type: {ext}")
            return None

        # Strip whitespace from column names — common after Excel exports
        df.columns = df.columns.str.strip()
        info(f"Loaded {filename}  ({len(df):,} rows, {len(df.columns)} columns)")
        return df

    except Exception as e:
        err(f"Could not read {filename}: {e}")
        return None


# ── Column validation ─────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, book_key: str, book_info: dict) -> bool:
    """Check that all required columns (embed + metadata) exist in the DataFrame."""
    needed = [book_info["col_text"]] + book_info["col_metadata"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        err(f"Missing columns in {book_info['filename']}: {missing}")
        info(f"Found columns: {list(df.columns)}")
        return False

    ok(f"All required columns present for {book_info['name']}")
    return True


# ── Reference builder ─────────────────────────────────────────────────────────

def build_reference(row: pd.Series, template: str) -> str:
    """
    Build a human-readable reference string from a row.
    Example: "{Book} {Chapter}:{Verse}" → "Genesis 1:1"
    Falls back gracefully if a column is missing or NaN.
    """
    try:
        # Extract column names from template
        import re
        keys = re.findall(r"\{(\w+)\}", template)
        values = {k: str(row.get(k, "?")).strip() for k in keys}
        return template.format(**values)
    except Exception:
        return "Unknown reference"


# ── Core indexing function ────────────────────────────────────────────────────

def index_book(book_key: str, book_info: dict, client, ef) -> bool:
    """
    Load, validate, clean and index one book into ChromaDB.
    Returns True on success, False on failure.
    """
    print(f"\n{'─' * 55}")
    print(f"  📚 {book_info['name']}  ({book_info['tradition']})")
    print(f"{'─' * 55}")

    # 1. Load file
    df = load_file(book_key, book_info)
    if df is None:
        return False

    # 2. Validate columns
    if not validate_columns(df, book_key, book_info):
        return False

    # 3. Drop rows where embed column is empty or NaN
    col_text = book_info["col_text"]
    before = len(df)
    df = df[df[col_text].notna() & (df[col_text].str.strip() != "")]
    dropped = before - len(df)
    if dropped:
        warn(f"Dropped {dropped} empty rows from {col_text} column")
    info(f"Verses to index: {len(df):,}")

    # 4. Create or replace ChromaDB collection
    collection_name = book_info["collection"]
    try:
        client.delete_collection(collection_name)
        info(f"Deleted existing '{collection_name}' collection — rebuilding fresh")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={
            "book":        book_info["name"],
            "tradition":   book_info["tradition"],
            "hnsw:space":  "ip",   # ← dot product — required for multi-qa-mpnet-base-dot-v1
        }
    )

    # 5. Build documents, metadatas, ids and index in batches
    BATCH_SIZE = 250   # safe for ChromaDB + sentence-transformers memory

    ids        = []
    documents  = []
    metadatas  = []

    for i, (_, row) in enumerate(df.iterrows()):
        # Document — the text that gets embedded
        doc_text = str(row[col_text]).strip()

        # Metadata — stored alongside, never embedded
        meta = {}
        for col in book_info["col_metadata"]:
            val = row.get(col, "")
            # Store as string; None/NaN → empty string
            meta[col] = "" if pd.isna(val) else str(val).strip()

        # Add computed reference string
        meta["reference"]  = build_reference(row, book_info["ref_template"])
        meta["book_key"]   = book_key
        meta["book_name"]  = book_info["name"]
        meta["tradition"]  = book_info["tradition"]
        meta["seq_index"]  = str(i)    # ← position index for context expansion

        ids.append(f"{book_key}_{i}")
        documents.append(doc_text)
        metadatas.append(meta)

        # Flush batch
        if len(ids) >= BATCH_SIZE:
            collection.add(documents=documents, ids=ids, metadatas=metadatas)
            print(f"  Indexed {i + 1:>6,} / {len(df):,} verses...", end="\r")
            ids, documents, metadatas = [], [], []

    # Flush remaining
    if ids:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    total = collection.count()
    ok(f"Indexed {total:,} verses into collection '{collection_name}'")
    return True


# ── Post-index verification ───────────────────────────────────────────────────

def verify_index(client) -> None:
    """
    Quick sanity check: query each collection with a simple test phrase
    and print the top result to confirm retrieval works.
    """
    print(f"\n{'═' * 55}")
    print("  🔍 Verification — test query per collection")
    print(f"{'═' * 55}")

    test_query = "love and compassion"
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    for book_key, book_info in BOOKS.items():
        try:
            col = client.get_collection(
                name=book_info["collection"],
                embedding_function=ef
            )
            results = col.query(query_texts=[test_query], n_results=1)

            doc  = results["documents"][0][0]
            meta = results["metadatas"][0][0]
            ref  = meta.get("reference", "?")

            print(f"\n  {book_info['name']}")
            print(f"  Ref  : {ref}")
            print(f"  Text : {doc[:120]}{'...' if len(doc) > 120 else ''}")

        except Exception as e:
            warn(f"Could not verify {book_info['name']}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Sacred Texts RAG — Index Builder (ONE-TIME)")
    print("=" * 55)
    print(f"  Data path  : {DATA_PATH}")
    print(f"  ChromaDB   : {CHROMA_PATH}")
    print(f"  Embed model: {EMBEDDING_MODEL}")

    client = get_client()
    ef     = get_ef()

    results = {}
    for book_key, book_info in BOOKS.items():
        success = index_book(book_key, book_info, client, ef)
        results[book_key] = success

    # Summary
    print(f"\n{'═' * 55}")
    print("  📊 Indexing Summary")
    print(f"{'═' * 55}")

    all_ok = True
    for book_key, book_info in BOOKS.items():
        success = results[book_key]
        status  = "✅" if success else "❌"
        try:
            count = client.get_collection(book_info["collection"]).count()
            print(f"  {status} {book_info['name']:<30} {count:>7,} verses")
        except Exception:
            print(f"  {status} {book_info['name']:<30}   not indexed")
        if not success:
            all_ok = False

    if all_ok:
        # Run verification queries
        verify_index(client)

        print(f"\n{'═' * 55}")
        ok("All books indexed successfully.")
        print("  👉 Next step: streamlit run app.py")
        print("  🗑️  You can delete the setup/ folder now.")
        print(f"{'═' * 55}")
    else:
        print(f"\n{'═' * 55}")
        err("Some books failed to index. Fix errors above and re-run.")
        print(f"{'═' * 55}")
        sys.exit(1)
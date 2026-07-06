"""
retriever.py — Multi-Pocket Retrieval Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Searches each book's ChromaDB collection independently (one "pocket" per book),
then bundles the top-K results from each into a single context packet for the LLM.

Pipeline per query:
  1. expand_query()  — LLM rewrites query into richer ancient-text vocabulary
  2. ChromaDB query  — fetch top_k * 2 candidates per book with embeddings
  3. mmr_rerank()    — pick top_k diverse passages using Maximal Marginal Relevance
  4. format_context()— bundle all passages for the LLM prompt
"""

import os
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from groq import Groq

from config import (
    CHROMA_PATH, BOOKS,
    TOP_K_PER_BOOK, EMBEDDING_MODEL, DRIFT_THRESHOLD, GROQ_MODEL,
    MIN_RELEVANCE, CONTEXT_EXPANSION
)

load_dotenv()

import csv
import json
from datetime import datetime
from config import BASE_DIR

# ── Singleton clients ─────────────────────────────────────────────────────────

_client = None
_ef     = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_ef():
    global _ef
    if _ef is None:
        _ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return _ef


def _get_collection(book_key: str):
    return _get_client().get_collection(
        name=BOOKS[book_key]["collection"],
        embedding_function=_get_ef()
    )


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """
    Use the LLM to rewrite the user's modern-English query into richer
    semantic search terms that match the vocabulary of ancient religious texts.

    Example:
      Input : "how to deal with anxiety"
      Output: "anxiety fear troubled heart worry restlessness peace
               trust surrender faith stillness inner calm divine refuge"

    Falls back to the original query if the API call fails.
    Note: uses the original query (not the expanded one) for drift detection
    so that topic comparison stays grounded in the user's own words.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return query

    prompt = (
        "Rewrite the following question as a rich semantic search phrase "
        "for finding relevant passages in ancient religious texts "
        "(Bible, Quran, Torah, Bhagavad Gita).\n"
        "Include synonyms, related theological concepts, and archaic equivalents.\n"
        "Return ONLY the expanded phrase — no explanation, no quotes, under 40 words.\n\n"
        f"Question: {query}"
    )

    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.3,
        )
        expanded = response.choices[0].message.content.strip()
        print(f"  🔍 Query expanded: {expanded[:80]}...")
        return expanded

    except Exception as e:
        print(f"  ⚠️  Query expansion failed ({e}). Using original query.")
        return query


# ── MMR reranking ─────────────────────────────────────────────────────────────

def mmr_rerank(
    passages: list,
    top_k: int,
    lambda_param: float = 0.7
) -> list:
    """
    Maximal Marginal Relevance — pick passages that are relevant to the
    query but diverse from each other.

    Uses the real passage embeddings stored in each passage dict,
    so similarity is computed in the actual vector space.

    Args:
        passages     : list of passage dicts with "embedding" and "relevance"
        top_k        : how many to return
        lambda_param : 1.0 = pure relevance, 0.0 = pure diversity (0.7 is balanced)

    Returns:
        top_k passages selected for both relevance and diversity
    """
    if len(passages) <= top_k:
        return passages

    selected  = []
    remaining = passages.copy()

    while remaining and len(selected) < top_k:
        mmr_scores = []

        for p in remaining:
            relevance = p["relevance"]

            if not selected:
                # Nothing selected yet — just use relevance
                max_sim = 0.0
            else:
                # Compute cosine similarity to each already-selected passage
                p_emb = np.array(p["embedding"])
                sims  = []
                for s in selected:
                    s_emb = np.array(s["embedding"])
                    denom = np.linalg.norm(p_emb) * np.linalg.norm(s_emb)
                    sim   = float(np.dot(p_emb, s_emb) / denom) if denom > 0 else 0.0
                    sims.append(sim)
                max_sim = max(sims)

            # MMR score: high relevance + low similarity to already selected
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr_score)

        best_idx = mmr_scores.index(max(mmr_scores))
        selected.append(remaining.pop(best_idx))

    return selected


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K_PER_BOOK) -> dict:
    """
    Full retrieval pipeline:
      1. Expand query with LLM for richer ancient-text vocabulary
      2. Fetch top_k * 2 candidates per book (with embeddings for MMR)
      3. Apply MMR to select top_k diverse + relevant passages
      4. Return results dict keyed by book_key

    Each passage dict includes:
        text        — English verse text
        reference   — citation e.g. "Al-Baqarah 2:255"
        relevance   — float 0–1, higher = more relevant
        distance    — raw ChromaDB cosine distance
        book_key    — e.g. "bible"
        book_name   — e.g. "King James Bible"
        tradition   — e.g. "Christianity"
        embedding   — passage vector (used for MMR, not shown in UI)
        meta        — full metadata dict (ayah_ar, Text_Hebrew, WordMeaning etc.)
    """
    results = {}

    # Step 1 — expand query (original kept for drift detection)
    expanded_query = expand_query(query)

    for book_key in BOOKS:
        try:
            collection = _get_collection(book_key)

            # Step 2 — fetch more candidates than needed so MMR can choose
            raw = collection.query(
                query_texts=[expanded_query],
                n_results=min(top_k * 2, collection.count()),
                include=["documents", "metadatas", "distances", "embeddings"]
            )

            candidates = []
            for doc, meta, dist, emb in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                raw["embeddings"][0]
            ):
                # ─── 🛠️ THE INNER PRODUCT (IP) SCALING ENGINE ───
                # ChromaDB's 'ip' distance returns negative dot product: dist = -(a · b)
                # We flip it to get the raw positive similarity score:
                raw_sim = -dist
                
                # Squash the unnormalized unscaled score into a clean 0.0 to 1.0 range.
                # A scaling factor of /15 maps standard MPNet scores beautifully:
                # - A raw_sim of ~0 (junk) becomes 0.50 (lowered to 0 by baseline shift below)
                # - A raw_sim of 15 (good match) scales to ~0.73
                # - A raw_sim of 30+ (elite match) scales to 0.90+
                scaled_score = float(1 / (1 + np.exp(-raw_sim / 15)))
                
                # Min-max baseline adjust: Shift it so zero-matching noise drops to absolute 0.0
                display_relevance = max(0.0, (scaled_score - 0.5) * 2)

                candidates.append({
                    "text":      doc,
                    "reference": meta.get("reference", ""),
                    "relevance": round(display_relevance, 4),
                    "distance":  round(dist, 4),
                    "book_key":  book_key,
                    "book_name": BOOKS[book_key]["name"],
                    "tradition": BOOKS[book_key]["tradition"],
                    "embedding": emb,     # kept for MMR only
                    "meta":      meta,
                })

            # Step 3 — MMR rerank to pick diverse top_k passages
            passages = mmr_rerank(candidates, top_k=top_k, lambda_param=0.7)

            # Step 4 — filter by relevance threshold
            # Keep only passages that score above MIN_RELEVANCE
            # Fallback: always keep the single best match so no tradition
            # is silently dropped from the comparison
            filtered = [p for p in passages if p["relevance"] >= MIN_RELEVANCE]
            if not filtered and passages:
                filtered = [passages[0]]  # keep top-1 even if below threshold
            passages = filtered

            # Step 5 — context expansion
            # For books where enabled, fetch the previous and next verse
            # so the LLM gets a complete thought rather than an isolated verse.
            # Uses seq_index stored in metadata during indexing.
            if CONTEXT_EXPANSION.get(book_key, False):
                for p in passages:
                    try:
                        seq_idx = int(p["meta"].get("seq_index", -1))
                        if seq_idx < 0:
                            continue

                        # Fetch previous verse
                        prev_result = collection.get(
                            where={"seq_index": str(seq_idx - 1)},
                            include=["documents"]
                        )
                        # Fetch next verse
                        next_result = collection.get(
                            where={"seq_index": str(seq_idx + 1)},
                            include=["documents"]
                        )

                        prev_text = prev_result["documents"][0].strip() \
                            if prev_result["documents"] else ""
                        next_text = next_result["documents"][0].strip() \
                            if next_result["documents"] else ""

                        # Wrap matched verse with neighbors
                        # Format: [prev] ¶ [MATCHED VERSE] ¶ [next]
                        parts = filter(None, [prev_text, p["text"].strip(), next_text])
                        p["text"] = " ¶ ".join(parts)

                    except Exception as e:
                        # Fallback gracefully — keep original verse text
                        print(f"  ⚠️  Context expansion failed for {book_key}: {e}")

            # Drop embedding from final output — not needed downstream
            for p in passages:
                p.pop("embedding", None)

            results[book_key] = passages

        except Exception as e:
            print(f"  ❌ Error retrieving {book_key}: {e}")
            results[book_key] = [{
                "text":      f"[Error retrieving from {BOOKS[book_key]['name']}: {e}]",
                "reference": "",
                "relevance": 0.0,
                "distance":  1.0,
                "book_key":  book_key,
                "book_name": BOOKS[book_key]["name"],
                "tradition": BOOKS[book_key]["tradition"],
                "meta":      {},
            }]

    return results


# ── Drift detection ───────────────────────────────────────────────────────────

def is_topic_drift(query_a: str, query_b: str) -> bool:
    """
    Returns True if the user has significantly changed topic.
    Compares the original (unexpanded) queries so drift detection
    stays grounded in the user's actual words.
    Below DRIFT_THRESHOLD → new topic → fresh retrieval triggered.
    """
    ef   = _get_ef()
    embs = ef([query_a, query_b])
    a, b = np.array(embs[0]), np.array(embs[1])
    cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return cosine_sim < DRIFT_THRESHOLD


# ── Context formatting for LLM ────────────────────────────────────────────────

def format_context(retrieved: dict) -> str:
    """
    Format retrieved passages into a structured string for the LLM prompt.
    Includes reference labels so the LLM can cite sources accurately.
    """
    lines = []
    for book_key, passages in retrieved.items():
        book_name = BOOKS[book_key]["name"]
        tradition = BOOKS[book_key]["tradition"]
        lines.append(f"\n### {book_name} ({tradition})")

        for i, p in enumerate(passages, start=1):
            ref = f" [{p['reference']}]" if p["reference"] else ""
            lines.append(f"[Passage {i}]{ref} {p['text']}")

    return "\n".join(lines)


# ── Index health check ────────────────────────────────────────────────────────

def check_index() -> dict:
    """Returns index status for each book. Used in app.py sidebar."""
    status   = {}
    client   = _get_client()
    existing = {col.name for col in client.list_collections()}

    for book_key, book_info in BOOKS.items():
        col_name = book_info["collection"]
        if col_name in existing:
            count = client.get_collection(col_name).count()
            status[book_key] = {"indexed": True, "chunks": count}
        else:
            status[book_key] = {"indexed": False, "chunks": 0}

    return status


# ── Performance scoring ───────────────────────────────────────────────────────

def calculate_system_score(retrieved_results: dict) -> float:
    """
    Calculates a global retrieval quality score using the Harmonic Mean
    of the top passage relevance score per book.
    Harmonic mean penalises heavily if any one book retrieves poorly.
    """
    scores = []
    for book_key, passages in retrieved_results.items():
        if passages:
            scores.append(passages[0]["relevance"])

    if not scores or any(s <= 0 for s in scores):
        return 0.0

    harmonic_mean = len(scores) / sum(1.0 / s for s in scores)
    return round(harmonic_mean, 4)

# This creates a 'retrieval_perf.csv' file right inside your main directory
LOG_FILE_PATH = os.path.join(BASE_DIR, "retrieval_perf.csv")

def log_retrieval_attempt(prompt: str, system_score: float, retrieved_results: dict):
    """
    Appends a new performance audit row to retrieval_perf.csv.
    Extracts the highest-ranking verse reference from each pocket for the log table.
    """
    file_exists = os.path.exists(LOG_FILE_PATH)
    
    # Isolate the top winning verse from each book pocket for tracking
    extracted_refs = {}
    for book_key, passages in retrieved_results.items():
        if passages and len(passages) > 0:
            extracted_refs[book_key] = passages[0].get("reference", "Unknown")
        else:
            extracted_refs[book_key] = "None"
            
    row_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "global_score": system_score,
        "pulled_verses": json.dumps(extracted_refs) # Saves as a clean JSON lookup string
    }
    
    fieldnames = ["timestamp", "prompt", "global_score", "pulled_verses"]
    
    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

# ── Offline test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Retriever Engine Offline...")
    test_phrase = "What is the right way?"

    raw_output       = retrieve(test_phrase, top_k=1)
    formatted_output = format_context(raw_output)
    system_perf      = calculate_system_score(raw_output)

    print(formatted_output)
    print(f"\n📈 GLOBAL RETRIEVAL PERFORMANCE SCORE: {system_perf}")
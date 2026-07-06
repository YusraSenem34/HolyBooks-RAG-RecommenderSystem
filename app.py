"""
app.py — Sacred Texts RAG · Streamlit UI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run with:
    streamlit run app.py
"""

import streamlit as st
from retriever import retrieve, is_topic_drift, check_index, format_context, calculate_system_score, log_retrieval_attempt, LOG_FILE_PATH
from synthesizer import synthesize
from config import BOOKS, SUGGESTED_THEMES

import pandas as pd
import os
import json

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sacred Texts · Theme Explorer",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }

    .main-title {
        font-size: 2.4rem;
        font-weight: 300;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 4px;
    }
    .passage-card {
        border-left: 3px solid;
        padding: 0.6rem 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 6px 6px 0;
        background: rgba(255,255,255,0.02);
    }
    .original-text {
        font-size: 0.85rem;
        color: #aaa;
        direction: rtl;   /* right-to-left for Arabic and Hebrew */
        text-align: right;
        margin-top: 4px;
        font-family: serif;
    }
    .reference-label {
        font-size: 0.75rem;
        font-weight: 600;
        opacity: 0.7;
        margin-bottom: 2px;
    }
    div[data-testid="column"] button {
        font-size: 0.8rem;
        padding: 0.2rem 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "retrieved_passages" not in st.session_state:
    st.session_state.retrieved_passages = None

if "last_query" not in st.session_state:
    st.session_state.last_query = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ✦ Sacred Texts RAG")
    st.caption("Comparative theme explorer · v1.0")

    st.divider()

    st.markdown("### 📚 Books in the Index")
    index_status = check_index()

    for book_key, info in BOOKS.items():
        status = index_status.get(book_key, {})
        if status.get("indexed"):
            verses = status["chunks"]
            st.markdown(
                f"<span style='color:{info['color']}'>●</span> "
                f"**{info['name']}** — {verses:,} verses",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<span style='color:#888'>○</span> "
                f"**{info['name']}** — *not indexed yet*",
                unsafe_allow_html=True
            )

    all_indexed = all(s.get("indexed") for s in index_status.values())
    if not all_indexed:
        st.warning(
            "⚠️ Some books are not indexed yet.\n\n"
            "Run:\n"
            "```\npython setup/02_build_index.py\n```"
        )

    st.divider()

    st.markdown("### ⚙️ Settings")
    top_k = st.slider(
        "Passages per book",
        min_value=1, max_value=5, value=3,
        help="How many verses to retrieve from each book per query"
    )

    show_original = st.toggle(
        "Show original language",
        value=True,
        help="Show Arabic (Quran) and Hebrew (Torah) alongside English"
    )

    show_word_meaning = st.toggle(
        "Show Gita word meanings",
        value=False,
        help="Show word-by-word Sanskrit commentary for Gita verses"
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.retrieved_passages   = None
        st.session_state.last_query           = None
        st.session_state.chat_messages        = []
        st.rerun()

    st.divider()
    st.caption(
        "**Sources:**\n"
        "Bible — Kaggle (phyred23/bibleverses)\n\n"
        "Quran — Kaggle (imrankhan197)\n\n"
        "Torah — MarkBuffalo/gen-tanakh\n\n"
        "Gita — Aabi0207/Bhagavad-Gita-GPT\n\n"
        "**LLM:** Llama 3.3 70B via Groq"
    )


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 class='main-title'>✦ Sacred Texts Theme Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='main-subtitle'>"
    "Ask any question and see how the Bible, Quran, Torah, and Bhagavad Gita respond — side by side."
    "</p>",
    unsafe_allow_html=True
)

# Tradition badges
cols = st.columns(len(BOOKS))
for col, (key, info) in zip(cols, BOOKS.items()):
    col.markdown(
        f"<span class='badge' style='background:{info['color']}22; "
        f"color:{info['color']}; border:1px solid {info['color']}55'>"
        f"{info['tradition']}</span>",
        unsafe_allow_html=True
    )

st.divider()

# ── Suggested themes ──────────────────────────────────────────────────────────

st.markdown("**✨ Try a theme:**")
selected_chip = None

chip_cols = st.columns(len(SUGGESTED_THEMES[:5]))
for col, theme in zip(chip_cols, SUGGESTED_THEMES[:5]):
    if col.button(theme, key=f"chip_{theme}"):
        selected_chip = theme

chip_cols2 = st.columns(len(SUGGESTED_THEMES[5:]))
for col, theme in zip(chip_cols2, SUGGESTED_THEMES[5:]):
    if col.button(theme, key=f"chip2_{theme}"):
        selected_chip = theme

st.divider()


# ── Passage card renderer ─────────────────────────────────────────────────────

def render_passage(p: dict, book_info: dict, index: int) -> None:
    """Render one retrieved passage with reference, text, and optional extras."""
    color = book_info["color"]
    meta  = p.get("meta", {})

    ref = p.get("reference", "")

    card_html = (
        f"<div class='passage-card' style='border-color:{color}'>"
        f"<div class='reference-label' style='color:{color}'>{ref}</div>"
        f"<em>{p['text'][:350]}{'...' if len(p['text']) > 350 else ''}</em>"
    )

    if show_original and p["book_key"] == "quran":
        arabic = meta.get("ayah_ar", "")
        if arabic:
            card_html += f"<div class='original-text'>{arabic}</div>"

    if show_original and p["book_key"] == "torah":
        hebrew = meta.get("Text_Hebrew", "")
        if hebrew:
            card_html += f"<div class='original-text'>{hebrew}</div>"

    card_html += "</div>"
    st.markdown(card_html, unsafe_allow_html=True)

    if show_word_meaning and p["book_key"] == "gita":
        word_meaning = meta.get("WordMeaning", "")
        if word_meaning:
            with st.expander(f"📖 Word meanings — {ref}"):
                st.caption(word_meaning)


# ── Chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if "passages" in msg and msg["passages"]:
                with st.expander("📖 Retrieved passages", expanded=False):
                    pcols = st.columns(len(BOOKS))
                    for col, (book_key, book_info) in zip(pcols, BOOKS.items()):
                        passages = msg["passages"].get(book_key, [])
                        with col:
                            st.markdown(
                                f"<span style='color:{book_info['color']}'>"
                                f"**{book_info['name']}**</span>",
                                unsafe_allow_html=True
                            )
                            for i, p in enumerate(passages, 1):
                                render_passage(p, book_info, i)

            st.markdown(msg["content"])


# ── Query handler ─────────────────────────────────────────────────────────────

def handle_query(query: str) -> None:
    """Core pipeline: retrieve → synthesize → display."""
    query = query.strip()
    if not query:
        return

    st.session_state.chat_messages.append({"role": "user", "content": query})

    is_followup       = False
    do_fresh_retrieve = True

    if (
        st.session_state.retrieved_passages is not None
        and st.session_state.last_query is not None
    ):
        drift = is_topic_drift(st.session_state.last_query, query)
        if not drift:
            is_followup       = True
            do_fresh_retrieve = False

    if do_fresh_retrieve:
        with st.spinner("🔍 Searching all four texts..."):
            passages = retrieve(query, top_k=top_k)
        st.session_state.retrieved_passages = passages
        st.session_state.last_query         = query
    else:
        passages = st.session_state.retrieved_passages

    # ─── FIX 3: CALL LOGGING PIPELINE AUTOMATICALLY ON SUBMISSION ───
    live_score = calculate_system_score(passages)
    log_retrieval_attempt(prompt=query, system_score=live_score, retrieved_results=passages)

    # Synthesize
    with st.spinner("✦ Comparing perspectives across traditions..."):
        answer = synthesize(
            question             = query,
            retrieved_passages   = passages,
            conversation_history = st.session_state.conversation_history,
            is_followup          = is_followup,
        )

    # Update history
    st.session_state.conversation_history.append({"role": "user",      "content": query})
    st.session_state.conversation_history.append({"role": "assistant",  "content": answer})
    st.session_state.chat_messages.append({
        "role":     "assistant",
        "content":  answer,
        "passages": passages,
    })

    st.rerun()


# Chip trigger
if selected_chip:
    handle_query(selected_chip)

# Text input
user_input = st.chat_input(
    "Ask about any theme — forgiveness, justice, the soul, death, compassion…"
)
if user_input:
    handle_query(user_input)

# ── Empty state ───────────────────────────────────────────────────────────────

if not st.session_state.chat_messages:
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#666'>
        <div style='font-size:2.5rem; letter-spacing:1rem; margin-bottom:1rem'>✦ ✦ ✦</div>
        <p style='font-size:1.1rem; font-style:italic'>
            Type a question or click a theme above.<br>
            Four traditions will answer together.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── FIX 1 & 2: RECONCILED TITLES & ASSIGNED ACCORDION EXPANDER FOR STABLE LOG VIEWS ───
st.markdown("---")
st.subheader("📊 System Performance Diagnostics")

# Swapped out unstable st.button inside dashboard layer for toggle container block
show_logs = st.toggle("🔍 Reveal Historical Diagnostic Logs", value=False)

if show_logs:
    if not os.path.exists(LOG_FILE_PATH):
        st.warning("No tracking performance logs found yet! Run a live query above to create your first entry.")
    else:
        df = pd.read_csv(LOG_FILE_PATH)
        
        avg_score = df["global_score"].mean()
        total_queries = len(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Queries Run", value=total_queries)
        with col2:
            status_emoji = "🌟" if avg_score >= 0.75 else "📈" if avg_score >= 0.60 else "⚠️"
            st.metric(label=f"{status_emoji} System Average Performance Score", value=f"{avg_score:.4f}")
            
        st.markdown("### Historical Retrieval Log Details")
        
        display_df = df.copy().sort_values(by="timestamp", ascending=False)
        
        def unpack_references(json_str):
            try:
                refs = json.loads(json_str)
                return " | ".join([f"{k.upper()}: {v}" for k, v in refs.items()])
            except:
                return json_str
                
        display_df["pulled_verses"] = display_df["pulled_verses"].apply(unpack_references)
        
        st.dataframe(
            display_df,
            column_config={
                "timestamp": "Date/Time",
                "prompt": "User Prompt",
                "global_score": st.column_config.NumberColumn("Global Score", format="%.4f"),
                "pulled_verses": "Top Passages Selected"
            },
            hide_index=True,
            use_container_width=True
        )
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Logs to CSV",
            data=csv_data,
            file_name="sacred_texts_rag_perf_logs.csv",
            mime="text/csv"
        )
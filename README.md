# HolyBooks-RAG-RecommenderSystem (Multi-Pocket RAG)

A production-grade, local Retrieval-Augmented Generation (RAG) pipeline designed to explore complex theological themes across four distinct traditions: **The King James Bible, The Quran, The Tanakh (JPS 1917), and the Bhagavad Gita**. 

This system solves cross-context imbalance by isolating texts into independent vector graphs, utilizing a fine-tuned QA embedding architecture to surface deep conceptual alignments instead of shallow keyword matches.

---

## Key Features & Architecture

* **Multi-Pocket Index Isolation:** Each sacred text is indexed into a dedicated, isolated collection inside ChromaDB. This prevents large datasets (e.g., the Bible) from mathematically crowding out smaller datasets (e.g., the Gita) during semantic scans.
* **Intent-Driven QA Embeddings:** Powered by `multi-qa-mpnet-base-dot-v1` (768 dimensions), fine-tuned on over 215M question-answering pairs to map natural user queries directly to scriptural resolution.
* **Optimized Inner Product (`ip`) Math:** Tailored for unnormalized vector configurations. It honors vector magnitude, rewarding passages that emphasize specific answers.
* **Neighbor-Context Expansion:** Automatically reconstructs complete narrative passages by dynamically fetching and stitching adjacent verses (`+/- 1` sequence indices) around high-scoring hits.
* **Retrieval Diagnostics:** Automatically scores global retrieval parity across traditions using a **Harmonic Mean Engine**, logging audit data into a persistent local analytics panel.

---

## Advanced Retrieval Layers

1. **LLM Query Expansion:** Pre-processes user prompts through Llama 3.3 70B to inject structural synonyms and conceptual phrasing adapted to archaic texts.
2. **Maximal Marginal Relevance (MMR):** Over-samples initial candidates (`top_k * 2`) and re-ranks them ($\lambda = 0.7$) to maximize topical diversity and eliminate structural redundancy.
3. **Sigmoid Normalization Pipeline:** Calibrates unnormalized negative dot-product values back into a standard, clean `0.0` to `1.0` percentage range with an anchored safety floor.

---

## Project Structure

📂 recommender_system/ <br>
├── 📂 chroma_db/                  # Local SQLite vector index table databases <br>
├── 📂 data/                       # Raw CSV scripture dataset resources <br>
├── 📄 setup.py/02_build_index.py  # Central schema blueprint, paths, and settings <br>
├── 📄 config.py                   # Central schema blueprint, paths, and settings <br>
├── 📄 retriever.py                # Multi-pocket retrieval, MMR, context expansion, & scaling <br>
├── 📄 synthesizer.py              # Llama 3.3 context synthesis engine via Groq <br>
├── 📄 app.py                      # Streamlit UI frontend & Performance Dashboard <br>
└── 📄 retrieval_perf.csv          # to track the relevance score (logs) <br>
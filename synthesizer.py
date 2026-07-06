"""
synthesizer.py — LLM Synthesis via Groq (Llama 3.3 70B)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Takes the bundled passages from the retriever and generates a structured
comparative response using Llama 3.3 70B through the Groq API.

Also handles the sticky-context logic for multi-turn conversations:
- Turn 1: fresh retrieval → generate response
- Turn 2+: reuse stored passages (unless topic drift detected)
"""

import os
from groq import Groq
from dotenv import load_dotenv
from retriever import format_context
from config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

load_dotenv()

# ── Groq client ───────────────────────────────────────────────────────────────

def _get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Copy .env.example to .env and add your key from https://console.groq.com"
        )
    return Groq(api_key=api_key)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scholar of comparative religion and philosophy — \
knowledgeable, balanced, and respectful of all traditions.

You will receive:
1. A user's question or theme
2. A set of relevant passages retrieved from four sacred texts:
   King James Bible (Christianity), The Quran (Islam),
   Torah JPS 1917 (Judaism), and the Bhagavad Gita (Hinduism)

Your task is to generate a structured comparative response with three sections:

## Overview
A 2–3 sentence synthesis of how all four traditions approach the theme.
Highlight both shared ground and key differences.

## Tradition by Tradition
For each of the four books, provide:
- **[Book Name] ([Tradition]):** A 3–5 sentence analysis grounded in the
  provided passage. What is this text's unique perspective? What language,
  metaphor, or concept does it use? How does it differ from the others?

## Key Takeaway
One paragraph comparing two or three of the texts directly.
What is the most interesting convergence or contrast across traditions?

Rules:
- Never declare one tradition superior or "correct".
- Stay grounded in the provided passages — don't invent quotes.
- Use accessible language suitable for all audiences.
- If a passage seems only loosely related to the question, say so briefly.
- Keep the total response under 600 words.
"""


# ── Turn prompt builder ───────────────────────────────────────────────────────

def _build_user_message(question: str, context: str) -> str:
    return f"""Question: {question}

Relevant passages retrieved from the four texts:
{context}

Please provide your comparative analysis."""


def _build_followup_message(question: str, context: str) -> str:
    return f"""Follow-up question: {question}

This is a follow-up to our previous discussion. Do NOT repeat the full 
overview from before. Instead:
- Build directly on the previous answer
- Focus on what is NEW or DEEPER about this specific follow-up
- Keep it concise — the user has already seen the overview

Same passages for reference:
{context}"""


# ── Main synthesis function ───────────────────────────────────────────────────

def synthesize(
    question: str,
    retrieved_passages: dict,
    conversation_history: list[dict],
    is_followup: bool = False
) -> str:
    """
    Generate a comparative answer from the LLM.

    Args:
        question:             The user's current question
        retrieved_passages:   Dict of {book_key: [passage dicts]} from retriever
        conversation_history: List of previous {role, content} turns for context
        is_followup:          True if this is a follow-up (uses sticky context)

    Returns:
        The LLM's comparative response as a string.
    """
    client  = _get_groq_client()
    context = format_context(retrieved_passages)

    # Build the new user message
    if is_followup:
        new_user_msg = _build_followup_message(question, context)
    else:
        new_user_msg = _build_user_message(question, context)

    # Assemble full message list: system + history + new turn
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history         # previous turns (for follow-ups)
    messages.append({"role": "user", "content": new_user_msg})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error calling Groq API: {e}\n\nCheck that your GROQ_API_KEY is set correctly in .env"

if __name__ == "__main__":
    from retriever import retrieve, format_context

    print("Testing Synthesizer...")
    query  = "Who is God?"
    result = retrieve(query, top_k=3)

    answer = synthesize(
        question             = query,
        retrieved_passages   = result,
        conversation_history = [],
        is_followup          = False,
    )
    print(answer)
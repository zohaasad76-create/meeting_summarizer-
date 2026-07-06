"""
qa_system.py — RAG-Based Query Answering Module
================================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

SRS References
--------------
§4.1.7  — Allow users to type natural language questions about the meeting
           and receive answers in English.
§5.1.9  — Accept a natural language question, combine it with the transcript
           as context, and send it to a language model to generate an answer.
§5.1.10 — Answer questions ONLY based on content present in the meeting
           transcript. If the topic is not mentioned, respond accordingly.
§5.1.11 — Retry failed model requests up to MAX_RETRIES times before raising.
§4.1.6  — All answers delivered entirely in English regardless of transcript
           language.

Design Reference (Assignment 3 §3.3)
--------------------------------------
    query_embedding    = embed(query)
    relevant_chunks    = similarity_search(query_embedding, transcript)
    If relevant_chunks is empty → return "Not found in meeting"
    context            = merge(relevant_chunks)
    answer             = LLM_generate(query, context)
    return answer

RAG Pipeline Explained
-----------------------
Retrieval-Augmented Generation (RAG) ensures the LLM answers only from the
actual meeting content (SRS §5.1.10), not from its general training knowledge.

Step 1 — CHUNK   : Split transcript into overlapping word-level windows.
Step 2 — EMBED   : Convert each chunk + the question into dense vectors using
                   the SentenceTransformer "all-MiniLM-L6-v2" model.
Step 3 — RETRIEVE: Compute cosine similarity between the question vector and
                   all chunk vectors; select the top-K most relevant chunks.
Step 4 — GENERATE: Pass the retrieved chunks as grounded context to Llama 3
                   via Groq, instructing it to answer only from that context.

Architecture Layer : AI Services Layer (Q&A sub-layer)
"""

import time

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    RAG_CHUNK_WORDS,
    RAG_TOP_K,
    MAX_RETRIES,
)


# ── Module-level singletons ───────────────────────────────────────────────────
# Both objects are expensive to initialise. Creating them once at import time
# means every call to answer_question() reuses the same loaded model and client
# without paying the startup cost again.

_client = Groq(api_key=GROQ_API_KEY)

print("[QASystem] Loading sentence-transformer embedding model...")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("[QASystem] Embedding model loaded ✅")


# ── Public entry point ────────────────────────────────────────────────────────

def answer_question(question: str, transcript: str) -> str:
    """
    Answer a natural language question using RAG over the meeting transcript.

    This is the only function called by main.py. It runs the full four-step
    RAG pipeline: chunk → embed → retrieve → generate.

    Parameters
    ----------
    question   : str
        The user's natural language question about the meeting.
    transcript : str
        Full English transcript stored in session memory.

    Returns
    -------
    str
        An English answer grounded strictly in the meeting transcript, or a
        "not mentioned" message if the topic does not appear in the content
        (SRS §5.1.10 alternative flow).

    Raises
    ------
    ValueError
        If the question or transcript is empty — caught by main.py and
        returned as an HTTP 400 error with a user-friendly message.
    RuntimeError
        If the Groq LLM call fails after MAX_RETRIES attempts.
    """
    # ── Input guards ──────────────────────────────────────────────────────────
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not transcript or not transcript.strip():
        raise ValueError(
            "No transcript available. Please upload and transcribe an audio "
            "file before asking questions."
        )

    print(f"[QASystem] Received question: '{question}'")

    # ── Step 1: Chunk the transcript ──────────────────────────────────────────
    chunks = _chunk_transcript(transcript)
    print(f"[QASystem] Transcript split into {len(chunks)} chunk(s) "
          f"of ~{RAG_CHUNK_WORDS} words each.")

    # ── Step 2 & 3: Embed + Retrieve top-K relevant chunks ───────────────────
    relevant_chunks = _retrieve_relevant_chunks(question, chunks)

    # ── Design §3.3 alternative flow: no relevant content found ──────────────
    if not relevant_chunks:
        print("[QASystem] No relevant chunks found — returning fallback.")
        return "This topic was not mentioned in the meeting."

    # ── Step 4: Generate answer grounded in retrieved context ─────────────────
    context = "\n\n".join(relevant_chunks)
    answer = _generate_answer_with_retry(question, context)

    print(f"[QASystem] Answer generated ✅ ({len(answer.split())} words)")
    return answer


# ── Internal helpers ──────────────────────────────────────────────────────────

def _chunk_transcript(transcript: str) -> list[str]:
    """
    Split the transcript into fixed-size word-level windows.

    Word-level chunking (vs character-level) preserves complete words at
    boundaries, which produces more coherent embedding vectors and better
    cosine similarity scores.

    Parameters
    ----------
    transcript : str
        Full merged English transcript from the audio processing module.

    Returns
    -------
    list[str]
        Ordered list of transcript segments, each containing at most
        RAG_CHUNK_WORDS words.
    """
    words = transcript.split()

    chunks = [
        " ".join(words[i: i + RAG_CHUNK_WORDS])
        for i in range(0, len(words), RAG_CHUNK_WORDS)
    ]

    # Filter out any empty strings that may result from extra whitespace
    return [c for c in chunks if c.strip()]


def _retrieve_relevant_chunks(question: str, chunks: list[str]) -> list[str]:
    """
    Embed the question and all transcript chunks, then retrieve the top-K
    most semantically similar chunks using cosine similarity.

    This is the core of the RAG retrieval step (Assignment 3 §3.3:
    similarity_search). The SentenceTransformer model maps both the question
    and chunks into the same high-dimensional vector space, allowing semantic
    (not just keyword) matching.

    Parameters
    ----------
    question : str
        The user's natural language question.
    chunks   : list[str]
        All word-level chunks from the transcript.

    Returns
    -------
    list[str]
        Up to RAG_TOP_K chunks ordered by descending relevance score.
        Returns an empty list if the transcript has no chunks.
    """
    if not chunks:
        return []

    # Encode all chunks in one batch call — more efficient than one-by-one
    chunk_embeddings = _embedder.encode(chunks, convert_to_numpy=True)

    # Encode the question as a single vector
    question_embedding = _embedder.encode([question], convert_to_numpy=True)

    # Compute cosine similarity between the question and every chunk
    # Shape: (1, num_chunks) → flatten to (num_chunks,)
    similarity_scores = cosine_similarity(question_embedding, chunk_embeddings)[0]

    # Sort indices by score descending, take top-K
    top_indices = np.argsort(similarity_scores)[::-1][:RAG_TOP_K]

    selected = [chunks[i] for i in top_indices]

    print(f"[QASystem] Top-{RAG_TOP_K} chunks retrieved. "
          f"Best similarity score: {similarity_scores[top_indices[0]]:.3f}")

    return selected


def _generate_answer_with_retry(question: str, context: str) -> str:
    """
    Call the Groq LLM to generate a grounded answer from the retrieved context.

    The system prompt strictly instructs the model to answer only from the
    provided context (SRS §5.1.10) and always in English (SRS §4.1.6).
    If the answer cannot be found, the model is instructed to return the
    standard fallback phrase rather than hallucinating.

    Retries up to MAX_RETRIES times with exponential back-off (SRS §5.1.11).

    Parameters
    ----------
    question : str
        The user's original question.
    context  : str
        Concatenated top-K transcript chunks used as the grounding context.

    Returns
    -------
    str
        The model's answer in English, grounded in the meeting transcript.

    Raises
    ------
    RuntimeError
        If all MAX_RETRIES attempts fail, with the last error message included.
    """
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise meeting assistant. "
                            "Answer questions ONLY using the transcript context "
                            "provided below. Do NOT use any outside knowledge. "
                            "Always respond in clear English only — never in Urdu "
                            "or any other language. "
                            "If the answer is not present in the context, respond "
                            "with exactly: "
                            "'This topic was not mentioned in the meeting.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Meeting Transcript Context:\n{context}\n\n"
                            f"Question: {question}\n\n"
                            "Answer (in English only, based strictly on the "
                            "context above):"
                        ),
                    },
                ],
                max_tokens=500,
                temperature=0.2,  # very low — factual accuracy over creativity
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            last_error = exc
            print(f"[QASystem] LLM call attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"[QASystem] Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Answer generation failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
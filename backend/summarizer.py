"""
summarizer.py — Meeting Summarization Module
=============================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

SRS References
--------------
§4.1.6  — Generate a structured meeting summary in English containing:
           Meeting Overview, Key Decisions, Action Items, Topics Discussed.
§5.1.7  — Send the English transcript to a language model to generate a
           structured summary with exactly those four sections.
§5.1.11 — Retry failed model requests up to MAX_RETRIES times before raising.
§5.2 (Performance) — 30-min meeting summarized within 3 minutes.

Design Reference (Assignment 3 §3.2)
--------------------------------------
Function summarize_text(transcript):
    If transcript is empty → return error
    Generate summary with FLAN-T5 (replaced with Llama 3 via Groq for
    production quality) using the four-section prompt
    Return structured summary

Architecture Layer : AI Services Layer (Summarization + Q&A sub-layer)

Implementation Notes
--------------------
Long transcripts are split into character-level chunks (SUMMARIZE_CHUNK_SIZE)
so each API call stays within the model's context window. Partial summaries
are then merged in a second pass to produce the final four-section output.
This mirrors the Assignment 3 alternative flow for long transcripts.
"""

import time

from groq import Groq

from backend.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    SUMMARIZE_CHUNK_SIZE,
    MAX_RETRIES,
)


# ── Groq client (single instance shared across this module) ───────────────────
_client = Groq(api_key=GROQ_API_KEY)


# ── Public entry point ────────────────────────────────────────────────────────

def summarize_transcript(transcript: str) -> str:
    """
    Generate a structured four-section meeting summary from an English transcript.

    Pipeline
    --------
    1. Validate input — return a clear error if transcript is empty.
    2. Split transcript into overlapping character chunks to handle long meetings.
    3. Summarize each chunk independently (partial summaries).
    4. Merge all partial summaries into one final structured output with the
       four sections mandated by SRS §4.1.6.

    Parameters
    ----------
    transcript : str
        Full English transcript produced by the audio_processor module.

    Returns
    -------
    str
        Structured summary string with clearly labelled sections:
            1. Meeting Overview
            2. Key Decisions
            3. Action Items
            4. Topics Discussed

    Raises
    ------
    ValueError
        If the transcript is empty or contains only whitespace (SRS §3.2
        design: "If transcript is empty → return error").
    RuntimeError
        If the Groq API call fails after MAX_RETRIES attempts on any chunk.
    """
    # Guard — empty transcript check (SRS §3.2 design alternative flow)
    if not transcript or not transcript.strip():
        raise ValueError(
            "Cannot generate summary: the transcript is empty. "
            "Please upload and transcribe an audio file first."
        )

    print(f"[Summarizer] Starting summarization — "
          f"{len(transcript.split())} words, "
          f"{len(transcript)} characters.")

    # Step 1 — Split into manageable chunks
    chunks = _chunk_text(transcript)
    print(f"[Summarizer] Transcript split into {len(chunks)} chunk(s).")

    # Step 2 — Summarize each chunk individually
    partial_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"[Summarizer] Summarizing chunk {i + 1}/{len(chunks)}...")
        partial = _summarize_chunk_with_retry(chunk, index=i + 1)
        partial_summaries.append(partial)

    # Step 3 — Merge partial summaries into one final structured output
    print("[Summarizer] Merging partial summaries into final output...")
    final_summary = _merge_summaries_with_retry(partial_summaries)

    print("[Summarizer] Summary generation complete ✅")
    return final_summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """
    Split a long transcript string into fixed-size character chunks.

    Each chunk is at most SUMMARIZE_CHUNK_SIZE characters so it fits within
    the LLM's context window. The split is naive (character-level) because
    the model handles partial sentences gracefully and the merge pass
    reconciles any boundary artefacts.

    Parameters
    ----------
    text : str
        Full transcript text to be chunked.

    Returns
    -------
    list[str]
        Ordered list of transcript segments ready for individual summarization.
    """
    return [
        text[i: i + SUMMARIZE_CHUNK_SIZE]
        for i in range(0, len(text), SUMMARIZE_CHUNK_SIZE)
    ]


def _summarize_chunk_with_retry(chunk: str, index: int) -> str:
    """
    Summarize a single transcript chunk with retry logic.

    Calls the Groq LLM with a focused prompt asking for bullet-point notes
    on the four key areas. Retries up to MAX_RETRIES times on failure with
    exponential back-off (SRS §5.1.11).

    Parameters
    ----------
    chunk : str
        A character-level slice of the full transcript.
    index : int
        1-based chunk index used only for log messages.

    Returns
    -------
    str
        Raw partial summary text from the model.

    Raises
    ------
    RuntimeError
        If all MAX_RETRIES attempts fail.
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
                            "You are an expert meeting summarizer. "
                            "Always respond in clear, professional English only. "
                            "Never use Urdu or any other language in your output."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Summarize the following portion of a meeting transcript.\n"
                            "Extract notes under these four headings:\n"
                            "- Key Points\n"
                            "- Decisions Made\n"
                            "- Action Items\n"
                            "- Topics Covered\n\n"
                            f"Transcript:\n{chunk}"
                        ),
                    },
                ],
                max_tokens=600,
                temperature=0.3,  # low temperature for factual, consistent output
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            last_error = exc
            print(f"[Summarizer] Chunk {index} attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"[Summarizer] Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Summarization failed for chunk {index} after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def _merge_summaries_with_retry(partial_summaries: list[str]) -> str:
    """
    Combine all partial summaries into one final structured meeting summary.

    Sends all partial notes to the LLM in a single prompt instructing it to
    deduplicate, consolidate, and format under the four SRS-mandated sections.
    Retries up to MAX_RETRIES times on failure (SRS §5.1.11).

    Parameters
    ----------
    partial_summaries : list[str]
        Raw bullet-point notes produced by _summarize_chunk_with_retry for
        each transcript chunk.

    Returns
    -------
    str
        Final structured summary with the four numbered sections:
            1. Meeting Overview
            2. Key Decisions
            3. Action Items
            4. Topics Discussed

    Raises
    ------
    RuntimeError
        If all MAX_RETRIES attempts for the merge call fail.
    """
    combined_notes = "\n\n---\n\n".join(partial_summaries)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert meeting summarizer. "
                            "Produce clean, professional output in English only. "
                            "Never use Urdu or any other language."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Below are partial notes extracted from different sections "
                            "of the same meeting transcript. Consolidate them into ONE "
                            "final structured summary. Remove duplicates. Be concise.\n\n"
                            "Use EXACTLY this format:\n\n"
                            "1. Meeting Overview\n"
                            "<2-3 sentence overview of what the meeting was about>\n\n"
                            "2. Key Decisions\n"
                            "• <decision>\n"
                            "• <decision>\n\n"
                            "3. Action Items\n"
                            "• <action item with owner if mentioned>\n"
                            "• <action item>\n\n"
                            "4. Topics Discussed\n"
                            "• <topic>\n"
                            "• <topic>\n\n"
                            f"Partial Notes:\n{combined_notes}"
                        ),
                    },
                ],
                max_tokens=900,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            last_error = exc
            print(f"[Summarizer] Merge attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"[Summarizer] Retrying merge in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Summary merge failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
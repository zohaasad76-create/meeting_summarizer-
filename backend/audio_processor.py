"""
audio_processor.py — Audio Processing & Transcription Module
=============================================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

SRS References
--------------
§4.1.1  — Accept uploaded audio files containing recorded meeting content.
§4.1.2  — Accept MP3, WAV, M4A, OGG formats only; reject all others with an
           error message.
§5.1.1  — Validate file format before processing begins.
§5.1.2  — Split files larger than 25 MB into chunks automatically.
§5.1.3  — Send each chunk to a multilingual speech recognition model (Whisper).
§5.1.4  — Merge all chunk transcriptions in correct order into one transcript.
§5.1.6  — Produce a final transcript entirely in English regardless of input
           language (handled by Whisper's translate task).
§5.1.11 — Retry failed model requests up to 3 times before raising an error.

Design Reference (Assignment 3 §3.1)
--------------------------------------
Class AudioProcessor with methods:
    validate_format()  → checks allowed extensions
    split_audio()      → chunks large files by size
    transcribe()       → calls Whisper per chunk with retry
    merge()            → joins chunk texts in order

Architecture Layer : Processing Layer → AI Services Layer
"""

import os
import time

from pydub import AudioSegment
from groq import Groq

from backend.config import (
    GROQ_API_KEY,
    MAX_FILE_SIZE_MB,
    ALLOWED_EXTENSIONS,
    WHISPER_MODEL,
    MAX_RETRIES,
)


# ── Groq client (single instance, reused across requests) ─────────────────────
_client = Groq(api_key=GROQ_API_KEY)


# ── Public entry point ────────────────────────────────────────────────────────

def process_audio(file_path: str) -> str:
    """
    Full pipeline: validate → split → transcribe → merge.

    This is the only function called by main.py. It orchestrates the three
    internal steps defined in the Assignment 3 detailed design.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the uploaded audio file on disk.

    Returns
    -------
    str
        A single, clean English transcript string with all chunks merged
        in the correct order.

    Raises
    ------
    ValueError
        If the file's extension is not in the supported set (MP3/WAV/M4A/OGG).
        This maps to SRS §5.1.1 alternative flow: display error to user.
    RuntimeError
        If transcription fails on any chunk after MAX_RETRIES attempts.
        This maps to SRS §5.1.11 retry policy.
    """
    # Step 1 — Validate format (SRS §5.1.1)
    _validate_format(file_path)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[AudioProcessor] File accepted: '{os.path.basename(file_path)}' "
          f"({file_size_mb:.2f} MB)")

    # Step 2 — Split into chunks if necessary (SRS §5.1.2)
    chunks = _split_audio(file_path)

    # Step 3 — Transcribe each chunk with retry (SRS §5.1.3 + §5.1.11)
    transcripts = _transcribe_chunks(chunks, original_path=file_path)

    # Step 4 — Merge in order (SRS §5.1.4)
    full_transcript = _merge(transcripts)

    print(f"[AudioProcessor] Transcription complete — "
          f"{len(full_transcript.split())} words total.")
    return full_transcript


# ── Internal helpers (mirror Assignment 3 class methods) ─────────────────────

def _validate_format(file_path: str) -> None:
    """
    Validate that the file extension is in the set of allowed audio formats.

    Implements SRS §4.1.2 and §5.1.1: the system shall accept MP3, WAV, M4A,
    and OGG only. Any other extension triggers an immediate rejection with a
    descriptive error message before any processing begins.

    Parameters
    ----------
    file_path : str
        Path to the uploaded file.

    Raises
    ------
    ValueError
        With a human-readable message listing the allowed formats so the
        frontend can display it directly to the user.
    """
    extension = os.path.splitext(file_path)[-1].lstrip(".").lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS)).upper()
        raise ValueError(
            f"Unsupported file format '.{extension}'. "
            f"Please upload one of the following formats: {allowed}."
        )

    print(f"[AudioProcessor] Format validation passed: .{extension}")


def _split_audio(file_path: str) -> list[str]:
    """
    Split the audio file into chunks if it exceeds MAX_FILE_SIZE_MB.

    Implements SRS §5.1.2: files larger than 25 MB are automatically split
    into 10-minute (600 000 ms) chunks so each fits within Groq's upload
    limit. Files within the size limit are returned as-is in a single-item
    list to keep the transcription loop uniform.

    Parameters
    ----------
    file_path : str
        Path to the original audio file.

    Returns
    -------
    list[str]
        Ordered list of file paths — either [file_path] for small files,
        or [chunk_0.mp3, chunk_1.mp3, …] for large files.
    """
    size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # No splitting needed — return original file wrapped in a list
    if size_mb <= MAX_FILE_SIZE_MB:
        print(f"[AudioProcessor] File size {size_mb:.1f} MB ≤ "
              f"{MAX_FILE_SIZE_MB} MB — no splitting required.")
        return [file_path]

    print(f"[AudioProcessor] File size {size_mb:.1f} MB > "
          f"{MAX_FILE_SIZE_MB} MB — splitting into 10-minute chunks...")

    audio = AudioSegment.from_file(file_path)
    chunk_duration_ms = 10 * 60 * 1000  # 10 minutes in milliseconds
    total_duration_ms = len(audio)

    chunk_paths: list[str] = []
    start_ms = 0
    index = 0

    while start_ms < total_duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk = audio[start_ms:end_ms]

        chunk_path = f"{file_path}_chunk_{index}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)

        print(f"[AudioProcessor] Chunk {index} saved: "
              f"{(end_ms - start_ms) / 1000:.0f}s → {chunk_path}")

        start_ms = end_ms
        index += 1

    print(f"[AudioProcessor] Created {len(chunk_paths)} chunks.")
    return chunk_paths


def _transcribe_single_chunk(chunk_path: str) -> str:
    """
    Call Whisper Large V3 via Groq to transcribe one audio chunk.

    Uses the 'translate' task so Whisper converts Urdu and Roman Urdu
    speech directly into English text, satisfying SRS §5.1.6 (output
    entirely in English regardless of input language).

    Parameters
    ----------
    chunk_path : str
        Path to a single audio chunk file.

    Returns
    -------
    str
        Raw transcription text in English for this chunk.

    Raises
    ------
    Exception
        Any Groq API or network error — the caller handles retries.
    """
    with open(chunk_path, "rb") as audio_file:
        result = _client.audio.translations.create(   # ← .translations not .transcriptions
    file=(os.path.basename(chunk_path), audio_file.read()),
    model=WHISPER_MODEL,
    response_format="text",
    # task parameter removed — translations.create() always outputs English
    prompt=(
        "Transcribe this meeting audio accurately. "
        "Output only English text. "
        "Include all speakers and discussions."
    ),
)

    # Groq returns either a plain string or an object with a .text attribute
    if isinstance(result, str):
        return result.strip()
    return result.text.strip()


def _transcribe_chunks(chunks: list[str], original_path: str) -> list[str]:
    """
    Transcribe every chunk with retry logic and clean up temp files.

    Implements SRS §5.1.3 (send to multilingual speech recognition model)
    and SRS §5.1.11 (retry up to MAX_RETRIES times before raising an error).

    Retry Strategy
    --------------
    - Attempt 1 : immediate
    - Attempt 2 : wait 2 seconds before retry
    - Attempt 3 : wait 4 seconds before retry
    - After MAX_RETRIES failures : raise RuntimeError

    Parameters
    ----------
    chunks        : list[str] — Ordered list of chunk file paths.
    original_path : str       — Path to the original file (never deleted).

    Returns
    -------
    list[str]
        Ordered list of transcript strings, one per chunk.

    Raises
    ------
    RuntimeError
        If any chunk fails all retry attempts, with the chunk index and
        the underlying error message included for debugging.
    """
    transcripts: list[str] = []

    for i, chunk_path in enumerate(chunks):
        print(f"[AudioProcessor] Transcribing chunk {i + 1}/{len(chunks)}: "
              f"{os.path.basename(chunk_path)}")

        last_error: Exception | None = None

        # ── Retry loop (SRS §5.1.11) ──────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                text = _transcribe_single_chunk(chunk_path)
                transcripts.append(text)
                print(f"[AudioProcessor] Chunk {i + 1} transcribed "
                      f"({len(text.split())} words) on attempt {attempt}.")
                break  # success — exit retry loop

            except Exception as exc:
                last_error = exc
                print(f"[AudioProcessor] Chunk {i + 1} attempt {attempt} "
                      f"failed: {exc}")

                if attempt < MAX_RETRIES:
                    wait = attempt * 2  # 2 s, 4 s between retries
                    print(f"[AudioProcessor] Retrying in {wait}s...")
                    time.sleep(wait)

        else:
            # All attempts exhausted — clean up and raise
            _cleanup_chunks(chunks, original_path)
            raise RuntimeError(
                f"Transcription failed for chunk {i + 1} after "
                f"{MAX_RETRIES} attempts. Last error: {last_error}"
            )

        # ── Clean up chunk temp file (keep original) ──────────────────────
        if chunk_path != original_path and os.path.exists(chunk_path):
            os.remove(chunk_path)
            print(f"[AudioProcessor] Temp chunk deleted: {chunk_path}")

    return transcripts


def _merge(transcripts: list[str]) -> str:
    """
    Join all chunk transcripts into one continuous English transcript.

    Implements SRS §5.1.4: merge all chunk transcriptions in correct order
    into a single complete transcript. A single space separates chunks to
    avoid double-spaces or missing word boundaries at chunk joins.

    Parameters
    ----------
    transcripts : list[str]
        Ordered transcript strings from each audio chunk.

    Returns
    -------
    str
        Single merged transcript string.
    """
    merged = " ".join(t for t in transcripts if t.strip())
    print(f"[AudioProcessor] Chunks merged — total length: "
          f"{len(merged)} characters.")
    return merged


def _cleanup_chunks(chunks: list[str], original_path: str) -> None:
    """
    Delete all temporary chunk files created during splitting.

    Called on error to ensure no orphaned temp files are left on disk,
    satisfying SRS §5.2 (Security): uploaded files processed temporarily
    and not stored permanently.

    Parameters
    ----------
    chunks        : list[str] — All chunk file paths to attempt deletion.
    original_path : str       — The original upload path — never deleted here.
    """
    for chunk_path in chunks:
        if chunk_path != original_path and os.path.exists(chunk_path):
            try:
                os.remove(chunk_path)
                print(f"[AudioProcessor] Cleanup: removed {chunk_path}")
            except OSError as e:
                print(f"[AudioProcessor] Cleanup warning: could not remove "
                      f"{chunk_path} — {e}")
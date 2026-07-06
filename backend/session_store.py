"""
session_store.py — Session Storage Module
==========================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

SRS References
--------------
§5.1.8  — Store the English transcript in session memory to allow multiple
           questions without re-processing the audio.
§5.2 (Security)   — Uploaded audio files should only be used for temporary
                    processing and should not be stored permanently on any
                    server. Session data expires automatically after 3 hours.
§5.2 (Modularity) — Storage is a separate module so it can be updated
                    independently of other components.

Design Reference (Assignment 3 §2.2 — Storage Layer)
------------------------------------------------------
- Stores session-based transcripts
- Prevents repeated processing
- Enables fast query response
- Uses Redis; falls back to in-memory dict if Redis is unavailable

Storage Strategy
----------------
PRIMARY   → Redis (recommended for production)
            Keys expire automatically after SESSION_EXPIRY_SEC (3 hours).
            Data survives server restarts within the TTL window.

FALLBACK  → In-process Python dict (development / environments without Redis)
            Data is lost if the server process restarts.
            Suitable for local testing and demo purposes.

The module detects which backend to use at import time by pinging Redis.
All four public functions (save_transcript, get_transcript, save_summary,
get_summary) work identically regardless of which backend is active.

Redis Key Schema
----------------
    meeting:transcript  → Full English transcript string
    meeting:summary     → Structured summary string

Both keys are set with a TTL of SESSION_EXPIRY_SEC so data is never stored
permanently, satisfying the SRS security requirement.
"""

import redis

from backend.config import (
    REDIS_HOST,
    REDIS_PORT,
    SESSION_EXPIRY_SEC,
)


# ── Redis connection attempt ───────────────────────────────────────────────────
# We try to connect and ping Redis at import time. If it succeeds, all
# subsequent calls use Redis. If it fails (e.g. Redis not installed in the
# dev environment), we silently fall back to the in-memory dict.

_redis_client: redis.Redis | None = None
_USE_REDIS: bool = False

try:
    _redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,   # return str, not bytes
        socket_connect_timeout=2,  # fail fast if Redis is unreachable
    )
    _redis_client.ping()
    _USE_REDIS = True
    print(f"[SessionStore] Redis connected at {REDIS_HOST}:{REDIS_PORT} ✅")

except redis.ConnectionError:
    _USE_REDIS = False
    print("[SessionStore] Redis not reachable — using in-memory fallback ⚠️")

except Exception as exc:
    _USE_REDIS = False
    print(f"[SessionStore] Redis init error ({exc}) — using in-memory fallback ⚠️")


# ── In-memory fallback store ──────────────────────────────────────────────────
# Only used when Redis is unavailable. A plain dict is sufficient for single-
# session demo purposes. In production, Redis should always be preferred.
_memory_store: dict[str, str] = {}


# ── Redis key constants ───────────────────────────────────────────────────────
_KEY_TRANSCRIPT = "meeting:transcript"
_KEY_SUMMARY    = "meeting:summary"


# ── Public API ────────────────────────────────────────────────────────────────

def save_transcript(text: str) -> None:
    """
    Persist the full English transcript to the active storage backend.

    Called by main.py immediately after audio_processor returns a transcript,
    so subsequent /summarize and /ask requests can retrieve it without
    reprocessing the audio (SRS §5.1.8).

    Parameters
    ----------
    text : str
        Full English transcript string produced by the audio processing module.

    Returns
    -------
    None

    Side Effects
    ------------
    - Redis  : Sets key 'meeting:transcript' with a TTL of SESSION_EXPIRY_SEC.
    - Memory : Stores value under key 'transcript' in the fallback dict.
    """
    if _USE_REDIS and _redis_client:
        _redis_client.setex(_KEY_TRANSCRIPT, SESSION_EXPIRY_SEC, text)
        print(f"[SessionStore] Transcript saved to Redis "
              f"(TTL: {SESSION_EXPIRY_SEC}s, {len(text)} chars).")
    else:
        _memory_store["transcript"] = text
        print(f"[SessionStore] Transcript saved to memory ({len(text)} chars).")


def get_transcript() -> str:
    """
    Retrieve the stored English transcript from the active storage backend.

    Returns an empty string rather than None so callers can safely check
    truthiness without additional None guards.

    Returns
    -------
    str
        The transcript string, or an empty string if no transcript has been
        stored yet (e.g. user calls /summarize before /upload).
    """
    if _USE_REDIS and _redis_client:
        value = _redis_client.get(_KEY_TRANSCRIPT)
        return value if value is not None else ""
    return _memory_store.get("transcript", "")


def save_summary(text: str) -> None:
    """
    Persist the structured meeting summary to the active storage backend.

    Called by main.py after summarizer.py returns the four-section summary,
    so it can be included in the downloadable report without regenerating.

    Parameters
    ----------
    text : str
        Structured summary string with the four SRS-mandated sections.

    Returns
    -------
    None

    Side Effects
    ------------
    - Redis  : Sets key 'meeting:summary' with a TTL of SESSION_EXPIRY_SEC.
    - Memory : Stores value under key 'summary' in the fallback dict.
    """
    if _USE_REDIS and _redis_client:
        _redis_client.setex(_KEY_SUMMARY, SESSION_EXPIRY_SEC, text)
        print(f"[SessionStore] Summary saved to Redis "
              f"(TTL: {SESSION_EXPIRY_SEC}s, {len(text)} chars).")
    else:
        _memory_store["summary"] = text
        print(f"[SessionStore] Summary saved to memory ({len(text)} chars).")


def get_summary() -> str:
    """
    Retrieve the stored structured summary from the active storage backend.

    Returns an empty string rather than None so callers can safely check
    truthiness without additional None guards.

    Returns
    -------
    str
        The summary string, or an empty string if no summary has been
        generated yet.
    """
    if _USE_REDIS and _redis_client:
        value = _redis_client.get(_KEY_SUMMARY)
        return value if value is not None else ""
    return _memory_store.get("summary", "")


def clear_session() -> None:
    """
    Delete both the transcript and summary from the active storage backend.

    Utility function for clearing stale session data between uploads, which
    prevents a new audio file's Q&A from accidentally returning answers based
    on the previous meeting's transcript.

    This function is not currently exposed as an API endpoint but is available
    for future use or testing.

    Returns
    -------
    None

    Side Effects
    ------------
    - Redis  : Deletes 'meeting:transcript' and 'meeting:summary' keys.
    - Memory : Removes both keys from the fallback dict if present.
    """
    if _USE_REDIS and _redis_client:
        _redis_client.delete(_KEY_TRANSCRIPT, _KEY_SUMMARY)
        print("[SessionStore] Session cleared from Redis.")
    else:
        _memory_store.pop("transcript", None)
        _memory_store.pop("summary", None)
        print("[SessionStore] Session cleared from memory.")


def get_storage_backend() -> str:
    """
    Return a human-readable label for the currently active storage backend.

    Useful for health-check endpoints or admin logging to confirm whether
    the production Redis instance is being used.

    Returns
    -------
    str
        Either "Redis" or "In-Memory (fallback)".
    """
    return "Redis" if _USE_REDIS else "In-Memory (fallback)"
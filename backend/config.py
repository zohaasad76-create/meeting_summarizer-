"""
config.py — Centralized Configuration Module
=============================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

Purpose
-------
Loads all environment variables from the .env file at startup and exposes
them as typed constants used across every other backend module.

Security Rules
--------------
- NO secret or API key is ever hardcoded in this file or anywhere else.
- All sensitive values MUST live in the .env file which is git-ignored.
- If a required variable is missing, a clear RuntimeError is raised at
  import time so the problem is caught before any request is processed.

Environment Variables Required (.env)
--------------------------------------
    GROQ_API_KEY        — Groq cloud API key for Whisper + Llama inference
    MAX_FILE_SIZE_MB    — (optional) chunk size limit, defaults to 25 MB
    SESSION_EXPIRY_SEC  — (optional) Redis TTL in seconds, defaults to 10800 (3 h)
    REDIS_HOST          — (optional) Redis hostname, defaults to "localhost"
    REDIS_PORT          — (optional) Redis port, defaults to 6379
"""

import os
from dotenv import load_dotenv

# Load variables from .env into the process environment.
# Must be called before any os.getenv() call in this module.
load_dotenv()


# ── Helper ────────────────────────────────────────────────────────────────────

def _require(name: str) -> str:
    """
    Fetch a required environment variable.

    Parameters
    ----------
    name : str
        The environment variable key to look up.

    Returns
    -------
    str
        The variable's value as a string.

    Raises
    ------
    RuntimeError
        If the variable is not set, so misconfiguration is caught at startup
        rather than silently failing mid-request.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"[CONFIG ERROR] Required environment variable '{name}' is not set. "
            f"Add it to your .env file and restart the server."
        )
    return value


def _optional_int(name: str, default: int) -> int:
    """
    Fetch an optional integer environment variable with a fallback default.

    Parameters
    ----------
    name    : str  — Environment variable key.
    default : int  — Value to use when the variable is absent or empty.

    Returns
    -------
    int
        Parsed integer value.
    """
    raw = os.getenv(name)
    if raw and raw.strip().isdigit():
        return int(raw.strip())
    return default


def _optional_str(name: str, default: str) -> str:
    """
    Fetch an optional string environment variable with a fallback default.

    Parameters
    ----------
    name    : str — Environment variable key.
    default : str — Value to use when the variable is absent or empty.

    Returns
    -------
    str
        The variable's value or the provided default.
    """
    return os.getenv(name, default).strip() or default


# ── Public Constants ──────────────────────────────────────────────────────────
# These are imported by all other backend modules.

# Groq API key — required; raises RuntimeError if absent
GROQ_API_KEY: str = _require("GROQ_API_KEY")

# Maximum audio chunk size before splitting (SRS §5.1.2 — 25 MB)
MAX_FILE_SIZE_MB: int = _optional_int("MAX_FILE_SIZE_MB", 25)

# Redis session TTL in seconds (SRS §5.1.8 — store for session duration)
# Default: 3 hours = 10 800 seconds
SESSION_EXPIRY_SEC: int = _optional_int("SESSION_EXPIRY_SEC", 10_800)

# Redis connection settings
REDIS_HOST: str = _optional_str("REDIS_HOST", "localhost")
REDIS_PORT: int = _optional_int("REDIS_PORT", 6379)

# Supported audio formats (SRS §4.1.2 — MP3, WAV, M4A, OGG only)
ALLOWED_EXTENSIONS: set[str] = {"mp3", "wav", "m4a", "ogg"}

# Maximum number of retry attempts for AI model calls (SRS §5.1.11)
MAX_RETRIES: int = 3

# Whisper model used via Groq (SRS §1.3 — openai/whisper-large-v3)
WHISPER_MODEL: str = "whisper-large-v3"

# Llama model used via Groq for summarization and Q&A
# (SRS §1.3 — google/flan-t5-large in design; replaced with Llama 3 via Groq
#  for production-quality output without local GPU requirement)
LLM_MODEL: str = "llama-3.3-70b-versatile"

# Transcript chunk size for summarization (characters)
SUMMARIZE_CHUNK_SIZE: int = 4_000

# Transcript chunk size for RAG retrieval (words)
RAG_CHUNK_WORDS: int = 200

# Number of top chunks to retrieve in RAG (SRS §3.3)
RAG_TOP_K: int = 3
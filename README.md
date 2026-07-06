# 🎙️ MeetAI — AI Meeting Summarizer & Query System

> **Course:** Software Engineering · Semester Project — Implementation & Deployment  
> **Authors:** Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)  
> **Stack:** FastAPI · Whisper Large V3 · Llama 3.3 · Groq API · RAG · Redis

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [System Architecture](#-system-architecture)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Setup & Installation](#-setup--installation)
7. [Environment Variables](#-environment-variables)
8. [Running the Project](#-running-the-project)
9. [API Endpoints](#-api-endpoints)
10. [How It Works](#-how-it-works)
11. [SRS Requirements Coverage](#-srs-requirements-coverage)
12. [Team Members](#-team-members)

---

## 📖 Project Overview

**MeetAI** is a web-based AI system that automatically processes meeting audio recordings and converts them into structured, searchable information.

Users can upload a recorded meeting audio file in **English, Urdu, or Roman Urdu**. The system will:

- Transcribe the audio into English text using **Whisper Large V3**
- Generate a **structured 4-section summary** using **Llama 3.3 via Groq**
- Allow users to **ask natural language questions** about the meeting using a **RAG pipeline**
- Let users **download** the full transcript and summary as a `.txt` report

The goal is to eliminate manual note-taking and ensure that important decisions and action items from meetings are never lost.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙️ Audio Upload | Supports MP3, WAV, M4A, OGG formats |
| ✂️ Auto-Chunking | Splits large files (>25 MB) into 10-minute chunks automatically |
| 🌐 Multilingual | Handles English, Urdu, and Roman Urdu — outputs English only |
| 📋 Smart Summary | 4-section structured summary: Overview, Decisions, Action Items, Topics |
| 💬 RAG Q&A | Ask anything about the meeting — answers grounded only in transcript |
| 🔁 Retry Logic | Retries failed AI calls up to 3 times with exponential back-off |
| 💾 Session Storage | Redis (with in-memory fallback) — no re-processing on follow-up questions |
| ⬇️ Download Report | Export full transcript + summary as a `.txt` file |
| 🔒 Secure | Audio files are never stored permanently (temporary processing only) |

---

## 🏗️ System Architecture

The system follows a **Layered Architecture** as defined in Assignment 3:

```
┌─────────────────────────────┐
│        Frontend UI          │  index.html (HTML/CSS/JS)
│  Upload · Summary · Q&A     │
└──────────────┬──────────────┘
               │ HTTP (REST)
               ▼
┌─────────────────────────────┐
│        Backend API          │  main.py (FastAPI)
│  /upload · /summarize       │
│  /ask · /download · /status │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│      Processing Layer       │  audio_processor.py
│  Validate · Split · Merge   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       AI Services Layer     │
│  Whisper  →  Transcription  │  audio_processor.py
│  Llama 3  →  Summarization  │  summarizer.py
│  RAG      →  Q&A System     │  qa_system.py
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       Storage Layer         │  session_store.py
│   Redis  /  In-Memory       │
└─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3.11+, FastAPI |
| Speech Recognition | Whisper Large V3 (via Groq API) |
| Summarization & Q&A | Llama 3.3 70B Versatile (via Groq API) |
| Embeddings (RAG) | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Similarity Search | `scikit-learn` — Cosine Similarity |
| Audio Processing | `pydub` |
| Session Storage | Redis (with in-memory fallback) |
| Config Management | `python-dotenv` |

---

## 📁 Project Structure

```
meetai/
│
├── index.html                  # Frontend — complete single-page UI
│
├── backend/
│   ├── __init__.py
│   ├── config.py               # Environment variables & constants
│   ├── main.py                 # FastAPI app — all API endpoints
│   ├── audio_processor.py      # Validate → Split → Transcribe → Merge
│   ├── summarizer.py           # Chunk-based LLM summarization
│   ├── qa_system.py            # RAG pipeline — embed, retrieve, generate
│   └── session_store.py        # Redis + in-memory session storage
│
├── uploads/                    # Temporary audio files (auto-created)
├── .env                        # Secret keys — NOT committed to git
├── .env.example                # Template for environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.11** or higher
- `pip` (Python package manager)
- A **Groq API key** — get one free at [console.groq.com](https://console.groq.com)
- Redis *(optional — system falls back to in-memory storage automatically)*

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/meetai.git
cd meetai
```

---

### Step 2 — Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
fastapi
uvicorn[standard]
python-dotenv
groq
pydub
redis
sentence-transformers
scikit-learn
numpy
```

---

### Step 4 — Configure Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Then open `.env` and add your Groq API key (see [Environment Variables](#-environment-variables) below).

---

### Step 5 — Install ffmpeg (required by pydub)

`pydub` requires `ffmpeg` to read audio files.

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (with Homebrew)
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# and add it to your system PATH
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root. **Never commit this file to Git.**

```env
# ── Required ──────────────────────────────────────────────
# Your Groq API key — get one at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here

# ── Optional (defaults shown) ──────────────────────────────
# Maximum audio chunk size before splitting (MB)
MAX_FILE_SIZE_MB=25

# Redis session expiry in seconds (default: 3 hours)
SESSION_EXPIRY_SEC=10800

# Redis connection settings
REDIS_HOST=localhost
REDIS_PORT=6379
```

An `.env.example` template is included in the repository.

> ⚠️ **Security Note:** The `.env` file contains secrets. It is listed in `.gitignore` and must never be pushed to any public repository.

---

## 🚀 Running the Project

### Start the Backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
[SessionStore] Redis connected at localhost:6379 ✅
[QASystem] Embedding model loaded ✅
```

> If Redis is not running, you will see a warning but the system continues with in-memory storage — no action needed for local development.

---

### Open the Frontend

Simply open `index.html` in your browser:

```bash
# macOS
open index.html

# Windows
start index.html

# Or just double-click index.html in File Explorer / Finder
```

The frontend connects to `http://127.0.0.1:8000` automatically.

---

### Verify the Backend is Running

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. You should see:

```json
{ "message": "MeetAI Running ✅", "status": "ok" }
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — confirms server is running |
| `GET` | `/status` | Returns session state and storage backend info |
| `POST` | `/upload` | Upload audio file → returns full English transcript |
| `POST` | `/summarize` | Generate 4-section summary from stored transcript |
| `POST` | `/ask` | Ask a question — RAG answer from transcript context |
| `GET` | `/download` | Download transcript + summary as `MeetAI_Report.txt` |

### Example — Upload Audio

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@meeting.mp3"
```

**Response:**
```json
{
  "filename": "meeting.mp3",
  "transcript": "The meeting began at 10 AM...",
  "word_count": 1240
}
```

### Example — Ask a Question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the action items?"}'
```

**Response:**
```json
{
  "question": "What were the action items?",
  "answer": "The action items were: 1. Ahmed to send the report by Friday..."
}
```

---

## 🔍 How It Works

### 1. Upload & Transcription Pipeline

```
User uploads audio
       │
       ▼
Validate file format (MP3/WAV/M4A/OGG)
       │
       ▼
File > 25 MB?  ──Yes──▶  Split into 10-min chunks
       │No
       ▼
Whisper Large V3 transcribes each chunk
(translate task → always outputs English)
       │
       ▼
Merge all chunks into one full transcript
       │
       ▼
Store transcript in Redis / Memory session
```

### 2. Summarization Pipeline

```
Retrieve transcript from session
       │
       ▼
Split into 4000-character chunks
       │
       ▼
Llama 3.3 summarizes each chunk
       │
       ▼
Merge partial summaries
       │
       ▼
Llama 3.3 produces final 4-section output:
  1. Meeting Overview
  2. Key Decisions
  3. Action Items
  4. Topics Discussed
```

### 3. RAG Q&A Pipeline

```
User asks a question
       │
       ▼
Split transcript into 200-word chunks
       │
       ▼
Embed question + all chunks
(sentence-transformers: all-MiniLM-L6-v2)
       │
       ▼
Cosine similarity → retrieve Top-3 chunks
       │
       ▼
Llama 3.3 generates answer from context only
(never uses outside knowledge)
       │
       ▼
Return answer to user
```

### 4. Retry Logic

Every AI model call is wrapped in a retry loop (SRS §5.1.11):

| Attempt | Wait Before Retry |
|---|---|
| 1st | Immediate |
| 2nd | 2 seconds |
| 3rd | 4 seconds |
| After 3rd failure | Raise error → HTTP 500 |

---

## ✅ SRS Requirements Coverage

| SRS Requirement | Status | Implemented In |
|---|---|---|
| §4.1.1 — Upload audio files | ✅ | `main.py /upload`, `index.html` |
| §4.1.2 — Accept MP3/WAV/M4A/OGG only | ✅ | `main.py`, `audio_processor.py`, `index.html` |
| §4.1.3 — Transcribe speech to text | ✅ | `audio_processor.py` |
| §4.1.4 — Handle English, Urdu, Roman Urdu | ✅ | Whisper `translate` task |
| §4.1.5 — Translate all content to English | ✅ | Whisper `translate` task |
| §4.1.6 — Generate 4-section summary | ✅ | `summarizer.py` |
| §4.1.7 — Natural language Q&A | ✅ | `qa_system.py` |
| §4.1.8 — Support long meetings | ✅ | Chunking in `audio_processor.py` |
| §4.1.9 — Display full transcript | ✅ | `index.html` transcript panel |
| §4.1.10 — Download as .txt file | ✅ | `main.py /download` |
| §5.1.1 — Server-side format validation | ✅ | `main.py`, `audio_processor.py` |
| §5.1.2 — Auto-split files > 25 MB | ✅ | `audio_processor.py` |
| §5.1.3 — Multilingual speech model | ✅ | Whisper Large V3 |
| §5.1.4 — Merge chunks in order | ✅ | `audio_processor.py` |
| §5.1.8 — Store transcript in session | ✅ | `session_store.py` |
| §5.1.9 — Combine question + context | ✅ | `qa_system.py` |
| §5.1.10 — Answer only from transcript | ✅ | System prompt in `qa_system.py` |
| §5.1.11 — Retry failed calls × 3 | ✅ | All AI modules |
| §5.1.12 — Downloadable .txt report | ✅ | `main.py /download` |
| §5.2 Security — No permanent storage | ✅ | Session TTL + temp files deleted |
| §5.2 Reliability — Error messages | ✅ | HTTP 400/500 + toast notifications |
| §5.2 Compatibility — Browser access | ✅ | Pure HTML/JS, no installation needed |

---

## 👩‍💻 Team Members

| Name | Student ID | Role |
|---|---|---|
| Fizzah Amir | BSCS 24031 | Backend Development, AI Integration |
| Zoha Asad | BSCS 24135 | Frontend Development, System Design |

---

## 📄 License

This project was developed as a semester project for academic purposes.  
© 2025 Fizzah Amir & Zoha Asad — BSCS, University.
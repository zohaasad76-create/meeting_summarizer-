"""
main.py — FastAPI Backend Entry Point
======================================
Project  : AI Meeting Summarizer & Query System
Authors  : Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)

SRS References
--------------
§4.1.1  — Allow users to upload audio files containing recorded meeting content.
§4.1.2  — Accept MP3, WAV, M4A, OGG; reject all others with an error message.
§4.1.6  — Generate a structured meeting summary with four sections.
§4.1.7  — Allow users to ask natural language questions and receive answers.
§4.1.9  — Display the full English transcript after processing.
§4.1.10 — Allow users to download the full transcript and summary as a .txt file.
§5.1.1  — Validate file format before processing (enforced here AND in
           audio_processor for defence-in-depth).
§5.1.8  — Store transcript in session memory for multi-question use.
§5.1.11 — Model errors and retry handled inside each sub-module; main.py
           catches and converts them into clean HTTP responses.
§5.2 (Security)     — Files used for temporary processing only.
§5.2 (Reliability)  — Invalid/unsupported formats handled without crashing;
                       appropriate error messages returned.
§5.2 (Compatibility)— Accessible via modern browsers; CORS enabled for all
                       origins during development.

Architecture Layer : Backend API Layer (sits between Frontend UI and
                     Processing / AI Services layers)

Endpoint Summary
----------------
GET  /           → Health check — confirms the server is running.
GET  /status     → Returns storage backend info and session state.
POST /upload     → Accepts audio file, validates format, runs full
                   transcription pipeline, stores transcript in session.
POST /summarize  → Reads transcript from session, generates four-section
                   summary, stores it in session.
POST /ask        → Reads transcript from session, runs RAG Q&A pipeline,
                   returns answer grounded in meeting content.
GET  /download   → Builds and streams a .txt report containing the full
                   transcript and summary.
"""

import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.audio_processor import process_audio
from backend.config import ALLOWED_EXTENSIONS
from backend.qa_system import answer_question
from backend.session_store import (
    clear_session,
    get_storage_backend,
    get_summary,
    get_transcript,
    save_summary,
    save_transcript,
)
from backend.summarizer import summarize_transcript


# ── Application instance ──────────────────────────────────────────────────────
app = FastAPI(
    title="MeetAI — AI Meeting Summarizer & Query System",
    description=(
        "Transcribes meeting audio (English / Urdu / Roman Urdu), "
        "generates structured summaries, and answers questions via RAG."
    ),
    version="1.0.0",
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allow the frontend (served from any origin during development) to call the
# API. In production this should be restricted to the deployed frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Upload directory ──────────────────────────────────────────────────────────
# Temporary storage for incoming audio files and generated report files.
# Audio files are processed and not stored permanently (SRS §5.2 Security).
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Pydantic request models ───────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """
    Request body schema for the POST /ask endpoint.

    Using a Pydantic model instead of a raw dict enforces type validation,
    produces automatic OpenAPI documentation, and raises a structured 422
    error if the client sends malformed JSON — satisfying the SRS reliability
    requirement (§4.2) without any manual validation code.

    Attributes
    ----------
    question : str
        The user's natural language question about the meeting. Must be a
        non-empty string of at most 500 characters.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language question about the meeting content.",
        examples=["What were the main action items from the meeting?"],
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _validate_extension(filename: str) -> None:
    """
    Validate that the uploaded file has an allowed audio extension.

    This is a defence-in-depth check at the API layer. The audio_processor
    module performs the same check, but catching it here lets us return a
    clean HTTP 400 before writing anything to disk (SRS §5.1.1, §4.2
    Reliability).

    Parameters
    ----------
    filename : str
        Original filename from the uploaded file (e.g. "meeting.mp3").

    Raises
    ------
    HTTPException (400)
        If the extension is not in ALLOWED_EXTENSIONS, with a message that
        lists the supported formats so the user knows exactly what to upload.
    """
    extension = os.path.splitext(filename)[-1].lstrip(".").lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS)).upper()
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '.{extension}'. "
                f"Please upload one of the following formats: {allowed}."
            ),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary="Health Check",
    tags=["System"],
)
def home() -> dict:
    """
    Health check endpoint.

    Confirms the FastAPI server is running and reachable. Returns a simple
    JSON object with a status message. Used by deployment monitors and the
    frontend to verify connectivity before making functional requests.

    Returns
    -------
    dict
        JSON: {"message": "MeetAI Running ✅", "status": "ok"}
    """
    return {"message": "MeetAI Running ✅", "status": "ok"}


@app.get(
    "/status",
    summary="Session & Storage Status",
    tags=["System"],
)
def status() -> dict:
    """
    Return the current session state and active storage backend.

    Useful for debugging and for the frontend to check whether a transcript
    is already loaded before enabling the summarise / ask buttons.

    Returns
    -------
    dict
        JSON containing:
        - storage_backend : "Redis" or "In-Memory (fallback)"
        - transcript_loaded : bool — True if a transcript is in session.
        - summary_loaded    : bool — True if a summary is in session.
    """
    return {
        "storage_backend": get_storage_backend(),
        "transcript_loaded": bool(get_transcript()),
        "summary_loaded": bool(get_summary()),
    }


@app.post(
    "/upload",
    summary="Upload Audio & Transcribe",
    tags=["Core Features"],
)
async def upload_file(file: UploadFile = File(...)) -> dict:
    """
    Accept an audio file, validate its format, transcribe it, and store the
    transcript in session memory.

    Pipeline
    --------
    1. Validate file extension (SRS §5.1.1) — return HTTP 400 on failure.
    2. Save uploaded file to the uploads/ directory temporarily.
    3. Clear any existing session data so the new transcript is fresh.
    4. Run the audio processing pipeline: validate → split → transcribe → merge.
    5. Store the resulting English transcript in session memory (SRS §5.1.8).
    6. Return the filename and transcript to the frontend.

    Parameters
    ----------
    file : UploadFile
        Multipart audio file uploaded from the frontend form.
        Accepted formats: MP3, WAV, M4A, OGG (SRS §4.1.2).

    Returns
    -------
    dict
        JSON containing:
        - filename   : str — Original uploaded filename.
        - transcript : str — Full English transcript of the meeting.
        - word_count : int — Number of words in the transcript.

    Raises
    ------
    HTTPException (400)
        If the file format is not supported.
    HTTPException (500)
        If transcription fails after all retry attempts.
    """
    # ── Step 1: Format validation (SRS §5.1.1) ────────────────────────────────
    _validate_extension(file.filename)

    # ── Step 2: Save file to disk temporarily ─────────────────────────────────
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[API /upload] File saved: {file_path}")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {exc}",
        )

    # ── Step 3: Clear stale session data ──────────────────────────────────────
    # Prevents Q&A answers from being based on the previous meeting's
    # transcript when the user uploads a new file.
    clear_session()

    # ── Step 4 & 5: Transcribe and store ─────────────────────────────────────
    try:
        transcript = process_audio(file_path)
        save_transcript(transcript)
    except ValueError as exc:
        # Format validation error from audio_processor (defence-in-depth)
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # Transcription failure after all retries exhausted
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during transcription: {exc}",
        )

    word_count = len(transcript.split())
    print(f"[API /upload] Transcript ready — {word_count} words.")

    return {
        "filename": file.filename,
        "transcript": transcript,
        "word_count": word_count,
    }


@app.post(
    "/summarize",
    summary="Generate Structured Meeting Summary",
    tags=["Core Features"],
)
def summarize() -> dict:
    """
    Generate a four-section structured summary from the stored transcript.

    Reads the English transcript from session memory and sends it through the
    summarization module. The resulting summary is also stored in session so
    it can be included in the downloadable report without regenerating.

    Four sections produced (SRS §4.1.6, §5.1.7)
    ---------------------------------------------
    1. Meeting Overview
    2. Key Decisions
    3. Action Items
    4. Topics Discussed

    Returns
    -------
    dict
        JSON containing:
        - summary : str — Full structured summary text with four sections.

    Raises
    ------
    HTTPException (400)
        If no transcript is found in session (user must upload first).
    HTTPException (500)
        If summarization fails after all retry attempts.
    """
    # ── Guard: transcript must exist ──────────────────────────────────────────
    transcript = get_transcript()
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail=(
                "No transcript found in session. "
                "Please upload and transcribe an audio file first."
            ),
        )

    # ── Generate and store summary ────────────────────────────────────────────
    try:
        summary = summarize_transcript(transcript)
        save_summary(summary)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during summarization: {exc}",
        )

    print("[API /summarize] Summary stored in session.")
    return {"summary": summary}


@app.post(
    "/ask",
    summary="Ask a Question About the Meeting",
    tags=["Core Features"],
)
def ask(data: QuestionRequest) -> dict:
    """
    Answer a natural language question using RAG over the stored transcript.

    The question is embedded and compared against transcript chunks using
    cosine similarity. The most relevant chunks are passed to the LLM as
    grounded context. The model is strictly instructed to answer only from
    that context (SRS §5.1.10).

    Users may ask multiple questions without re-uploading the audio because
    the transcript is cached in session memory (SRS §5.1.8).

    Parameters
    ----------
    data : QuestionRequest
        Pydantic-validated request body containing the user's question.

    Returns
    -------
    dict
        JSON containing:
        - question : str — The original question (echoed for UI convenience).
        - answer   : str — English answer grounded in the meeting transcript,
                           or "This topic was not mentioned in the meeting."
                           if the content is absent (SRS §5.1.10 alt flow).

    Raises
    ------
    HTTPException (400)
        If no transcript is found in session.
    HTTPException (500)
        If the LLM call fails after all retry attempts.
    """
    # ── Guard: transcript must exist ──────────────────────────────────────────
    transcript = get_transcript()
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail=(
                "No transcript found in session. "
                "Please upload and transcribe an audio file first."
            ),
        )

    # ── Run RAG pipeline ──────────────────────────────────────────────────────
    try:
        answer = answer_question(data.question, transcript)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during Q&A: {exc}",
        )

    return {
        "question": data.question,
        "answer": answer,
    }


@app.get(
    "/download",
    summary="Download Transcript & Summary Report",
    tags=["Core Features"],
)
def download() -> FileResponse:
    """
    Build and stream a formatted .txt report containing the full transcript
    and the structured summary.

    The report is written to the uploads/ directory and served as a file
    download. This satisfies SRS §4.1.10 (users can download the full
    transcript and summary as a text file) and §5.1.12 (generate a
    downloadable .txt file when requested).

    The file is generated fresh on each request to ensure it always reflects
    the latest session data. If no transcript or summary exists, placeholder
    text is included so the file is always valid.

    Returns
    -------
    FileResponse
        Streams "MeetAI_Report.txt" to the client as an attachment.

    Raises
    ------
    HTTPException (500)
        If the file cannot be written to disk.
    """
    transcript = get_transcript()
    summary    = get_summary()

    # ── Build report content ──────────────────────────────────────────────────
    separator  = "=" * 60
    divider    = "-" * 40

    report_lines = [
        separator,
        "        MeetAI — AI Meeting Summarizer Report",
        "        Fizzah Amir (BSCS 24031) · Zoha Asad (BSCS 24135)",
        separator,
        "",
        "FULL TRANSCRIPT",
        divider,
        transcript if transcript else "No transcript available.",
        "",
        "MEETING SUMMARY",
        divider,
        summary if summary else "No summary available. Click 'Generate Summary' first.",
        "",
        separator,
        "Report generated by MeetAI — AI Meeting Summarizer & Query System",
        separator,
    ]

    report_content = "\n".join(report_lines)

    # ── Write to disk ─────────────────────────────────────────────────────────
    report_path = os.path.join(UPLOAD_DIR, "MeetAI_Report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[API /download] Report written: {report_path} "
              f"({len(report_content)} chars)")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write report file: {exc}",
        )

    return FileResponse(
        path=report_path,
        filename="MeetAI_Report.txt",
        media_type="text/plain",
    )
"""FabricAI — minimal FastAPI backend with session persistence and file uploads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Mounted as a sub-app under the swatchon FastAPI server. Frontend lives
# in server/fabricai_frontend/static.
_project_root = Path(__file__).resolve().parents[2]
_static_dir = _project_root / "server" / "fabricai_frontend" / "static"

from .openai_client import ask, get_openai_model
from .session_store import (
    UPLOADS_DIR,
    append_message,
    create_session,
    delete_session,
    read_session,
)
from .context_provider import (
    build_system_instructions,
    ROLES,
    TOPICS,
    LANGUAGES,
    STARTER_QUESTIONS,
)

app = FastAPI(title="FabricAI", version="0.8.0")

# Mounted under /fabricai of the host app; same-origin to the rest of the UI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve fabricai's own static assets (the standalone /fabricai/ chat UI).
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Serve uploaded files. Create the directory eagerly so StaticFiles doesn't fail.
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Resolved model name for the /api/config endpoint.
_MODEL = get_openai_model()


# --- Filename sanitisation ---

_SAFE_RE = re.compile(r"[^\w.\-]")


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    return _SAFE_RE.sub("_", name)


# --- Request / Response models ---

# Max images per request and max size per data URL (~10 MB base64).
_MAX_IMAGES = 3
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class AskRequest(BaseModel):
    session_id: str
    question: str
    attachments: Optional[List[Dict]] = None
    images: Optional[List[str]] = None
    role: Optional[str] = None        # one of context_provider.ROLES keys
    topic: Optional[str] = None       # one of context_provider.TOPICS keys
    language: Optional[str] = None    # one of context_provider.LANGUAGES keys, or free-form for "other"


class AskResponse(BaseModel):
    answer: str
    model: str


class SessionResponse(BaseModel):
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict]


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page chat UI."""
    html_path = _static_dir / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def api_config():
    """Return client-visible configuration: model name + the role/topic/language
    taxonomies the UI should render. Single source of truth.
    """
    return {
        "model": _MODEL,
        "roles": [{"id": k, "label": v} for k, v in ROLES.items()],
        "topics": [{"id": k, "label": v} for k, v in TOPICS.items()],
        "languages": [{"id": k, "label": v} for k, v in LANGUAGES.items()],
        "starter_questions": STARTER_QUESTIONS,
        "default_topic": "explain_results",
        "default_language": "en",
    }


@app.post("/api/session", response_model=SessionResponse)
async def api_create_session():
    """Create a new chat session and return its id."""
    session_id = create_session()
    return SessionResponse(session_id=session_id)


@app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
async def api_get_session(session_id: str):
    """Return the full message history for a session."""
    messages = read_session(session_id)
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.post("/api/session/{session_id}/reset", response_model=SessionResponse)
async def api_reset_session(session_id: str):
    """Wipe the given session's history and create a fresh session.

    Returns the new session_id; the client should switch to it.
    """
    delete_session(session_id)
    new_id = create_session()
    return SessionResponse(session_id=new_id)


@app.post("/api/upload")
async def api_upload(
    session_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Accept one or more file uploads for a session.

    Files are stored under backend/data/uploads/<session_id>/.
    Returns an attachments array with metadata and a servable URL.
    """
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    attachments: list[dict] = []
    for f in files:
        safe_name = _safe_filename(f.filename or "file")
        dest = session_dir / safe_name
        content = await f.read()
        dest.write_bytes(content)
        attachments.append({
            "name": safe_name,
            "mime": f.content_type or "application/octet-stream",
            "size": len(content),
            "url": f"/uploads/{session_id}/{safe_name}",
        })

    return {"attachments": attachments}


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest):
    """Forward the user's question to OpenAI, persist both messages, and return the answer."""
    try:
        # Validate images if provided.
        images = req.images or []
        if len(images) > _MAX_IMAGES:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Too many images (max {_MAX_IMAGES})."},
            )
        for img in images:
            if len(img) > _MAX_IMAGE_BYTES:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Image too large (max ~10 MB)."},
                )

        # Persist the user message (with optional attachments).
        append_message(
            req.session_id,
            role="user",
            content=req.question,
            attachments=req.attachments,
        )

        # Build the knowledge-grounded system prompt: onboarding doc +
        # snapshot of recent swatchon results + the strict-source rules.
        instructions = build_system_instructions(
            user_question=req.question,
            role=req.role,
            topic=req.topic,
            language=req.language,
        )

        # Call OpenAI (with images and grounded system prompt).
        result = ask(
            req.question,
            images=images if images else None,
            instructions=instructions,
        )

        # Persist the assistant response.
        append_message(
            req.session_id,
            role="assistant",
            content=result["answer"],
            model=result["model"],
        )

        return AskResponse(**result)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

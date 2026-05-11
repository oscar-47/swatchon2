"""JSONL-based session history storage.

Each session is a single file: backend/data/sessions/<session_id>.jsonl
Each line is a JSON object with: ts, role, content, and optionally model and attachments.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Directory where session files live (created lazily).
_SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"

# Directory where uploaded files are stored.
UPLOADS_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"

# Polished prologue shown at the start of every new session.
PROLOGUE = (
    "Welcome to FabricAI, your AI-powered assistant for the fabric supply chain. "
    "I can assist you with anything related to your supply chain, and more. "
    "How can I help you today?"
)


def _ensure_dir() -> None:
    """Create the sessions directory if it does not exist."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_path(session_id: str) -> Path:
    return _SESSIONS_DIR / f"{session_id}.jsonl"


def create_session() -> str:
    """Create a new session, write the prologue, and return the session_id."""
    _ensure_dir()
    session_id = str(uuid.uuid4())
    # Write the system prologue as the first message.
    append_message(session_id, role="system", content=PROLOGUE)
    return session_id


def append_message(
    session_id: str,
    *,
    role: str,
    content: str,
    model: str | None = None,
    attachments: list[dict] | None = None,
) -> dict:
    """Append a single message to the session's JSONL file and return it."""
    _ensure_dir()
    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "content": content,
    }
    if model is not None:
        entry["model"] = model
    if attachments:
        entry["attachments"] = attachments

    with open(_session_path(session_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def read_session(session_id: str) -> list[dict]:
    """Return all messages for the given session, or an empty list if not found."""
    path = _session_path(session_id)
    if not path.exists():
        return []
    messages: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def delete_session(session_id: str) -> bool:
    """Remove a session's history file. Returns True if deleted, False if missing."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False

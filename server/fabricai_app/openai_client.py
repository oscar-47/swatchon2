"""Thin wrapper around the OpenAI Python SDK (Responses API).

Credential resolution lives here too. Order of precedence:
  1. The OPENAI_API_KEY / OPENAI_MODEL environment variable, if set.
  2. The constants in fabricai/backend/app/config.py, if that file exists.
  3. Raise a clear RuntimeError pointing the user to the README.

config.py is intentionally git-ignored — request it from the project
administrator. The defensive import below means the rest of the module
still loads if config.py is missing, so a clone-and-run workflow that
exports OPENAI_API_KEY in the shell still works.
"""

from __future__ import annotations

import os

from openai import OpenAI

# --- Defensive load of the local config file --------------------------------

try:
    from .config import OPENAI_API_KEY as _FILE_KEY
    from .config import OPENAI_MODEL as _FILE_MODEL
except ImportError:
    # config.py wasn't shipped — that's fine, env-var override may still cover us.
    _FILE_KEY = ""
    _FILE_MODEL = ""

_PLACEHOLDER_KEY = "sk-your-key-here"


# --- Credential resolution --------------------------------------------------

def get_openai_api_key() -> str:
    """Return the OpenAI API key.

    Resolution order (first non-empty, non-placeholder wins):
      1. The OPENAI_API_KEY environment variable.
      2. The OPENAI_API_KEY constant in fabricai/backend/app/config.py.

    Raises RuntimeError if neither is set or if the value is the placeholder.
    """
    key = os.environ.get("OPENAI_API_KEY") or _FILE_KEY
    if not key or key == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Either obtain config.py from the "
            "project administrator and place it at fabricai/backend/app/config.py, "
            "or export OPENAI_API_KEY in your shell. "
            "See README.md → 'Setting your OpenAI API key'."
        )
    return key


def get_openai_model() -> str:
    """Return the OpenAI model name (env var overrides the file constant)."""
    return os.environ.get("OPENAI_MODEL") or _FILE_MODEL or "gpt-5.4-mini"


# --- OpenAI client ----------------------------------------------------------

# Initialise once at module level.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_openai_api_key())
    return _client


def ask(
    question: str,
    model: str | None = None,
    images: list[str] | None = None,
    instructions: str | None = None,
) -> dict:
    """Send *question* (and optional base64 images) to OpenAI.

    Returns {"answer": ..., "model": ...}.
    When *images* is provided, builds a multi-part content array so the
    model can see the images via the Responses API input_image type.
    When *instructions* is provided, it becomes the Responses API system
    prompt — used for knowledge grounding (see context_provider.py).
    """
    model = model or get_openai_model()
    client = _get_client()

    if images:
        # Build structured input with text + image parts.
        content: list[dict] = [{"type": "input_text", "text": question}]
        for data_url in images:
            content.append({
                "type": "input_image",
                "image_url": data_url,
            })
        api_input = [{"role": "user", "content": content}]
    else:
        # Plain text — simple string input.
        api_input = question

    kwargs: dict = {"model": model, "input": api_input}
    if instructions:
        kwargs["instructions"] = instructions

    response = client.responses.create(**kwargs)

    return {
        "answer": response.output_text,
        "model": model,
    }

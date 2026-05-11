"""Thin wrapper around the OpenAI Python SDK (Responses API).

Credential resolution — supports both vanilla OpenAI and Azure OpenAI.

Provider selection:
  - If AZURE_OPENAI_ENDPOINT is set, the Azure Responses API is used.
    Required env vars in that mode:
      AZURE_OPENAI_ENDPOINT       (e.g. https://<resource>.openai.azure.com)
      AZURE_OPENAI_API_KEY        Azure resource key
      AZURE_OPENAI_DEPLOYMENT     deployment name (used as the model arg)
    Optional:
      AZURE_OPENAI_API_VERSION    default: 2025-04-01-preview
  - Otherwise, vanilla OpenAI is used, with OPENAI_API_KEY / OPENAI_MODEL.

All values come from env vars or the git-ignored local config.py — never
hard-coded — so no provider URL or key ships in the source tree.
"""

from __future__ import annotations

import os

from openai import AzureOpenAI, OpenAI

# --- Defensive load of the local config file --------------------------------

try:
    from .config import OPENAI_API_KEY as _FILE_KEY
    from .config import OPENAI_MODEL as _FILE_MODEL
except ImportError:
    # config.py wasn't shipped — that's fine, env-var override may still cover us.
    _FILE_KEY = ""
    _FILE_MODEL = ""

_PLACEHOLDER_KEY = "sk-your-key-here"
_DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"


# --- Provider selection -----------------------------------------------------

def _azure_endpoint() -> str:
    return (os.environ.get("AZURE_OPENAI_ENDPOINT") or "").rstrip("/")


def _is_azure() -> bool:
    return bool(_azure_endpoint())


# --- Credential resolution --------------------------------------------------

def get_openai_api_key() -> str:
    """Return the active API key for the configured provider.

    In Azure mode this returns AZURE_OPENAI_API_KEY; otherwise the standard
    OpenAI key (env var first, then config.py).
    """
    if _is_azure():
        key = os.environ.get("AZURE_OPENAI_API_KEY") or ""
        if not key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT is set but AZURE_OPENAI_API_KEY is empty."
            )
        return key

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
    """Return the model / deployment name to use in API calls.

    Azure mode: AZURE_OPENAI_DEPLOYMENT (the SDK passes this as `model`).
    Default mode: OPENAI_MODEL env var, then config.py, then a sensible default.
    """
    if _is_azure():
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or _FILE_MODEL
        if not deployment:
            raise RuntimeError(
                "AZURE_OPENAI_DEPLOYMENT must be set when AZURE_OPENAI_ENDPOINT is."
            )
        return deployment
    return os.environ.get("OPENAI_MODEL") or _FILE_MODEL or "gpt-5.4-mini"


# --- OpenAI / AzureOpenAI client -------------------------------------------

# Initialise once at module level; type widened to cover either SDK class.
_client: OpenAI | AzureOpenAI | None = None


def _get_client() -> OpenAI | AzureOpenAI:
    global _client
    if _client is None:
        if _is_azure():
            _client = AzureOpenAI(
                azure_endpoint=_azure_endpoint(),
                api_key=get_openai_api_key(),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION")
                            or _DEFAULT_AZURE_API_VERSION,
            )
        else:
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

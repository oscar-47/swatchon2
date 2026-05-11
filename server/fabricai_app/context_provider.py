"""Knowledge grounding for FabricAI.

For every user question, this module produces a system-prompt string that:

  1. Carries the Fabric Recognition onboarding document (mtime-cached).
  2. Carries a snapshot of the user's most recent classification results.
  3. Lays down strict rules that confine the model to those two sources
     for platform-specific questions.
  4. Optionally adds a ROLE LENS (which dimensions of the answer to
     emphasise — content differs by role, not voice/persona) and a
     TOPIC SCOPE (which permission grant: strict CONTEXT-only vs.
     CONTEXT + general knowledge).
  5. Optionally instructs the model to respond in a non-English language.
  6. Mandates a "Sources:" citations block at the end of every answer.

The model itself is the gatekeeper for the restriction — there are no
tools exposed to it, so it physically cannot read anything outside the
prompt. The prompt is the contract.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

log = logging.getLogger(__name__)

# --- Configuration ---------------------------------------------------------

_FABRICAI_ROOT = Path(__file__).resolve().parents[2]   # swatchon2/
_FABRICFLOW_ROOT = _FABRICAI_ROOT.parent                # fabricflow/

ONBOARDING_PATH = Path(
    os.environ.get(
        "FABRICAI_ONBOARDING_PATH",
        # Look first at the swatchon2 root, then fall back to fabricflow/.
        str((_FABRICAI_ROOT / "onboarding_explain.md") if (_FABRICAI_ROOT / "onboarding_explain.md").exists()
            else (_FABRICFLOW_ROOT / "onboarding_explain.md")),
    )
)

# URL of the Fabric Recognition (swatchon) results API.
SWATCHON_BASE_URL = os.environ.get("SWATCHON_BASE_URL", "http://localhost:8000")
SWATCHON_HTTP_TIMEOUT = float(os.environ.get("SWATCHON_HTTP_TIMEOUT", "1.5"))
MAX_RESULTS_IN_CONTEXT = int(os.environ.get("FABRICAI_MAX_RESULTS_IN_CONTEXT", "5"))


# --- Public taxonomies (consumed by the frontend via /api/roles /api/topics)

ROLES: dict[str, str] = {
    "commercial":     "Commercial",
    "retail":         "Retail",
    "sourcing":       "Sourcing",
    "operations":     "Operations",
    "sustainability": "Sustainability",
}

TOPICS: dict[str, str] = {
    "explain_results":    "Explain Results",
    "fabric_knowledge":   "Fabric Knowledge",
    "explain_technology": "Explain Technology",
    "supply_chain":       "Supply Chain",
}

# Five starter questions per topic. Same count across topics for tidy UI.
STARTER_QUESTIONS: dict[str, list[str]] = {
    "explain_results": [
        "What does my latest Fabric Recognition result mean?",
        "Why isn't the system 100% confident about this fabric?",
        "What kinds of garments could be made from this fabric?",
        "How accurate is this classification, and what would make it more reliable?",
        "Is this fabric a single layer or a double-cloth?",
    ],
    "fabric_knowledge": [
        "How is the detected fabric type traditionally constructed?",
        "What's the difference between Twill and Satin?",
        "What's the difference between Jersey and Rib Knit?",
        "What's the difference between knit and woven fabrics?",
        "Which fabric types drape softly, and which hold structure?",
    ],
    "explain_technology": [
        "How does Fabric Recognition work, in plain terms?",
        "How does the Digital Fabric Passport work?",
        "How does FabricAI access Fabric Recognition results?",
        "What is multi-view consensus and why does it matter?",
        "How does FabricAI know when to refuse a question?",
    ],
    "supply_chain": [
        "Which fabric type is the most economical to source at scale?",
        "Which fabric construction has the lowest carbon footprint in manufacturing?",
        "Where are the major manufacturing hubs for this kind of fabric?",
        "Which certifications matter most for this kind of fabric?",
        "What's a typical lead time for sourcing this fabric type?",
    ],
}

LANGUAGES: dict[str, str] = {
    "en":    "English",
    "zh":    "中文",
    "tr":    "Türkçe",
    "it":    "Italiano",
    "es":    "Español",
    "other": "Other",
}


# --- Role lenses — evidence-routing, not persona costume ------------------
# Each lens is a small set of decision dimensions. The model is instructed
# to emphasise / de-emphasise these dimensions when looking at the same
# underlying CONTEXT. Voice stays neutral — only content priorities shift.

ROLE_LENS: dict[str, str] = {
    "commercial": """\
ROLE LENS — COMMERCIAL / STRATEGY
Emphasise: market positioning, value proposition, brand-fit story, retail
price-point implications, margin and competitiveness signal.
De-emphasise: deep mill-side process detail, niche specifications.
End the answer with one short line framed as a commercial decision
("Recommend: keep / shelve / re-position").""",
    "retail": """\
ROLE LENS — RETAIL / CONSUMER
Emphasise: end-consumer relevance — care instructions, comfort, durability,
shelf-appeal signals, story angles a salesperson can use.
De-emphasise: B2B sourcing or manufacturing-floor detail.
End the answer with one short line framed as a retail-floor decision
("Recommend: front-of-rack / mid-rack / pass").""",
    "sourcing": """\
ROLE LENS — SOURCING / PROCUREMENT
Emphasise: specification match (composition, weight, width, structure),
MOQ feasibility, lead time, supplier risk, certifications relevant to
procurement (OEKO-TEX, GOTS, BCI, GRS).
De-emphasise: aesthetic / styling judgements, end-consumer storytelling.
End the answer with one concrete next step a sourcing manager could put
on a Monday-morning to-do list.""",
    "operations": """\
ROLE LENS — OPERATIONS / PRODUCTION
Emphasise: production fit, batch consistency, defect-risk surface,
finishing and shrinkage considerations, throughput, tolerance bands.
De-emphasise: brand positioning narrative.
End the answer with a production go / no-go recommendation and the
single most important tolerance to watch.""",
    "sustainability": """\
ROLE LENS — SUSTAINABILITY / COMPLIANCE
Emphasise: certifications (organic, recycled, OEKO-TEX, GOTS, GRS),
composition (natural vs. synthetic, recycled content), origin
transparency, LCA dimensions where evidence exists, EU DPP-style
compliance signals.
De-emphasise: pure aesthetics or commercial framing without an impact
angle.
Be cautious about claims you cannot evidence — prefer "evidence shows
X" over "X is true". End with a compliance status and the single
piece of missing evidence that would let you make a firm call.""",
}


# --- Topic scopes — permission grant for general knowledge ----------------

TOPIC_SCOPE: dict[str, str] = {
    "explain_results": """\
TOPIC SCOPE — EXPLAIN RESULTS
This question is about a specific Fabric Recognition result.
You MUST answer using ONLY the CONTEXT (recent results + onboarding doc).
If the result the user references is not in the CONTEXT, say so plainly.""",
    "fabric_knowledge": """\
TOPIC SCOPE — FABRIC KNOWLEDGE
This question is about fabric construction, terminology, or fibre
science. You MAY use general textile knowledge alongside the CONTEXT.
Stay grounded in textile fact; do not improvise specifications you
can't justify.""",
    "explain_technology": """\
TOPIC SCOPE — EXPLAIN TECHNOLOGY
This question is about how the platform works.
You MUST answer using ONLY the CONTEXT (onboarding doc).
Do NOT speculate about model architecture, training data, or
implementation details that are not in the doc. If the user asks for
those, decline politely.""",
    "supply_chain": """\
TOPIC SCOPE — SUPPLY CHAIN
This question is about sourcing, manufacturing, sustainability or
supply-chain operations. You MAY use general supply-chain and textile
knowledge alongside the CONTEXT. Where you cite numerical facts, give
the typical industry source category (e.g. "industry LCA databases",
"trade-press surveys") rather than inventing precise figures.""",
}


# --- Onboarding-doc cache (mtime-keyed) -----------------------------------

_onboarding_cache: dict[str, Any] = {"mtime": None, "text": None}


def load_onboarding() -> str:
    """Return the onboarding document text. Reloads if the file changed."""
    try:
        st = ONBOARDING_PATH.stat()
    except FileNotFoundError:
        log.warning("onboarding doc not found at %s", ONBOARDING_PATH)
        return ""

    if _onboarding_cache["mtime"] != st.st_mtime:
        try:
            text = ONBOARDING_PATH.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("onboarding doc unreadable: %s", exc)
            return _onboarding_cache.get("text") or ""
        _onboarding_cache["mtime"] = st.st_mtime
        _onboarding_cache["text"] = text
    return _onboarding_cache["text"] or ""


# --- Fabric Recognition results fetch -------------------------------------

def _http_get_json(url: str, timeout: float = SWATCHON_HTTP_TIMEOUT) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def fetch_recent_results(limit: int = MAX_RESULTS_IN_CONTEXT) -> list[dict]:
    url = f"{SWATCHON_BASE_URL.rstrip('/')}/api/results/recent?limit={int(limit)}"
    payload = _http_get_json(url)
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    return results if isinstance(results, list) else []


def fetch_full_result(fabric_id: str) -> dict | None:
    if not fabric_id or "/" in fabric_id:
        return None
    url = f"{SWATCHON_BASE_URL.rstrip('/')}/api/results/{fabric_id}"
    payload = _http_get_json(url)
    return payload if isinstance(payload, dict) else None


# --- Base rules block -----------------------------------------------------

_BASE_RULES = """\
You are FabricAI, the assistant for the Fabric Recognition platform.

STRICT RULES — follow exactly:

1. PLATFORM-SPECIFIC QUESTIONS — when the user asks about Fabric
   Recognition, the Digital Fabric Passport, the FabricAI assistant,
   the user interface, or what a specific result field means:
   - You MUST answer using ONLY the CONTEXT provided below, unless
     the TOPIC SCOPE block explicitly grants you general knowledge.
   - If the answer is not in the CONTEXT and not allowed by the
     scope, reply with exactly:
     "I don't have that information in my Fabric Recognition knowledge base."
   - Then optionally suggest where the user might find it (e.g. the
     user manual, the Fabric Recognition support team) — but do NOT
     invent details.

2. RECENT RESULTS — if the user references a specific classification
   result, look first in the "RECENT RESULTS" block of the CONTEXT.
   If the fabric_id they reference is there, answer from that record.
   If it isn't there, say so plainly.

3. NEVER reveal: API endpoint URLs, file paths, environment variables,
   credentials, model architectures, training data sources, or anything
   else that is not present in the CONTEXT. If the user asks for those,
   politely decline.

4. NEVER claim to have run a classification or fetched live data — you
   only see what the CONTEXT shows. Phrase observations accordingly.

5. CITATIONS — every answer MUST end with a "Sources:" line listing
   what you drew on. Use these exact source-tags:
     - "Onboarding §X" for sections of the onboarding document
     - "Result {fabric_id}" for a specific recent result
     - "General textile knowledge" for non-CONTEXT facts (only
       allowed when the TOPIC SCOPE explicitly permits)
     - "General supply-chain knowledge" for supply-chain facts
   Format example:
     Sources: Onboarding §3.2; Result 2766d39be300; General textile knowledge.
   If you genuinely have no source, write "Sources: none" — do not
   fabricate a citation.

6. Be concise. Plain prose. No marketing fluff.
"""


# --- System-prompt builder ------------------------------------------------

def build_system_instructions(
    user_question: str | None = None,
    role: str | None = None,
    topic: str | None = None,
    language: str | None = None,
    recent_limit: int = MAX_RESULTS_IN_CONTEXT,
) -> str:
    """Compose the full system-prompt string for one /api/ask call.

    Args:
        user_question: The user's question. Currently unused but kept
            for future selective grounding.
        role: One of the keys in ROLES, or None. Adds a ROLE LENS block.
        topic: One of the keys in TOPICS, or None. Adds a TOPIC SCOPE
            block; without it, defaults to platform-strict (no general
            knowledge).
        language: An ISO key in LANGUAGES, or a free-form string for
            "other". If None or "en", no language directive is added.
        recent_limit: Cap on how many recent results to include.
    """
    onboarding = load_onboarding()
    results = fetch_recent_results(limit=recent_limit)

    parts: list[str] = [_BASE_RULES]

    if role and role in ROLE_LENS:
        parts.append("\n" + ROLE_LENS[role])

    if topic and topic in TOPIC_SCOPE:
        parts.append("\n" + TOPIC_SCOPE[topic])

    if language and language not in (None, "en"):
        lang_label = LANGUAGES.get(language, language)
        parts.append(
            "\nLANGUAGE — respond in "
            f"{lang_label}. The user's question may be in any language; "
            "your reply must be in the requested language. Source-tag "
            "names in the citations block stay in English."
        )

    # Build the CONTEXT
    if results:
        results_block = json.dumps(results, ensure_ascii=False, indent=2)
    else:
        results_block = (
            "(no results available — Fabric Recognition has produced none "
            "yet, or the Fabric Recognition server is offline.)"
        )

    onboarding_block = onboarding or (
        "(onboarding document unavailable. Tell the user you cannot "
        "answer onboarding questions right now.)"
    )

    parts.append("\n=== CONTEXT BEGIN ===\n")
    parts.append("## RECENT RESULTS (newest first)\n\n" + results_block + "\n")
    parts.append("\n## FABRIC RECOGNITION ONBOARDING & USER GUIDE\n\n"
                 + onboarding_block + "\n")
    parts.append("=== CONTEXT END ===\n")

    return "".join(parts)

"""Independent support-relation review for quote-backed claims.

The claim extractor labels its own quote as direct support. High-materiality
claims must not inherit that self-label, so this module asks a separate model
call to classify the relationship between the claim, exact quote, and fetched
source excerpt. Failures remain ``None`` and therefore fail closed at the gate.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage

from dr_core.models.enums import SupportRelation
from dr_core.models.ledger import Claim

_SYSTEM = (
    "You independently review whether a quoted span supports a factual claim. "
    "Classify conservatively. supports_directly means the quote states the claim without an unstated inferential step; "
    "supports_inferentially means the claim follows only through reasoning; qualifies means the quote supports a narrower or caveated version; "
    "context_only means it supplies background but not the proposition; contradicts means it conflicts; stale means it is explicitly obsolete for the claim's time frame. "
    "Do not copy the extractor's label. Respond with strict JSON only."
)
_SHAPE = '{"relation":"supports_directly|supports_inferentially|qualifies|context_only|contradicts|stale","reasoning":"short audit note"}'


def _get_relation_model():
    """``DR_RELATION_REVIEW_MODEL`` overrides which model reviews support
    relations; absent that, this falls back to the verify-tier default owned
    by ``dr_core.models.tiers``."""
    from deerflow.models import create_chat_model
    from dr_core.models import tiers

    return create_chat_model(os.environ.get("DR_RELATION_REVIEW_MODEL") or tiers.verify_model_name())


def _usage(response) -> dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}
    if response is None:
        return totals
    meta = getattr(response, "usage_metadata", None) or {}
    totals["input_tokens"] = int(meta.get("input_tokens") or 0)
    totals["output_tokens"] = int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    totals["cache_read_tokens"] = int(details.get("cache_read") or 0)
    totals["cache_creation_tokens"] = int(details.get("cache_creation") or 0)
    totals["calls"] = 1
    return totals


def _parse(text: str) -> tuple[SupportRelation | None, str | None]:
    try:
        payload = json.loads(text.strip())
    except (AttributeError, TypeError, ValueError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    try:
        relation = SupportRelation(payload.get("relation"))
    except ValueError:
        return None, None
    note = str(payload.get("reasoning") or "").strip()[:2000] or None
    return relation, note


async def review_support_relation(
    claim: Claim,
    source: dict | None,
    evidence_excerpt: str | None,
) -> tuple[SupportRelation | None, str | None, dict[str, int]]:
    if claim.support is None or not claim.support.quote.strip():
        return None, None, _usage(None)
    prompt = {
        "claim": claim.text,
        "supporting_quote": claim.support.quote,
        "source": {
            "title": (source or {}).get("title"),
            "url_or_id": (source or {}).get("url_or_id"),
        },
        "fetched_source_excerpt": evidence_excerpt,
        "response_shape": _SHAPE,
    }
    try:
        response = await _get_relation_model().ainvoke([SystemMessage(content=_SYSTEM), HumanMessage(content=json.dumps(prompt, sort_keys=True))])
    except Exception:
        return None, None, _usage(None)
    content = response.content if isinstance(response.content, str) else str(response.content)
    relation, note = _parse(content)
    return relation, note, _usage(response)

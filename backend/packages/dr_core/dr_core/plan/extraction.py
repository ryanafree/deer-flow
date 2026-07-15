"""Requirement extraction (D9), the planning half of dr_core.plan.

Extracts the turn's explicit requirements from the LATEST real (non-hidden)
user question via a structured or-mid call (mirrors dr_core.verify.votes's
injectable module-level model-factory idiom: ``_get_plan_model`` is a
separate factory so tests can monkeypatch it without touching
``deerflow.models``). The MODEL authors requirement CONTENT (kind/text/
entities/window/must_cover); code mints the deterministic id and
pydantic-validates before the caller writes the ledger -- content generation
upstream of state, deterministic transport into it (the C3-permitted
pattern, D9).

Degrades, never invents, never raises: a model-call failure, unparseable
output, or an individual malformed item all drop that piece (log-and-degrade)
rather than fabricating a requirement or crashing the node.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from dr_core.models.derive import norm
from dr_core.models.enums import RequirementKind
from dr_core.models.ledger import Requirement

logger = logging.getLogger(__name__)

MAX_REQUIREMENTS_PER_TURN = 8

_INSTRUCTIONS = (
    "You are a research-planning assistant for a research-assurance pipeline. Read the user's "
    "research question and extract the DISTINCT, checkable requirements it implies -- the specific "
    "asks a complete answer must cover. Do not invent asks the question does not make.\n\n"
    "Respond with ONLY a JSON array (no prose, no markdown fences), each element shaped exactly:\n"
    '{"kind": "entity|comparison|metric|date_window|deliverable|subtopic", '
    '"text": "<the requirement, in your own words>", '
    '"entities": ["<optional entity/operand strings>"], '
    '"window": {"from": "<ISO date or null>", "to": "<ISO date or null>"} or null, '
    '"must_cover": <bool -- true iff the question cannot be answered without this>, '
    '"evidence_class": "academic|primary_data|practitioner|news|any -- the kind of evidence a complete '
    "answer needs (academic = peer-reviewed/working-paper literature, primary_data = official/structured "
    "data series or filings, practitioner = industry/vendor analysis, news = journalism; use any if the "
    "requirement doesn't call for a specific kind)\", "
    '"freshness": "foundational|frontier|any -- whether the canonical/older literature or the most '
    'current sources are what this requirement needs; use any if either works"}\n'
    f"Return at most {MAX_REQUIREMENTS_PER_TURN} requirements, ordered by importance."
)


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}


def _accumulate_usage(usage: dict[str, int], response) -> None:
    if response is None:
        return
    meta = getattr(response, "usage_metadata", None) or {}
    usage["input_tokens"] += int(meta.get("input_tokens") or 0)
    usage["output_tokens"] += int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    usage["cache_read_tokens"] += int(details.get("cache_read") or 0)
    usage["cache_creation_tokens"] += int(details.get("cache_creation") or 0)
    usage["calls"] += 1


def _get_plan_model():
    """Module-level injectable factory -- tests monkeypatch this name
    directly, mirroring ``dr_core.verify.votes._get_vote_model``."""
    from deerflow.models import create_chat_model

    return create_chat_model(os.environ.get("DR_PLAN_MODEL", "or-mid"))


def _requirement_id(text: str, kind: str) -> str:
    """Deterministic id: same (text, kind) always resolves to the same
    requirement, so re-extraction across turns/re-merges is idempotent."""
    return hashlib.sha256((norm(text) + kind).encode("utf-8")).hexdigest()[:16]


def _is_hidden(message: AnyMessage) -> bool:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return bool(isinstance(kwargs, dict) and kwargs.get("hide_from_ui") is True)


def latest_real_user_question(messages: list[AnyMessage]) -> str | None:
    """Walk ``messages`` backward for the last REAL (non-hidden) HumanMessage.

    Every system-injected human turn this codebase produces -- the D9
    ``<dr_corrective>`` bounce, goal continuations, dynamic-context
    reminders, view-image injections -- is marked ``hide_from_ui: True``
    (the pervasive convention; see ``runtime/goal.py::_is_visible_message``),
    so a single check skips all of them without naming each one.
    """
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if _is_hidden(message):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        content = content.strip()
        if content:
            return content
    return None


def _parse_json_array(text: str) -> list | None:
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
    except (TypeError, ValueError):
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except (TypeError, ValueError):
            return None
    return parsed if isinstance(parsed, list) else None


def _to_requirement(raw: object) -> Requirement | None:
    if not isinstance(raw, dict):
        return None
    try:
        kind = RequirementKind(raw.get("kind"))
    except ValueError:
        return None
    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    entities = raw.get("entities")
    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
        entities = []
    window = raw.get("window")
    if not isinstance(window, dict):
        window = None
    try:
        return Requirement(
            id=_requirement_id(text, kind.value),
            kind=kind,
            text=text,
            entities=entities,
            window=window,
            must_cover=bool(raw.get("must_cover", False)),
            # Requirement's field_validators coerce a missing/invalid value to
            # "any" (settled question 12 / test 5), so no validation is needed
            # here beyond a plain .get -- a non-str raw value coerces too since
            # the validator's membership check simply fails it into "any".
            evidence_class=raw.get("evidence_class"),
            freshness=raw.get("freshness"),
        )
    except ValidationError as exc:
        logger.warning("plan.extraction: dropping invalid requirement %r: %s", raw, exc)
        return None


async def extract_requirements(question: str) -> tuple[list[Requirement], dict[str, int]]:
    """Extract up to ``MAX_REQUIREMENTS_PER_TURN`` requirements from
    ``question`` via a structured or-mid call. Returns ``(requirements,
    usage)``. A model-call failure or a response that is not a parseable
    JSON array degrades to an EMPTY list (log-and-degrade); an individual
    malformed element is dropped and logged rather than discarding the
    whole batch."""
    usage = _empty_usage()
    try:
        model = _get_plan_model()
        response = await model.ainvoke(
            [
                SystemMessage(content=_INSTRUCTIONS),
                HumanMessage(content=f"Research question:\n{question}"),
            ]
        )
    except Exception as exc:
        logger.warning("plan.extraction: model call failed, degrading to no requirements: %s", exc)
        return [], usage

    _accumulate_usage(usage, response)
    content = response.content if isinstance(response.content, str) else str(response.content)
    parsed = _parse_json_array(content)
    if parsed is None:
        logger.warning("plan.extraction: unparseable model output, degrading to no requirements")
        return [], usage

    requirements: list[Requirement] = []
    seen_ids: set[str] = set()
    for raw in parsed:
        requirement = _to_requirement(raw)
        if requirement is None or requirement.id in seen_ids:
            continue
        seen_ids.add(requirement.id)
        requirements.append(requirement)
        if len(requirements) >= MAX_REQUIREMENTS_PER_TURN:
            break
    return requirements, usage

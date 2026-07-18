"""Coverage mapping (D9), the second half of dr_core.plan.

Maps every unmapped (ACTIVE requirement, claim) pair to a
``dr_core.models.ledger.CoverageMapping`` via a single structured or-mid
call, using the same injectable-factory idiom as ``extraction.py`` /
``dr_core.verify.votes``. Referential integrity is the hard backstop: a
result naming a requirement/claim id this pass never asked about is dropped
and logged, never written -- this single check also catches any id the
model might invent, since ``pair_set`` is built only from known ids.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from dr_core.connectors.registry import by_name, load_connectors
from dr_core.models.enums import CoverageRelation
from dr_core.models.ledger import Claim, CoverageMapping, Requirement
from dr_core.plan.evidence_class import resolve_evidence_class

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You are a coverage-mapping assistant for a research-assurance pipeline. You are given a list of "
    "REQUIREMENTS (explicit asks a complete answer must satisfy) and a list of CLAIMS (facts recorded "
    "so far). For EVERY (requirement, claim) pair listed under PAIRS, decide how much that claim "
    "contributes toward answering that requirement. A target_requirement_id is the extractor's candidate link, not proof; confirm it independently. "
    "Classify the relation according to the requirement kind:\n"
    "- For a subtopic requirement, use direct when the claim is a concrete supported finding squarely about that subtopic. "
    "One claim need not exhaust the whole subtopic; the coverage gate aggregates multiple direct findings.\n"
    "- For a comparison requirement, use direct only when the claim states a relationship between the compared entities, "
    "and set relationship_stated=true.\n"
    "- For a metric requirement, use direct only when the claim supplies the requested metric, date or period, and value.\n"
    "- For a date_window requirement, use direct only when the claim is dated within or explicitly addresses the requested window.\n"
    "- For an entity or deliverable requirement, use direct when the claim supplies a concrete supported finding that squarely "
    "advances the named entity or requested deliverable.\n"
    "Tangential context or generic topical overlap is partial; unrelated material is none. A claim's quote must support the claim, "
    "but the quote need not restate the broader requirement verbatim.\n\n"
    "Respond with ONLY a JSON array (no prose, no markdown fences), one element per pair, shaped exactly:\n"
    '{"requirement_id": "<id>", "claim_id": "<id>", "relation": "direct|partial|none", '
    '"elements_satisfied": ["<optional short strings naming which parts of the requirement this claim answers>"], '
    '"relationship_stated": <true|false|null -- for comparison requirements only: whether this claim states the '
    "relationship between the compared entities; null if not applicable>}\n"
    'Include an element for every pair, even when relation is "none".'
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
    directly, mirroring ``dr_core.verify.votes._get_vote_model`` /
    ``dr_core.plan.extraction._get_plan_model``. Model name resolution lives
    in ``dr_core.models.tiers`` (the sole owner of the plan-tier env knob)."""
    from deerflow.models import create_chat_model
    from dr_core.models import tiers

    return create_chat_model(tiers.plan_model_name())


def unmapped_pairs(
    active_requirement_ids: Sequence[str],
    claim_ids: Sequence[str],
    existing_coverage: Mapping[str, dict],
) -> list[tuple[str, str]]:
    """The cross product of ACTIVE requirements x claims, minus pairs already
    present in ``existing_coverage`` (keyed ``f"{requirement_id}:{claim_id}"``)."""
    pairs: list[tuple[str, str]] = []
    for req_id in active_requirement_ids:
        for claim_id in claim_ids:
            if f"{req_id}:{claim_id}" not in existing_coverage:
                pairs.append((req_id, claim_id))
    return pairs


def _format_requirement(requirement: Requirement) -> dict:
    return {"id": requirement.id, "kind": requirement.kind.value, "text": requirement.text, "entities": requirement.entities}


def _format_claim(claim: Claim, sources: Mapping[str, dict], connectors_by_name: Mapping[str, object]) -> dict:
    source = sources.get(claim.source_id) or {}
    return {
        "id": claim.claim_id,
        "text": claim.text,
        "supporting_quote": claim.support.quote if claim.support else None,
        "target_requirement_ids": claim.target_requirement_ids,
        "source_id": claim.source_id,
        "source_system": source.get("source_system"),
        "source_evidence_class": resolve_evidence_class(source.get("source_system", "web"), source.get("url_or_id"), connectors_by_name),
    }


def _build_prompt(
    requirements: Sequence[Requirement],
    claims: Sequence[Claim],
    pairs: Sequence[tuple[str, str]],
    sources: Mapping[str, dict],
) -> str:
    connectors_by_name = by_name(load_connectors())
    payload = {
        "requirements": [_format_requirement(r) for r in requirements],
        "claims": [_format_claim(c, sources, connectors_by_name) for c in claims],
        "pairs": [{"requirement_id": req_id, "claim_id": claim_id} for req_id, claim_id in pairs],
    }
    return json.dumps(payload, sort_keys=True)


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


async def map_coverage(
    active_requirements: Mapping[str, Requirement],
    claims: Mapping[str, Claim],
    existing_coverage: Mapping[str, dict],
    *,
    sources: Mapping[str, dict] | None = None,
) -> tuple[dict[str, dict], dict[str, int]]:
    """Map requirement-targeted pairs in bounded per-requirement calls.

    Claims that carry ``target_requirement_ids`` are evaluated only against
    those candidate requirements; claims recorded before a plan existed keep
    the legacy active-requirements cross product. Each requirement is sent in
    a separate model call so unrelated topics cannot drown out its evidence.
    Returns ``(new_coverage, usage)``, where
    ``new_coverage`` is keyed ``f"{requirement_id}:{claim_id}"``. A
    model-call failure or unparseable response degrades to no new mappings;
    an individual result naming an id outside this pass's own
    requirements/claims, or a pair not actually asked about, is dropped and
    logged (referential integrity) rather than written."""
    usage = _empty_usage()
    pairs = unmapped_pairs(list(active_requirements.keys()), list(claims.keys()), existing_coverage)
    pairs = [
        (req_id, claim_id)
        for req_id, claim_id in pairs
        if not claims[claim_id].target_requirement_ids or req_id in claims[claim_id].target_requirement_ids
    ]
    if not pairs:
        return {}, usage

    pair_set = set(pairs)
    result: dict[str, dict] = {}
    model = _get_plan_model()
    for requirement_id in active_requirements:
        batch_pairs = [pair for pair in pairs if pair[0] == requirement_id]
        if not batch_pairs:
            continue
        batch_claims = [claims[claim_id] for _, claim_id in batch_pairs]
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content=_INSTRUCTIONS),
                    HumanMessage(
                        content=_build_prompt(
                            [active_requirements[requirement_id]],
                            batch_claims,
                            batch_pairs,
                            sources or {},
                        )
                    ),
                ]
            )
        except Exception as exc:
            logger.warning("plan.mapping: model call failed for requirement %s, degrading that batch to no new mappings: %s", requirement_id, exc)
            continue

        _accumulate_usage(usage, response)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json_array(content)
        if parsed is None:
            logger.warning("plan.mapping: unparseable model output for requirement %s, degrading that batch to no new mappings", requirement_id)
            continue

        batch_pair_set = set(batch_pairs)
        for raw in parsed:
            if not isinstance(raw, dict):
                continue
            req_id = raw.get("requirement_id")
            claim_id = raw.get("claim_id")
            if (req_id, claim_id) not in batch_pair_set or (req_id, claim_id) not in pair_set:
                logger.warning("plan.mapping: dropping mapping for unrequested/unknown pair (%r, %r)", req_id, claim_id)
                continue
            try:
                relation = CoverageRelation(raw.get("relation"))
            except ValueError:
                logger.warning("plan.mapping: dropping mapping with invalid relation %r for pair (%s, %s)", raw.get("relation"), req_id, claim_id)
                continue
            elements = raw.get("elements_satisfied")
            if not isinstance(elements, list) or not all(isinstance(e, str) for e in elements):
                elements = []
            relationship_stated = raw.get("relationship_stated")
            if not isinstance(relationship_stated, bool):
                relationship_stated = None
            mapping = CoverageMapping(
                requirement_id=req_id,
                claim_id=claim_id,
                relation=relation,
                elements_satisfied=elements,
                relationship_stated=relationship_stated,
            )
            result[f"{req_id}:{claim_id}"] = mapping.model_dump(mode="json")
    return result, usage

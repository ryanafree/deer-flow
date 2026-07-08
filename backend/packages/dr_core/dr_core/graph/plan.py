"""plan_coverage_node -- the D9 outer-graph planning node (research ->
plan_coverage -> verify -> eligibility_gate).

Extraction is TURN-SCOPED: it runs only on the first pass of a turn
(``dr_run["gate_decision"] != "research"``), never on a corrective
re-entry, and writes ``dr_run["active_requirement_ids"]`` naming this
turn's requirements. A corrective re-entry leaves that key untouched by
simply never including it in the returned ``dr_run`` update -- ``merge_run``
is last-writer-wins per key, so an absent key is a no-op, not a clear.

Mapping runs EVERY pass (including corrective re-entries): unmapped claims x
this turn's ACTIVE requirements, via ``dr_core.plan.mapping.map_coverage``.
"""

from __future__ import annotations

from dr_core.models.ledger import Claim, Requirement
from dr_core.plan.extraction import extract_requirements, latest_real_user_question
from dr_core.plan.mapping import map_coverage


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}


def _merge_usage(totals: dict[str, int], delta: dict[str, int]) -> None:
    for key in totals:
        totals[key] += delta.get(key, 0)


async def plan_coverage_node(state) -> dict:
    dr_run: dict = state.get("dr_run") or {}
    dr_requirements: dict = state.get("dr_requirements") or {}
    dr_claims: dict = state.get("dr_claims") or {}
    dr_coverage: dict = state.get("dr_coverage") or {}

    usage = _empty_usage()
    new_requirements: dict[str, dict] = {}
    run_update: dict = {}

    if dr_run.get("gate_decision") != "research":
        # D9 turn scoping: only the first pass of a turn extracts. A
        # corrective re-entry re-enters with gate_decision == "research" and
        # skips this branch entirely, so active_requirement_ids below is
        # simply never overwritten for it.
        question = latest_real_user_question(state.get("messages") or [])
        active_ids: list[str] = []
        if question:
            requirements, extract_usage = await extract_requirements(question)
            _merge_usage(usage, extract_usage)
            new_requirements = {requirement.id: requirement.model_dump(mode="json") for requirement in requirements}
            active_ids = [requirement.id for requirement in requirements]
        run_update["active_requirement_ids"] = active_ids
        active_requirement_ids = active_ids
    else:
        active_requirement_ids = dr_run.get("active_requirement_ids") or []

    known_requirements = {**dr_requirements, **new_requirements}
    active_requirements: dict[str, Requirement] = {req_id: Requirement.model_validate(known_requirements[req_id]) for req_id in active_requirement_ids if req_id in known_requirements}
    claims: dict[str, Claim] = {claim_id: Claim.model_validate(payload) for claim_id, payload in dr_claims.items()}

    new_coverage, mapping_usage = await map_coverage(active_requirements, claims, dr_coverage)
    _merge_usage(usage, mapping_usage)

    run_update["plan_usage"] = usage
    result: dict = {"dr_run": run_update}
    if new_requirements:
        result["dr_requirements"] = new_requirements
    if new_coverage:
        result["dr_coverage"] = new_coverage
    return result

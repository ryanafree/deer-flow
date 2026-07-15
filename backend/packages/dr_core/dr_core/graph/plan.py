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

from langchain_core.messages import HumanMessage

from dr_core.models.ledger import Claim, Requirement
from dr_core.plan.extraction import extract_requirements, latest_real_user_question
from dr_core.plan.mapping import map_coverage
from dr_core.plan.schedule import format_schedule_lines, schedule_queries


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}


def _merge_usage(totals: dict[str, int], delta: dict[str, int]) -> None:
    for key in totals:
        totals[key] += delta.get(key, 0)


def _schedule_directive_message(lines: list[str]) -> HumanMessage:
    """Hidden HumanMessage carrying the entry research plan, mirroring the
    gate's corrective-message convention (hide_from_ui)."""
    content = "\n".join(
        [
            "<dr_research_plan>",
            "Work each scheduled item below. For every item, search with the named tool (or web_search if none is named), then record each supported fact via the record_claim tool, citing a source_id from the result.",
            *lines,
            "</dr_research_plan>",
        ]
    )
    return HumanMessage(content=content, additional_kwargs={"hide_from_ui": True, "deerflow_dr_schedule": True})


async def plan_requirements_node(state) -> dict:
    """D11 item 4, wiring (i): the deliverable-preset entry planning pass
    (initialize -> plan_requirements -> research). Extracts this turn's
    requirements BEFORE any retrieval and schedules a directed first research
    pass via the same ``schedule_queries`` implementation the gate's
    corrective retry uses (one implementation, two wirings)."""
    # Local import: gate hosts the process-wide connectors cache; gate does not
    # import this module, so this stays acyclic.
    from dr_core.graph.gate import _connectors_by_name

    dr_run: dict = state.get("dr_run") or {}
    usage = _empty_usage()

    question = dr_run.get("question") or latest_real_user_question(state.get("messages") or [])
    requirements: list[Requirement] = []
    if question:
        requirements, extract_usage = await extract_requirements(question)
        _merge_usage(usage, extract_usage)

    # entry_plan_usage, not plan_usage: plan_coverage_node writes plan_usage
    # every pass and merge_run is last-writer-wins per key.
    run_update: dict = {
        "active_requirement_ids": [requirement.id for requirement in requirements],
        "requirements_planned": True,
        "entry_plan_usage": usage,
    }
    result: dict = {"dr_run": run_update}
    if requirements:
        result["dr_requirements"] = {requirement.id: requirement.model_dump(mode="json") for requirement in requirements}
        scheduled = schedule_queries(requirements, dr_run.get("profile") or "general", _connectors_by_name())
        result["messages"] = [_schedule_directive_message(format_schedule_lines(scheduled))]
    return result


async def plan_coverage_node(state) -> dict:
    dr_run: dict = state.get("dr_run") or {}
    dr_requirements: dict = state.get("dr_requirements") or {}
    dr_claims: dict = state.get("dr_claims") or {}
    dr_coverage: dict = state.get("dr_coverage") or {}

    usage = _empty_usage()
    new_requirements: dict[str, dict] = {}
    run_update: dict = {}

    if dr_run.get("gate_decision") != "research" and not dr_run.get("requirements_planned"):
        # D9 turn scoping: only the first pass of a turn extracts. A
        # corrective re-entry re-enters with gate_decision == "research" and
        # skips this branch entirely, so active_requirement_ids below is
        # simply never overwritten for it. requirements_planned means the
        # D11 entry planning node already extracted this turn (deliverable-
        # preset path) -- re-extracting here would double-spend and could
        # reshuffle the turn's requirement scope mid-run.
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
    # D11 run scoping: never map prior-run claims onto this turn's
    # requirements (gate/render/verify apply the same baseline filter).
    baseline_claim_ids = set(dr_run.get("baseline_claim_ids") or [])
    claims: dict[str, Claim] = {claim_id: Claim.model_validate(payload) for claim_id, payload in dr_claims.items() if claim_id not in baseline_claim_ids}

    new_coverage, mapping_usage = await map_coverage(active_requirements, claims, dr_coverage)
    _merge_usage(usage, mapping_usage)

    run_update["plan_usage"] = usage
    result: dict = {"dr_run": run_update}
    if new_requirements:
        result["dr_requirements"] = new_requirements
    if new_coverage:
        result["dr_coverage"] = new_coverage
    return result

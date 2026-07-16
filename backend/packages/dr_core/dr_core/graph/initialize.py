"""initialize_node -- the D11 run-initialization / scoping entry node.

Runs at graph entry on EVERY invocation (deterministic, no LLM, so it never
drags an ordinary chat turn into planning machinery -- D7's rationale is
about model-driven planning, not state bookkeeping). It:

- mints ``research_run_id`` and records the turn's question;
- resolves the ``deliverable`` preset: an input-state ``dr_run["deliverable"]``
  (the benchmark runner's existing convention) or
  ``configurable["dr_deliverable"]`` (the Gateway/headless path -- closes the
  review's "Gateway invocation does not initialize research intent" gap
  without touching app/ code);
- snapshots ``baseline_claim_ids`` / ``baseline_source_ids`` so gate, verify,
  and render can scope this run's evidence to claims recorded THIS run (the
  historical-claim-contamination fix);
- clears every turn-scoped loop/bookkeeping key, so a later research turn on
  the same thread never inherits a spent retry budget, a stale breaker
  signature, a stale requirement scope, or a prior turn's stop_reason /
  citation ordinals.

``route_after_initialize`` then hosts the D11 entry decision: a preset
deliverable routes ``plan_requirements -> research`` (plan before retrieval);
otherwise the D7/D9 interactive topology is entered unchanged
(research-first, ``route_after_research`` gating on the turn's markers).
"""

from __future__ import annotations

import uuid
from datetime import date

from langchain_core.messages import ToolMessage

from dr_core.plan.extraction import latest_real_user_question

# Turn-scoped keys reset on every new graph invocation. deliverable/profile/
# depth are deliberately NOT in this list: deliverable is re-resolved below,
# profile and depth are run inputs the caller owns.
_TURN_SCOPED_RESETS: dict = {
    "gate_decision": None,
    "gate_retries": 0,
    "gate_ledger_sig": None,
    "active_requirement_ids": [],
    "requirements_planned": False,
    "stop_reason": None,
    "citation_ordinals": None,
    "requirements_covered": None,
    "requirements_must_cover": None,
    "must_cover_states": None,
    "first_pass_requirements_covered": None,
    "first_pass_requirements_must_cover": None,
    "first_pass_must_cover_states": None,
    "tool_call_count": 0,
    "tool_call_counts": {},
}


def initialize_node(state, config=None) -> dict:
    dr_run = state.get("dr_run") or {}
    configurable = ((config or {}).get("configurable") or {}) if isinstance(config, dict) else {}

    deliverable = bool(dr_run.get("deliverable")) or bool(configurable.get("dr_deliverable"))

    run_update = dict(_TURN_SCOPED_RESETS)
    run_update["research_run_id"] = uuid.uuid4().hex
    run_update["current_date"] = date.today().isoformat()
    run_update["deliverable"] = deliverable
    run_update["question"] = latest_real_user_question(state.get("messages") or [])
    run_update["baseline_claim_ids"] = sorted((state.get("dr_claims") or {}).keys())
    run_update["baseline_source_ids"] = sorted((state.get("dr_sources") or {}).keys())
    run_update["_counted_tool_call_ids"] = sorted(str(message.tool_call_id) for message in (state.get("messages") or []) if isinstance(message, ToolMessage) and message.tool_call_id)
    return {"dr_run": run_update}


def route_after_initialize(state) -> str:
    """D11 item 4: wherever ``deliverable`` is preset at entry, intent is known
    before any model turn, so planning MUST precede retrieval. Otherwise the
    interactive D7 path (research-first) is entered exactly as ruled."""
    dr_run = state.get("dr_run") or {}
    if dr_run.get("deliverable"):
        return "plan"
    return "research"

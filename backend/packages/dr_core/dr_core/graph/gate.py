"""Eligibility gate node -- real implementation (D6 rulings B, C, D, E).

Derives publication_status over dr_claims (pure functions, no LLM/network),
decides publish-readiness, and either freezes the citation-ordinal map for
render or bounces a corrective research turn. The cap+one-strike breaker
guarantee the research<->gate cycle terminates (D6 ruling C).
"""

from __future__ import annotations

import hashlib
import json

from langchain_core.messages import HumanMessage

from dr_core.models import Claim, PublicationStatus, StopReason
from dr_core.models.derive import derived_materiality, is_grounded, publication_status

RETRY_CAP = 2  # D6-C: N=2 retries (3 research passes total).


def _corrective_message(dr_claims: dict, excluded: list[tuple[str, str]]) -> HumanMessage:
    """Build the hidden corrective HumanMessage, mirroring
    ``runtime/goal.py::make_goal_continuation_message``'s hide_from_ui
    convention (goal.py:365-371): a HumanMessage marked invisible to the UI
    via ``additional_kwargs``, appended so re-entry into the research
    subgraph is a fresh model loop over the same messages channel.
    """
    if not dr_claims:
        gap = "no eligible claims recorded -- no claims have been recorded yet."
    else:
        examples = ", ".join(f"{claim_id} ({reason})" for claim_id, reason in excluded[:3])
        gap = f"no eligible claims recorded -- all {len(dr_claims)} recorded claim(s) are ineligible: {examples}."
    content = (
        "<dr_corrective>\n"
        f"Gap: {gap}\n"
        "Search for sources and record each supported fact via the record_claim tool, "
        "citing a source_id from a web_search/web_fetch result.\n"
        "</dr_corrective>"
    )
    return HumanMessage(
        content=content,
        additional_kwargs={
            "hide_from_ui": True,
            "deerflow_dr_corrective": True,
        },
    )


def eligibility_gate(state) -> dict:
    """Derive publish-readiness over ``dr_claims`` and route via ``dr_run``.

    ELIGIBLE = publication_status(claim, conflicts=()) != excluded AND the
    claim's cited source_id is present in dr_sources. Storage rule (D6-B):
    publication_status is DERIVED ONLY here for routing -- it is never
    written back onto a claim (render re-derives it identically). The
    citation-ordinal map and loop bookkeeping ARE stored, in dr_run.
    """
    dr_claims = state.get("dr_claims") or {}
    dr_sources = state.get("dr_sources") or {}
    dr_run = state.get("dr_run") or {}

    eligible_ids: list[str] = []
    excluded: list[tuple[str, str]] = []
    for claim_id, payload in dr_claims.items():
        claim = Claim.model_validate(payload)
        # E2 (D6 ruling E): requirement-coverage is out of scope this
        # sub-phase, so no mappings/requirements exist yet -- derived_materiality
        # caps the unmapped claim at MEDIUM, exactly as it does for a real run
        # before coverage mappings land.
        materiality = derived_materiality(claim, [], {})
        grounded = is_grounded(claim, materiality)
        # TODO(phase2): wire conflicts -- conflict detection is a later piece.
        status = publication_status(claim, materiality=materiality, grounded=grounded, conflicts=())
        source_ok = claim.source_id in dr_sources
        if status != PublicationStatus.EXCLUDED and source_ok:
            eligible_ids.append(claim_id)
        else:
            excluded.append((claim_id, status.value if source_ok else "missing_source"))

    retries = dr_run.get("gate_retries", 0)
    prev_sig = dr_run.get("gate_ledger_sig")

    sig = hashlib.sha256(json.dumps([sorted(eligible_ids), len(dr_claims), len(dr_sources)], sort_keys=True).encode()).hexdigest()
    breaker_fired = retries > 0 and prev_sig == sig
    retries_exhausted = retries >= RETRY_CAP

    if eligible_ids or retries_exhausted or breaker_fired:
        # PROCEED to render. Freeze the citation-ordinal map: insertion order
        # of dr_claims, 1-based, restricted to the eligible set.
        eligible_set = set(eligible_ids)
        citation_ordinals: dict[str, int] = {}
        for claim_id in dr_claims:
            if claim_id in eligible_set:
                citation_ordinals[claim_id] = len(citation_ordinals) + 1
        run_update = {
            "gate_decision": "render",
            "citation_ordinals": citation_ordinals,
            "gate_ledger_sig": sig,
        }
        if not eligible_ids:
            # Forced render with the only E2 gap: zero eligible claims.
            run_update["stop_reason"] = StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        return {"dr_run": run_update}

    # RETRY: eligible empty, retries remain, breaker not fired.
    corrective = _corrective_message(dr_claims, excluded)
    return {
        "messages": [corrective],
        "dr_run": {
            "gate_retries": retries + 1,
            "gate_ledger_sig": sig,
            "gate_decision": "research",
        },
    }

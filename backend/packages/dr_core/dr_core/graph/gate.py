"""Eligibility gate node -- real implementation (D6 rulings B, C, D, E; extended by
D9 for the requirement-coverage predicate).

Derives publication_status over dr_claims (pure functions, no LLM/network),
decides publish-readiness, and either freezes the citation-ordinal map for
render or bounces a corrective research turn. The cap+one-strike breaker
guarantee the research<->gate cycle terminates (D6 ruling C). On PROCEED the
gate also resets the turn-scoped loop bookkeeping (``deliverable``,
``gate_retries``, ``gate_ledger_sig``, ``active_requirement_ids``) per D7/D9,
so a later research turn on the same thread starts fresh instead of
inheriting a spent retry budget, a stale breaker signature, or a stale
requirement scope. The corrective (retry) branch leaves ``deliverable`` and
``active_requirement_ids`` untouched -- that is what keeps the mid-loop
re-entry gated (and requirement-scoped) even if the model records nothing
again.

D9 extends the PROCEED/RETRY predicate: a turn no longer proceeds just
because ELIGIBLE is non-empty -- any ACTIVE must-cover requirement still
UNCOVERED also forces a retry (subject to the same cap + one-strike
breaker). Materiality is now real: `derived_materiality` gets the claim's
actual coverage mappings and the full requirement set, so a claim directly
supporting a must-cover requirement derives HIGH materiality (D2 verbatim
port), tightening `is_grounded` exactly as state.js specified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from langchain_core.messages import HumanMessage

from dr_core.models import Claim, ClaimEvidence, StopReason, derived_materiality, evidence_state, is_grounded, publication_status
from dr_core.models.eligibility import ineligibility_reason
from dr_core.models.enums import PublicationStatus, RequirementState
from dr_core.models.ledger import CoverageMapping, Requirement

RETRY_CAP = 2  # D6-C: N=2 retries (3 research passes total).


def _corrective_message(dr_claims: dict, excluded: list[tuple[str, str]], eligible_ids: Sequence[str], uncovered_requirement_texts: Sequence[str] = ()) -> HumanMessage:
    """Build the hidden corrective HumanMessage, mirroring
    ``runtime/goal.py::make_goal_continuation_message``'s hide_from_ui
    convention (goal.py:365-371): a HumanMessage marked invisible to the UI
    via ``additional_kwargs``, appended so re-entry into the research
    subgraph is a fresh model loop over the same messages channel.

    D9: the eligibility gap (``eligible_ids`` empty) and the coverage gap
    (an uncovered must-cover requirement) are independent triggers -- either,
    both, or neither may be true on a given retry, so each gets its own
    paragraph rather than one conflated message.
    """
    lines = ["<dr_corrective>"]
    if not eligible_ids:
        if not dr_claims:
            gap = "no eligible claims recorded -- no claims have been recorded yet."
        else:
            examples = ", ".join(f"{claim_id} ({reason})" for claim_id, reason in excluded[:3])
            gap = f"no eligible claims recorded -- all {len(dr_claims)} recorded claim(s) are ineligible: {examples}."
        lines.append(f"Gap: {gap}")
    if uncovered_requirement_texts:
        req_lines = "\n".join(f"- {text}" for text in uncovered_requirement_texts)
        lines.append(f"The following required item(s) have no direct supporting claim yet:\n{req_lines}")
    lines.append("Search for sources and record each supported fact via the record_claim tool, citing a source_id from a web_search/web_fetch result.")
    lines.append("</dr_corrective>")
    return HumanMessage(
        content="\n".join(lines),
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

    D9: PROCEED additionally requires every ACTIVE must-cover requirement's
    `evidence_state` to be non-UNCOVERED (PARTIAL is acceptable; only
    UNCOVERED forces a retry). "ACTIVE" means named in
    `dr_run["active_requirement_ids"]` -- a stale prior-turn must-cover
    requirement never holds this turn's gate hostage.
    """
    dr_claims = state.get("dr_claims") or {}
    dr_sources = state.get("dr_sources") or {}
    dr_requirements = state.get("dr_requirements") or {}
    dr_coverage = state.get("dr_coverage") or {}
    dr_run = state.get("dr_run") or {}

    requirements_by_id: dict[str, Requirement] = {req_id: Requirement.model_validate(payload) for req_id, payload in dr_requirements.items()}
    coverage_mappings: list[CoverageMapping] = [CoverageMapping.model_validate(payload) for payload in dr_coverage.values()]
    mappings_by_req: dict[str, list[CoverageMapping]] = {}
    mappings_by_claim: dict[str, list[CoverageMapping]] = {}
    for mapping in coverage_mappings:
        mappings_by_req.setdefault(mapping.requirement_id, []).append(mapping)
        mappings_by_claim.setdefault(mapping.claim_id, []).append(mapping)

    eligible_ids: list[str] = []
    excluded: list[tuple[str, str]] = []
    claims_by_id: dict[str, ClaimEvidence] = {}
    for claim_id, payload in dr_claims.items():
        claim = Claim.model_validate(payload)
        mappings_for_claim = mappings_by_claim.get(claim_id, ())
        # Derivation lives in dr_core.models.eligibility (shared with the
        # render node so gate and render can never disagree; D6-B consistency
        # requirement), now fed the REAL mappings/requirements (D9).
        reason = ineligibility_reason(claim, dr_sources, mappings_for_claim, requirements_by_id)
        if reason is None:
            eligible_ids.append(claim_id)
        else:
            excluded.append((claim_id, reason))

        # Non-excluded claims feed derive.evidence_state's claims_by_id (D9):
        # a claim's materiality/groundedness there must match its eligibility
        # derivation above -- one shared formula, two consumers.
        materiality = derived_materiality(claim, mappings_for_claim, requirements_by_id)
        status = publication_status(claim, materiality=materiality, grounded=is_grounded(claim, materiality), conflicts=())
        if status == PublicationStatus.EXCLUDED:
            continue
        source = dr_sources.get(claim.source_id) or {}
        claims_by_id[claim_id] = ClaimEvidence(
            grounded=is_grounded(claim, materiality),
            source_id=claim.source_id,
            publication_date=source.get("publication_date"),
        )

    active_ids: list[str] = dr_run.get("active_requirement_ids") or []
    must_cover_states: list[tuple[str, RequirementState]] = []
    for req_id in sorted(active_ids):
        requirement = requirements_by_id.get(req_id)
        if requirement is None or not requirement.must_cover:
            continue
        state_ = evidence_state(requirement, mappings_by_req.get(req_id, ()), claims_by_id)
        must_cover_states.append((req_id, state_))

    any_uncovered = any(state_ == RequirementState.UNCOVERED for _, state_ in must_cover_states)
    uncovered_texts = [requirements_by_id[req_id].text for req_id, state_ in must_cover_states if state_ == RequirementState.UNCOVERED]

    retries = dr_run.get("gate_retries", 0)
    prev_sig = dr_run.get("gate_ledger_sig")

    # D9: the breaker signature includes every ACTIVE must-cover requirement's
    # current evidence_state -- a retry that adds coverage without adding
    # claims is progress the signature must see, or the breaker would fire on
    # a converging loop.
    sig_source = [sorted(eligible_ids), len(dr_claims), len(dr_sources), sorted((req_id, state_.value) for req_id, state_ in must_cover_states)]
    sig = hashlib.sha256(json.dumps(sig_source, sort_keys=True).encode()).hexdigest()
    breaker_fired = retries > 0 and prev_sig == sig
    retries_exhausted = retries >= RETRY_CAP

    ready = bool(eligible_ids) and not any_uncovered

    if ready or retries_exhausted or breaker_fired:
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
            # D7/D9 reset: clears turn-scoped loop bookkeeping on PROCEED so a
            # later research turn on this thread starts fresh.
            "deliverable": False,
            "gate_retries": 0,
            "gate_ledger_sig": None,
            "active_requirement_ids": [],
        }
        if must_cover_states:
            covered_count = sum(1 for _, state_ in must_cover_states if state_ == RequirementState.COVERED)
            run_update["requirements_covered"] = covered_count
            run_update["requirements_must_cover"] = len(must_cover_states)
            # Frozen (not recomputed at render time) so render's Unsubstantiated
            # section names exactly the ACTIVE must-covers this gate decision
            # evaluated, not whatever the ledger looks like whenever render runs.
            run_update["must_cover_states"] = {req_id: state_.value for req_id, state_ in must_cover_states}
        else:
            covered_count = 0
        if not eligible_ids or (must_cover_states and covered_count < len(must_cover_states)):
            # Forced render with an E2/D9 gap: zero eligible claims, or an
            # active must-cover requirement short of full coverage.
            run_update["stop_reason"] = StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        return {"dr_run": run_update}

    # RETRY: not ready, retries remain, breaker not fired.
    corrective = _corrective_message(dr_claims, excluded, eligible_ids, uncovered_texts)
    return {
        "messages": [corrective],
        "dr_run": {
            "gate_retries": retries + 1,
            "gate_ledger_sig": sig,
            "gate_decision": "research",
        },
    }

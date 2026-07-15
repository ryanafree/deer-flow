"""Pure functions ported from harness/state.js (the readable spec) and the matching
harness/dr.js runtime block, per the D2 ruling (build-logs/fable-consult-02-RULING.md).

PURE by contract, same as the source: no I/O, no clock, no randomness, no mutation of
arguments. Every input is passed in; every output is a plain value or a new model
instance. This is the domain layer only — no LangGraph, no channels (ruling section B
is Phase 2).

Scope note (for the orchestrator): this ports every function state.js exports THAT
evals/state_test.js exercises, EXCEPT the claim-selection / rate-limit-hardening
helpers (rankClaimForProtection, selectProtectedClaimIds, capPendingClaims) and
normalizeRequirements. Those operate over the engine's loosely-typed claim pool during
a live run (scheduling/selection over ad hoc dicts with fields like relation_effective,
verification_status='not-verified') rather than the pydantic ledger schema the D2
ruling's section A actually defines, and are not named in the worker contract's SOURCE
list or ruling section A. Flagged for orchestrator review rather than silently ported
into a shape that would fight the clean schema.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dr_core.models.enums import (
    CitationStatus,
    ConflictOutcome,
    CoverageRelation,
    DataProvenance,
    GateFlag,
    Materiality,
    PublicationStatus,
    RequirementKind,
    RequirementState,
    StopReason,
    SupportRelation,
    VerificationStatus,
)
from dr_core.models.ledger import Claim, Conflict, CoverageMapping, Requirement, Vote

# ---------------------------------------------------------------------------
# Constants (ported verbatim from state.js / dr.js)
# ---------------------------------------------------------------------------

MATERIALITY_RANK: dict[Materiality, int] = {Materiality.LOW: 0, Materiality.MEDIUM: 1, Materiality.HIGH: 2}
_RANK_TO_MATERIALITY: dict[int, Materiality] = {0: Materiality.LOW, 1: Materiality.MEDIUM, 2: Materiality.HIGH}

# Support-relation precedence (P0-4). Higher = more supportive.
RELATION_STRENGTH: dict[SupportRelation, int] = {
    SupportRelation.SUPPORTS_DIRECTLY: 4,
    SupportRelation.SUPPORTS_INFERENTIALLY: 3,
    SupportRelation.QUALIFIES: 2,
    SupportRelation.CONTEXT_ONLY: 1,
    # none / null / 'unreviewed' / contradicts / stale -> 0 (dict miss)
}
GROUNDING_MIN_STRENGTH = 2  # qualifies and up establish a claim

UNREVIEWED = "unreviewed"  # ruling divergence 3: non-grounding sentinel, not a SupportRelation member

# Per-profile gate flags that bar a claim from counting as grounded support (dr.js
# GROUNDING_BLOCKING_FLAGS — all six, not just the financial three).
GROUNDING_BLOCKING_FLAGS: set[GateFlag] = {
    GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE,
    GateFlag.PERIOD_MISMATCH,
    GateFlag.MISSING_VINTAGE,
    GateFlag.VENDOR_REPORTED,
    GateFlag.UNREPRODUCED_PAPER_CLAIM,
    GateFlag.OVERSTATED_EVIDENCE,
}

REFUTATIONS_REQUIRED = 2  # dr.js:106


def norm(s: str | None) -> str:
    """Mirrors state.js norm: lowercase, collapse whitespace, trim."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ---------------------------------------------------------------------------
# Materiality
# ---------------------------------------------------------------------------


def materiality_of(importance: int | None) -> Materiality:
    """Mirrors state.js materialityOf."""
    i = importance or 0
    if i >= 4:
        return Materiality.HIGH
    if i == 3:
        return Materiality.MEDIUM
    return Materiality.LOW


def cap_materiality(level: Materiality, max_level: Materiality) -> Materiality:
    lr = MATERIALITY_RANK.get(level, 0)
    mr = MATERIALITY_RANK.get(max_level, 2)
    return _RANK_TO_MATERIALITY[min(lr, mr)]


def derived_materiality(
    claim: Claim,
    mappings_for_claim: Sequence[CoverageMapping],
    requirements_by_id: Mapping[str, Requirement],
) -> Materiality:
    """Ports state.js derivedMateriality. Uses claim.importance (the ORIGINAL
    extractor rating) directly — never a previously-derived materiality — so
    re-running this each pass cannot compound. dr_core's Claim has no materiality
    field at all (it is derived-only), so non-compounding is structural here, not
    just a convention."""
    maps = mappings_for_claim or []
    direct_must_cover = any(m.relation == CoverageRelation.DIRECT and (req := requirements_by_id.get(m.requirement_id)) is not None and req.must_cover for m in maps)
    if direct_must_cover:
        return Materiality.HIGH
    extractor = materiality_of(claim.importance)
    if not maps:
        return cap_materiality(extractor, Materiality.MEDIUM)
    return extractor


def materiality_uncalibrated(materialities: Sequence[Materiality], threshold: float = 0.9) -> bool:
    """Calibration backstop: if almost everything is high, materiality is doing no
    rationing work. Takes already-derived materiality values directly (a signature
    simplification vs. state.js's `claims` list — same semantics, no claim shape
    needed since the caller already has the values)."""
    if not materialities:
        return False
    high = sum(1 for m in materialities if m == Materiality.HIGH)
    return high / len(materialities) > threshold


# ---------------------------------------------------------------------------
# P0-4: conservative effective relation + structured/quote grounding paths
# ---------------------------------------------------------------------------


def relation_strength(rel: SupportRelation | str | None) -> int:
    return RELATION_STRENGTH.get(rel, 0)  # type: ignore[arg-type]


def is_grounding_relation(rel: SupportRelation | str | None) -> bool:
    return relation_strength(rel) >= GROUNDING_MIN_STRENGTH


def effective_relation(
    extractor_rel: SupportRelation | str,
    reviewer_rel: SupportRelation | str | None,
    materiality: Materiality,
) -> str:
    """Ports state.js effectiveRelation EXACTLY (trap b: ~12 load-bearing tests).

    UNSTRUCTURED (quote-bearing) claims ONLY — data_ref/structured claims must never
    be routed through here (trap e; see structured_grounded/is_grounded). The
    extractor self-labels support; an independent reviewer re-labels from
    (claim, quote) only. On disagreement the WEAKER label wins. A missing reviewer
    (None — ruling divergence 3: "unavailable" is None) must not silently inherit the
    extractor's label for a high-materiality claim: it returns "unreviewed" (strength
    0, does not ground). contradicts/stale pass through unchanged.
    """
    if extractor_rel == SupportRelation.CONTRADICTS or extractor_rel == SupportRelation.STALE:
        return extractor_rel
    if reviewer_rel == SupportRelation.CONTRADICTS:
        return SupportRelation.CONTRADICTS
    if reviewer_rel is None:
        return UNREVIEWED if materiality == Materiality.HIGH else (extractor_rel or UNREVIEWED)
    return reviewer_rel if relation_strength(reviewer_rel) <= relation_strength(extractor_rel) else extractor_rel


def structured_grounded(claim: Claim, materiality: Materiality) -> bool:
    """Ports state.js structuredGrounded. Structured (data_ref) claims are grounded
    by the structured-data audit, not the quote/relation path: the retrieved value
    must match the claim, with no blocking gate flag set."""
    if not claim.data_ref:
        return False
    flags = claim.gate_flags
    if GateFlag.MISSING_VINTAGE in flags or GateFlag.PERIOD_MISMATCH in flags or GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE in flags:
        return False
    if claim.data_provenance == DataProvenance.MISMATCH:
        return False
    if materiality == Materiality.HIGH:
        return claim.data_provenance == DataProvenance.MATCHED
    return claim.data_provenance != DataProvenance.MISMATCH  # medium/low: provisional unless audited as mismatch


def is_grounded(claim: Claim, materiality: Materiality, *, post_verify: bool = False) -> bool:
    """Ports state.js/dr.js isGrounded. Branch-first on data_ref (trap e): structured
    and unstructured claims ground on SEPARATE paths, checked only after the shared
    gate-flag and citation/verify gates that apply to both."""
    flag_ok = not any(f in GROUNDING_BLOCKING_FLAGS for f in claim.gate_flags)
    verify_ok = claim.citation_status != CitationStatus.NOT_FOUND and (not post_verify or claim.verification.status != VerificationStatus.KILLED_ON_REFUTE)
    if not flag_ok or not verify_ok:
        return False

    if claim.data_ref:
        return structured_grounded(claim, materiality)

    if not claim.support or not claim.support.quote.strip():
        return False
    rel = claim.support.effective_relation(materiality)
    if not is_grounding_relation(rel):
        return False
    ex_rel = claim.support.relation_extractor
    if ex_rel == SupportRelation.SUPPORTS_INFERENTIALLY and not (claim.support.inference_note and claim.support.inference_note.strip()):
        return False
    if ex_rel == SupportRelation.QUALIFIES and not (claim.support.qualifier and claim.support.qualifier.strip()):
        return False
    if materiality == Materiality.HIGH and len(claim.support.quote.strip().split()) < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# P0-3: kind-specific coverage rules over grounded mapped claims
# ---------------------------------------------------------------------------


@dataclass
class ClaimEvidence:
    """Per-claim facts evidence_state needs beyond the mapping itself: whether it
    counts as grounded in the CURRENT set, its source (subtopic diversity), its
    publication date (date_window freshness), and its resolved evidence class
    (SPEC_evidence_routing_2026-07-10.md Stage 2 -- resolved by the caller via
    dr_core.plan.evidence_class.resolve_evidence_class, never computed here).
    Mirrors state.js's claimsById shape ({source_ref, publication_date, grounded})."""

    grounded: bool = True
    source_id: str | None = None
    publication_date: str | None = None
    evidence_class: str = "any"


def evidence_state(
    requirement: Requirement,
    mappings_for_req: Sequence[CoverageMapping],
    claims_by_id: Mapping[str, ClaimEvidence],
    *,
    min_recent_claims: int = 1,
    subtopic_min_claims: int = 2,
    subtopic_min_sources: int = 2,
) -> RequirementState:
    """Ports state.js evidenceState. mappings_for_req is assumed pre-filtered to the
    current claim set by the caller; this additionally drops claims absent from
    claims_by_id or explicitly not grounded, matching the JS defensive filter."""
    maps = [m for m in mappings_for_req if (ev := claims_by_id.get(m.claim_id)) is not None and ev.grounded is not False]
    directs = [m for m in maps if m.relation == CoverageRelation.DIRECT]
    any_evidence = any(m.relation in (CoverageRelation.DIRECT, CoverageRelation.PARTIAL) for m in maps)

    # Evidence-class gate (SPEC_evidence_routing_2026-07-10.md Stage 2, test 7):
    # a requirement with a specific evidence_class only counts DIRECT mappings
    # whose claim resolved to that SAME class toward COVERED; wrong-class
    # DIRECT evidence still counts toward any_evidence/PARTIAL (settled
    # question 6 -- weak evidence, not zero evidence). requirement.evidence_class
    # == "any" is a no-op filter, so that path is byte-for-byte the pre-Stage-2
    # behavior (regression requirement, test 7).
    if requirement.evidence_class == "any":
        class_directs = directs
    else:
        class_directs = [m for m in directs if claims_by_id.get(m.claim_id, ClaimEvidence()).evidence_class == requirement.evidence_class]

    def partial_or() -> RequirementState:
        return RequirementState.PARTIAL if any_evidence else RequirementState.UNCOVERED

    if requirement.kind == RequirementKind.DELIVERABLE:
        # Output-contract requirement: never satisfiable from research claims.
        return RequirementState.DEFERRED

    if requirement.kind == RequirementKind.COMPARISON:
        operands = [norm(e) for e in (requirement.entities or []) if e]

        def relationship_covered(m: CoverageMapping) -> bool:
            if m.relationship_stated is not None:
                return m.relationship_stated
            els = [norm(e) for e in m.elements_satisfied]
            if not operands:
                return True
            return all(any(o in e or e in o for e in els) for o in operands)

        covered = any(relationship_covered(m) for m in class_directs)
        return RequirementState.COVERED if covered else partial_or()

    if requirement.kind == RequirementKind.METRIC:
        covered = any(len(m.elements_satisfied) > 0 for m in class_directs)
        return RequirementState.COVERED if covered else partial_or()

    if requirement.kind == RequirementKind.DATE_WINDOW:
        window = requirement.window or {}
        w_from = window.get("from")
        if not w_from:
            # No parsed window: freshness is unverifiable. Do not hang a must-cover
            # requirement open forever; treat direct evidence as covered.
            return RequirementState.COVERED if class_directs else partial_or()
        w_to = window.get("to")
        in_window = [m for m in class_directs if (d := claims_by_id.get(m.claim_id, ClaimEvidence()).publication_date) and d >= w_from and (not w_to or d <= w_to)]
        if len(in_window) >= min_recent_claims:
            return RequirementState.COVERED
        return RequirementState.PARTIAL if class_directs else partial_or()

    if requirement.kind == RequirementKind.SUBTOPIC:
        srcs = {claims_by_id.get(m.claim_id, ClaimEvidence()).source_id for m in class_directs}
        srcs.discard(None)
        if len(class_directs) >= subtopic_min_claims and len(srcs) >= subtopic_min_sources:
            return RequirementState.COVERED
        return partial_or()

    # entity (and default): a DIRECT claim answering the proposition about the entity.
    return RequirementState.COVERED if len(class_directs) >= 1 else partial_or()


def terminal_state(
    requirement: Requirement,
    ev_state: RequirementState,
    *,
    scheduled_at_least_once: bool = False,
    run_ended: bool = False,
    budget_ended: bool = False,
) -> RequirementState:
    """Ports state.js terminalState: maps a live evidence state to a terminal label
    when the loop ends, given how much work the requirement actually received."""
    if ev_state == RequirementState.COVERED:
        return RequirementState.COVERED
    if requirement.kind == RequirementKind.DELIVERABLE:
        return RequirementState.DEFERRED  # resolved by close_deliverable_on_synth, not here
    if not scheduled_at_least_once:
        return RequirementState.NOT_ATTEMPTED
    if budget_ended:
        return RequirementState.BLOCKED_BUDGET
    if run_ended:
        return RequirementState.PARTIAL if ev_state == RequirementState.PARTIAL else RequirementState.SEARCH_EXHAUSTED
    return ev_state  # still running


def close_deliverable_on_synth(requirement: Requirement, report_ok: bool) -> RequirementState | None:
    """Ports state.js closeDeliverablesOnSynth for a SINGLE requirement (pure — the
    caller assigns the result; trap a bars in-place mutation). Deliverables rest at
    'deferred' during research; a non-placeholder report closes them, once."""
    if requirement.kind == RequirementKind.DELIVERABLE and requirement.terminal_state == RequirementState.DEFERRED and report_ok:
        return RequirementState.COVERED
    return requirement.terminal_state


def reopens_requirement(killed_claim_id: str, mappings_for_req: Sequence[CoverageMapping]) -> bool:
    """Ports state.js reopensRequirement: a verifier kill reopens a requirement only
    if the killed claim was its ONLY direct support."""
    directs = [m for m in mappings_for_req if m.relation == CoverageRelation.DIRECT]
    was_direct = any(m.claim_id == killed_claim_id for m in directs)
    remaining = [m for m in directs if m.claim_id != killed_claim_id]
    return was_direct and not remaining


# ---------------------------------------------------------------------------
# Attempt accounting + conflict outcomes
# ---------------------------------------------------------------------------


@dataclass
class WorkItem:
    scheduled: bool = False
    executed: bool = False


def counts_as_attempt(work_item: WorkItem | None) -> bool:
    """Ports state.js countsAsAttempt: an attempt counts only when the requirement's
    query was actually SCHEDULED and EXECUTED. A pass merely elapsing is not one."""
    return bool(work_item and work_item.scheduled and work_item.executed)


def normalize_conflict_outcome(verdict: str | None) -> ConflictOutcome:
    try:
        return ConflictOutcome(verdict)
    except ValueError:
        return ConflictOutcome.UNRESOLVED_PERSISTENT


def conflict_blocks_convergence(outcome: ConflictOutcome | None) -> bool:
    return outcome is None or outcome == ConflictOutcome.UNRESOLVED_PERSISTENT


# ---------------------------------------------------------------------------
# Requirement-aware stop classification
# ---------------------------------------------------------------------------


@dataclass
class StopBase:
    """The engine's raw shouldStop() result: reason is one of
    converged|iter-cap|budget|diminishing|dead-end|continue — a different, upstream
    vocabulary from StopReason, which is why it stays a plain str rather than an
    enum member."""

    stop: bool
    reason: str


@dataclass
class StopClassification:
    stop: bool
    reason: StopReason


def classify_stop(
    base: StopBase,
    *,
    open_must_cover: int = 0,
    min_attempts_met: bool = True,
) -> StopClassification:
    """Ports state.js classifyStop: refines the engine's base stop signal using
    requirement progress. Only StopReason.CONVERGED is success; a stop forced by a
    cap while must-cover work remains is named honestly."""
    if not base.stop:
        return StopClassification(False, StopReason.CONTINUE)

    if base.reason == "converged":
        return StopClassification(
            True,
            StopReason.CONVERGED if open_must_cover == 0 else StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS,
        )
    if base.reason == "budget":
        return StopClassification(True, StopReason.BUDGET_EXHAUSTED)
    if open_must_cover == 0:
        return StopClassification(True, StopReason.CONVERGED)
    if base.reason == "iter-cap":
        return StopClassification(True, StopReason.ITERATION_CAP)
    if not min_attempts_met:
        return StopClassification(False, StopReason.CONTINUE)
    return StopClassification(True, StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS)


def is_success_stop(reason: StopReason) -> bool:
    return reason == StopReason.CONVERGED


# ---------------------------------------------------------------------------
# Verification transitions (REFUTATIONS_REQUIRED = 2)
# ---------------------------------------------------------------------------


def is_single_clear(vote: Vote, risk_reasons: Sequence[str]) -> bool:
    """First-vote fast path (dr.js verifyClaimIds): clean vote, no risk reasons."""
    return not vote.refuted and not vote.abstain and vote.confidence == "high" and len(risk_reasons) == 0


def aggregate_verification_status(votes: Sequence[Vote]) -> VerificationStatus:
    """Ports dr.js aggregateVerificationStatus: 3-complete-vote aggregation, narrowed
    per D11 item 1 (DECISIONS.md): SUPPORTED requires at least one affirmative
    (non-refuted, non-abstain) vote among the valid votes. A zero-affirmative mixed
    pattern below the refute kill threshold (e.g. one refute plus abstains) now
    aggregates to NOT_VERIFIED instead of the dr.js verbatim else-branch SUPPORTED.
    Intentional fork divergence -- see FORK_DELTA.md."""
    valid = [v for v in votes if v is not None]
    refutes = sum(1 for v in valid if v.refuted)
    abstains = sum(1 for v in valid if v.abstain)
    if refutes >= REFUTATIONS_REQUIRED:
        return VerificationStatus.KILLED_ON_REFUTE
    if valid and abstains == len(valid):
        return VerificationStatus.NOT_VERIFIED
    if valid and not any(not v.refuted and not v.abstain for v in valid):
        return VerificationStatus.NOT_VERIFIED
    return VerificationStatus.SUPPORTED


def assert_verification_transition_allowed(current: VerificationStatus, new: VerificationStatus) -> None:
    """Trap (c): the repair pass must never reset a completed verification record
    (dr.js:2687 guards resume). killed_on_refute is terminal — nothing may leave it,
    not even re-selection for re-verify. supported/not_verified -> pending (re-select)
    is legal and NOT guarded here."""
    if current == VerificationStatus.KILLED_ON_REFUTE and new != VerificationStatus.KILLED_ON_REFUTE:
        raise ValueError(f"illegal transition: killed_on_refute is terminal, cannot move to {new}")


# ---------------------------------------------------------------------------
# Citation / data-provenance transitions (D2: advance-only ladders, sibling to
# assert_verification_transition_allowed above)
# ---------------------------------------------------------------------------

CITATION_TRANSITIONS: dict[CitationStatus, frozenset[CitationStatus]] = {
    CitationStatus.UNRESOLVED: frozenset({CitationStatus.UNRESOLVED, CitationStatus.RESOLVED, CitationStatus.NOT_FOUND}),
    CitationStatus.RESOLVED: frozenset({CitationStatus.RESOLVED}),
    CitationStatus.NOT_FOUND: frozenset({CitationStatus.NOT_FOUND}),
}


def assert_citation_transition_allowed(current: CitationStatus, new: CitationStatus) -> None:
    """CitationStatus is an advance-only ladder (D2): unresolved -> resolved and
    unresolved -> not_found are the only legal advances. resolved and not_found are
    terminal -- nothing may leave either. A same-state re-application is always a
    no-op (present in the allowed set for every current value)."""
    if new not in CITATION_TRANSITIONS[current]:
        raise ValueError(f"illegal transition: citation_status cannot move from {current} to {new}")


PROVENANCE_TRANSITIONS: dict[DataProvenance, frozenset[DataProvenance]] = {
    DataProvenance.UNAUDITED: frozenset({DataProvenance.UNAUDITED, DataProvenance.MATCHED, DataProvenance.MISMATCH}),
    DataProvenance.MATCHED: frozenset({DataProvenance.MATCHED}),
    DataProvenance.MISMATCH: frozenset({DataProvenance.MISMATCH}),
}


def assert_provenance_transition_allowed(current: DataProvenance, new: DataProvenance) -> None:
    """DataProvenance is an advance-only ladder (D2): unaudited -> matched and
    unaudited -> mismatch are the only legal advances. matched and mismatch are
    terminal audit outcomes -- nothing may leave either. A same-state re-application
    is always a no-op (present in the allowed set for every current value)."""
    if new not in PROVENANCE_TRANSITIONS[current]:
        raise ValueError(f"illegal transition: data_provenance cannot move from {current} to {new}")


# ---------------------------------------------------------------------------
# PublicationStatus derivation (fixed precedence) + the claimCaveat table
# ---------------------------------------------------------------------------


def claim_caveat(claim: Claim) -> str | None:
    """Ports dr.js claimCaveat VERBATIM (the fixed GateFlag -> string table).
    Order is significant — first matching flag wins, same as the JS if-chain."""
    flags = claim.gate_flags
    if GateFlag.VENDOR_REPORTED in flags:
        return "vendor-reported and not independently reproduced"
    if GateFlag.UNREPRODUCED_PAPER_CLAIM in flags:
        return "preprint benchmark without released code"
    if GateFlag.OVERSTATED_EVIDENCE in flags:
        return "the source type does not support the claim strength"
    if GateFlag.PERIOD_MISMATCH in flags:
        return "the source period does not match the claimed period"
    if GateFlag.MISSING_VINTAGE in flags:
        return "the required source vintage was unavailable"
    if GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE in flags:
        return "the number lacks a primary-source trace"
    return None


def publication_status(
    claim: Claim,
    *,
    materiality: Materiality,
    grounded: bool,
    conflicts: Sequence[Conflict] = (),
) -> PublicationStatus:
    """FIXED PRECEDENCE, pure function of (Claim, conflicts) — ruling section A:
    1. killed_on_refute OR citation not_found OR data_provenance mismatch -> excluded.
    2. any gate_flag OR member of an unresolved (or None-outcome) conflict -> contested
       (caveat = claim_caveat(claim), the fixed table above).
    3. verification supported AND grounded -> supported.
    4. else -> not_verified.
    context_only/unreviewed effective relations never ground (is_grounded/
    relation_strength), so such claims can never reach 'supported' via step 3.
    """
    if claim.verification.status == VerificationStatus.KILLED_ON_REFUTE or claim.citation_status == CitationStatus.NOT_FOUND or claim.data_provenance == DataProvenance.MISMATCH:
        return PublicationStatus.EXCLUDED

    in_unresolved_conflict = any(claim.claim_id in c.claim_ids and (c.outcome is None or c.outcome == ConflictOutcome.UNRESOLVED_PERSISTENT) for c in conflicts)
    if claim.gate_flags or in_unresolved_conflict:
        return PublicationStatus.CONTESTED

    if claim.verification.status == VerificationStatus.SUPPORTED and grounded:
        return PublicationStatus.SUPPORTED

    return PublicationStatus.NOT_VERIFIED

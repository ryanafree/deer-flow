"""Enum port of the DeepResearch harness's state machine (harness/state.js) plus the
schema enums from the D2 ruling (build-logs/fable-consult-02-RULING.md, section A).

StrEnum everywhere: a stored value round-trips through JSON/pydantic as a plain string
(channel serialization is model_dump(mode="json") per ruling section B) while a typo is
still catchable in Python, mirroring state.js's frozen-object enums. PublicationStatus
is DERIVED ONLY (ruling divergence 1: eligibility is derived, never a mutable status
field) and must never become a stored field on any model.
"""

from __future__ import annotations

from enum import StrEnum


class Materiality(StrEnum):
    """state.js MATERIALITY. Not one of the ruling's "11 enums" (it is never a stored
    ledger field) but required as the parameter/return type of materiality_of,
    derived_materiality, effective_relation, and structured_grounded — kept as a
    StrEnum for the same typo-safety reason as the rest."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RequirementKind(StrEnum):
    ENTITY = "entity"
    COMPARISON = "comparison"
    METRIC = "metric"
    DATE_WINDOW = "date_window"
    DELIVERABLE = "deliverable"
    SUBTOPIC = "subtopic"


class RequirementState(StrEnum):
    """First three are LIVE (recomputed by derive.evidence_state); the rest are
    terminal labels stamped once by derive.terminal_state when the run ends."""

    UNCOVERED = "uncovered"
    PARTIAL = "partial"
    COVERED = "covered"
    DEFERRED = "deferred"
    SEARCH_EXHAUSTED = "search_exhausted"
    BLOCKED_BUDGET = "blocked_budget"
    NOT_ATTEMPTED = "not_attempted"


class CoverageRelation(StrEnum):
    DIRECT = "direct"
    PARTIAL = "partial"
    NONE = "none"


class SupportRelation(StrEnum):
    """Extractor/reviewer self-labels only. "unreviewed" is a computed EFFECTIVE
    value (ruling divergence 3: reviewer "unavailable" is None; "unreviewed" exists
    only as derive.effective_relation's return sentinel) and is deliberately NOT a
    member here, keeping this enum clean."""

    SUPPORTS_DIRECTLY = "supports_directly"
    SUPPORTS_INFERENTIALLY = "supports_inferentially"
    QUALIFIES = "qualifies"
    CONTEXT_ONLY = "context_only"
    CONTRADICTS = "contradicts"
    STALE = "stale"


class GateFlag(StrEnum):
    VENDOR_REPORTED = "vendor_reported"
    UNREPRODUCED_PAPER_CLAIM = "unreproduced_paper_claim"
    OVERSTATED_EVIDENCE = "overstated_evidence"
    PERIOD_MISMATCH = "period_mismatch"
    MISSING_VINTAGE = "missing_vintage"
    NUMERIC_WITHOUT_PRIMARY_TRACE = "numeric_without_primary_trace"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    SUPPORTED = "supported"
    NOT_VERIFIED = "not_verified"
    KILLED_ON_REFUTE = "killed_on_refute"


class CitationStatus(StrEnum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"


class DataProvenance(StrEnum):
    UNAUDITED = "unaudited"
    MATCHED = "matched"
    MISMATCH = "mismatch"


class ConflictOutcome(StrEnum):
    RESOLVED_SUPPORTS = "resolved_supports"
    RESOLVED_REFUTES = "resolved_refutes"
    RESOLVED_QUALIFIED = "resolved_qualified"
    UNRESOLVED_PERSISTENT = "unresolved_persistent"


class StopReason(StrEnum):
    """Only CONVERGED is success (derive.is_success_stop); the rest are honest
    non-success terminals, or CONTINUE while the loop is still running."""

    CONVERGED = "converged"
    COMPLETED_WITH_OPEN_REQUIREMENTS = "completed_with_open_requirements"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ITERATION_CAP = "iteration_cap"
    CONTINUE = "continue"


class PublicationStatus(StrEnum):
    """DERIVED ONLY (ruling section A) — never stored on Claim; always recomputed by
    derive.publication_status() from verification/citation/provenance/gate state."""

    SUPPORTED = "supported"
    NOT_VERIFIED = "not_verified"
    CONTESTED = "contested"
    EXCLUDED = "excluded"

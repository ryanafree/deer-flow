"""Structured-data provenance audit (D8), ported from dr.js:1880-1893.

Pure, network-free: ``audit_provenance`` decides a target ``DataProvenance`` (or
``None`` to stay UNAUDITED) from a claim's ``data_ref`` and existing ``gate_flags``
only. It never touches ``gate_flags`` itself -- flags are minted at record time
(``claim_tool.py``) or by upstream extraction, never appended here.

``data_ref`` is a loosely-typed dict (no fixed schema, same latitude the ledger
models give it elsewhere -- see ``test_state_machine.py``'s ``{"value": "1"}``
fixtures). This audit reads two conventional keys when present: ``period`` (the
retrieved data point's own period, e.g. an EDGAR fact's fiscal-period end date) and
``claimed_period`` (the period the claim text asserts the value is FOR, when a
structured-claim writer populates it); their disagreement is the "stale vintage"
trap (`eval/profile_questions.yaml`'s ``financial_vintage`` fixture: a real FY2023
figure mislabeled as FY2024). A ``source_class`` of ``primary_filing`` or
``official_stat`` (the two structured source classes ``fetch/structured.py``'s
EDGAR/FRED connectors emit) marks the data_ref as primary-source.
"""

from __future__ import annotations

from dr_core.models.enums import DataProvenance, GateFlag, VerificationStatus
from dr_core.models.ledger import Claim

_MISMATCH_BLOCKING_FLAGS: frozenset[GateFlag] = frozenset(
    {
        GateFlag.MISSING_VINTAGE,
        GateFlag.PERIOD_MISMATCH,
        GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE,
    }
)
_PRIMARY_SOURCE_CLASSES = frozenset({"primary_filing", "official_stat"})


def audit_provenance(claim: Claim) -> DataProvenance | None:
    """The target ``data_provenance`` to advance to, or ``None`` to stay
    UNAUDITED. Only claims WITH a ``data_ref`` are audited at all; an already
    non-UNAUDITED claim is a no-op (the ladder is advance-only -- re-auditing a
    terminal outcome is pointless and would raise upstream in ``merge_ledger`` if
    the two disagreed, so this function simply declines to re-decide it).

    Mismatch evidence -> MISMATCH: any of the three blocking gate flags already
    present, OR a stated ``claimed_period`` that disagrees with the data_ref's own
    ``period``. No blocking flags, a primary-source class, and a present period ->
    MATCHED. Anything else (insufficient evidence either way) -> None.
    """
    if not claim.data_ref or claim.data_provenance != DataProvenance.UNAUDITED:
        return None

    if any(flag in claim.gate_flags for flag in _MISMATCH_BLOCKING_FLAGS):
        return DataProvenance.MISMATCH

    period = claim.data_ref.get("period")
    claimed_period = claim.data_ref.get("claimed_period")
    if claimed_period is not None and period is not None and str(claimed_period) != str(period):
        return DataProvenance.MISMATCH

    is_primary = claim.data_ref.get("source_class") in _PRIMARY_SOURCE_CLASSES
    if is_primary and period:
        return DataProvenance.MATCHED

    return None


def rescue_unaudited_matched(claim: Claim) -> DataProvenance | None:
    """Post-verify rescue (dr.js:2693): a data_ref claim that survives the vote
    pass as SUPPORTED with provenance still UNAUDITED advances to MATCHED --
    surviving adversarial verification is itself confirmatory evidence for a
    structured value that the deterministic audit above could not resolve either
    way. Returns ``None`` (no advance) for anything else, including claims already
    MATCHED/MISMATCH or not SUPPORTED."""
    if claim.data_ref and claim.data_provenance == DataProvenance.UNAUDITED and claim.verification.status == VerificationStatus.SUPPORTED:
        return DataProvenance.MATCHED
    return None

"""Structured-data provenance audit (D8, amended by D11 item 3), ported from
dr.js:1880-1893.

Pure, network-free: ``audit_provenance`` decides a target ``DataProvenance`` (or
``None`` to stay UNAUDITED) from a claim's ``data_ref``, its own ``text``, and
existing ``gate_flags`` only. It never touches ``gate_flags`` itself -- flags are
minted at record time (``claim_tool.py``) or by upstream extraction, never
appended here.

``data_ref`` is a loosely-typed dict (no fixed schema, same latitude the ledger
models give it elsewhere -- see ``test_state_machine.py``'s ``{"value": "1"}``
fixtures). This audit reads several conventional keys when present: ``period``
(the retrieved data point's own period, e.g. an EDGAR fact's fiscal-period end
date), ``claimed_period`` (the period the claim text asserts the value is FOR,
when a structured-claim writer populates it), ``value``/``unit`` (the retrieved
data point's own figure -- see ``dr_core/verify/provenance_compare.py``). A
disagreeing ``claimed_period`` vs. ``period`` is the "stale vintage" trap
(`eval/profile_questions.yaml`'s ``financial_vintage`` fixture: a real FY2023
figure mislabeled as FY2024). A ``source_class`` of ``primary_filing`` or
``official_stat`` (the two structured source classes ``fetch/structured.py``'s
EDGAR/FRED connectors emit), or ``primary_database`` (S9-C: WRDS's CRSP/Compustat
institutional academic data -- primary in the same sense, just not a filing or a
government statistic), marks the data_ref as primary-source.

D11 item 3 (DECISIONS.md): MATCHED over-promised relative to what the audit
actually checked -- a primary-source data_ref with a present period reached
MATCHED without ever comparing the claim's own asserted FIGURE against the
data_ref's retrieved value. MATCHED now additionally requires an affirmative
value/unit match (``provenance_compare.compare_claim_to_data_ref``, tolerance-
table normalized). The post-verify SUPPORTED+UNAUDITED -> MATCHED rescue
(dr.js:2693) that used to paper over this gap is REMOVED -- see
``dr_core/graph/verify.py`` -- because text-only voters never compare values,
so the rescue laundered a vote outcome into a provenance assertion the votes
never made. Accepted cost (D11 item 3): a data_ref claim whose record carries
no comparable value (or whose own text carries no parseable figure) now stays
UNAUDITED permanently instead of ever reaching MATCHED via the rescue.
"""

from __future__ import annotations

from dr_core.models.enums import DataProvenance, GateFlag
from dr_core.models.ledger import Claim
from dr_core.verify.provenance_compare import Comparison, compare_claim_to_data_ref

_MISMATCH_BLOCKING_FLAGS: frozenset[GateFlag] = frozenset(
    {
        GateFlag.MISSING_VINTAGE,
        GateFlag.PERIOD_MISMATCH,
        GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE,
    }
)
_PRIMARY_SOURCE_CLASSES = frozenset({"primary_filing", "official_stat", "primary_database"})


def audit_provenance(claim: Claim) -> DataProvenance | None:
    """The target ``data_provenance`` to advance to, or ``None`` to stay
    UNAUDITED. Only claims WITH a ``data_ref`` are audited at all; an already
    non-UNAUDITED claim is a no-op (the ladder is advance-only -- re-auditing a
    terminal outcome is pointless and would raise upstream in ``merge_ledger`` if
    the two disagreed, so this function simply declines to re-decide it).

    Mismatch evidence -> MISMATCH: any of the three blocking gate flags already
    present, OR a stated ``claimed_period`` that disagrees with the data_ref's own
    ``period``. Otherwise, a primary-source class with a present period is
    ELIGIBLE for MATCHED, but (D11 item 3) only actually reaches it when the
    claim's own asserted figure affirmatively matches the data_ref's ``value``
    (+ optional ``unit``) within tolerance after scale normalization; an
    affirmative contradiction there is also MISMATCH. Anything inconclusive
    (unparseable claim text, no comparable record value, incomparable units, not
    primary-source, or no period) -> None.
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
    if not (is_primary and period):
        return None

    outcome = compare_claim_to_data_ref(claim.text, claim.data_ref)
    if outcome is Comparison.MATCH:
        return DataProvenance.MATCHED
    if outcome is Comparison.MISMATCH:
        return DataProvenance.MISMATCH
    return None  # INCONCLUSIVE: unparseable/unnormalizable/incomparable -> stays UNAUDITED

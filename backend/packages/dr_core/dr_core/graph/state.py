"""Graph state for the dr_core outer-graph / middleware-contributed schema (D5).

Five channels carry the research-assurance ledger across the outer graph and
the nested lead-agent subgraph: ``dr_sources``, ``dr_claims``,
``dr_requirements``, ``dr_coverage`` (id-keyed dicts of ``model_dump(mode="json")``
payloads, per D2) and ``dr_run`` (a run-level marker dict, e.g.
``{"deliverable": bool, ...}``).

The four id-keyed channels use ``merge_ledger``, the strict keyed-monotonic /
transition-guarded reducer D2 specifies. ``merge_by_id`` is kept as the plain
building block (and its own pre-existing unit tests) but no longer backs any
channel here.
"""

from __future__ import annotations

from typing import Annotated

from langchain.agents import AgentState

from dr_core.models.derive import (
    assert_citation_transition_allowed,
    assert_provenance_transition_allowed,
    assert_verification_transition_allowed,
)
from dr_core.models.enums import CitationStatus, DataProvenance
from dr_core.models.ledger import SupportRecord, VerificationRecord

_MISSING = object()


def merge_by_id(existing: dict[str, dict] | None, new: dict[str, dict] | None) -> dict[str, dict]:
    """Keyed dict-merge reducer for the four id-keyed dr_* channels.

    ``new`` of ``None`` keeps ``existing`` unchanged; otherwise later writes
    win per key. Superseded by ``merge_ledger`` for the actual dr_* channels
    below; kept as the plain building block.
    """
    if new is None:
        return existing or {}
    if existing is None:
        return dict(new)
    merged = dict(existing)
    merged.update(new)
    return merged


def _merge_record(record_id: str, old: dict, new: dict) -> dict:
    """Field-by-field merge of two ``model_dump(mode="json")`` payloads for the
    SAME id, per D2's keyed-monotonic / transition-guarded rule.

    - ``verification`` is the one field the models actually mark advance-only:
      ``derive.assert_verification_transition_allowed`` guards its ``status``
      sub-field (the terminal ``killed_on_refute`` check; every other
      transition, including supported/not_verified -> pending re-select, is
      legal per that function). Both sides are validated to
      ``VerificationRecord`` only to run the guard (pydantic at the node
      boundary, per D2); once the guard passes, the whole ``verification``
      dict from ``new`` wins (last-writer-wins on its non-status sub-fields
      -- votes/complete/mode/selected/risk_reasons -- since one verify pass
      writes them atomically together; only status is a real ladder).
    - ``citation_status`` and ``data_provenance`` are the other two D2
      advance-only ladders: ``derive.assert_citation_transition_allowed`` and
      ``derive.assert_provenance_transition_allowed`` guard them the same way
      -- unresolved/unaudited may advance to either terminal outcome, and the
      terminal outcomes may never be left. A legal advance merges (new wins);
      an illegal one raises.
    - Every other field: an unchanged value passes through; a genuine
      divergence (two different values, neither a no-op) RAISES rather than
      silently picking one (D2: "same-field divergent writes fail loudly").
    """
    merged = dict(old)
    for key, new_val in new.items():
        old_val = old.get(key, _MISSING)
        if old_val is _MISSING or old_val == new_val:
            merged[key] = new_val
            continue
        if key == "verification" and isinstance(old_val, dict) and isinstance(new_val, dict):
            old_status = VerificationRecord.model_validate(old_val).status
            new_status = VerificationRecord.model_validate(new_val).status
            if old_status != new_status:
                assert_verification_transition_allowed(old_status, new_status)
            merged[key] = new_val
            continue
        if key == "support" and isinstance(old_val, dict) and isinstance(new_val, dict):
            old_support = SupportRecord.model_validate(old_val)
            new_support = SupportRecord.model_validate(new_val)
            if old_support.quote != new_support.quote or old_support.relation_extractor != new_support.relation_extractor:
                raise ValueError(f"divergent support assertion for id {record_id!r}")
            if old_support.relation_reviewer is not None and old_support.relation_reviewer != new_support.relation_reviewer:
                raise ValueError(f"divergent support review for id {record_id!r}")
            if old_support.reviewer_note is not None and old_support.reviewer_note != new_support.reviewer_note:
                raise ValueError(f"divergent support review note for id {record_id!r}")
            merged[key] = new_val
            continue
        if key == "citation_status":
            assert_citation_transition_allowed(CitationStatus(old_val), CitationStatus(new_val))
            merged[key] = new_val
            continue
        if key == "data_provenance":
            assert_provenance_transition_allowed(DataProvenance(old_val), DataProvenance(new_val))
            merged[key] = new_val
            continue
        raise ValueError(f"divergent write to field {key!r} for id {record_id!r}: {old_val!r} != {new_val!r}")
    return merged


def merge_ledger(existing: dict[str, dict] | None, new: dict[str, dict] | None) -> dict[str, dict]:
    """Strict keyed-monotonic / transition-guarded reducer (D2) for the four
    id-keyed dr_* channels (dr_sources/dr_claims/dr_requirements/dr_coverage).

    - ``new`` of ``None`` keeps ``existing`` unchanged; ``existing`` of
      ``None`` returns ``dict(new)``.
    - Distinct ids COMMUTE: ids present in only one side pass through
      unchanged.
    - Same id in both = an UPDATE: identical dicts are a no-op (idempotent --
      required for D5 outer-graph re-application safety, must never raise);
      otherwise merged field-by-field via ``_merge_record`` (illegal backward
      verification transitions and other same-field divergent writes RAISE).
    - Channel values stay plain ``dict[str, dict]`` throughout; pydantic is
      only used internally to run the transition guard (D2: pydantic types
      only at node boundaries).
    """
    if new is None:
        return existing or {}
    if existing is None:
        return dict(new)
    merged = dict(existing)
    for record_id, new_record in new.items():
        old_record = existing.get(record_id)
        if old_record is None or old_record == new_record:
            merged[record_id] = new_record
        else:
            merged[record_id] = _merge_record(record_id, old_record, new_record)
    return merged


def merge_run(existing: dict | None, new: dict | None) -> dict:
    """Dict-update reducer for the run-level ``dr_run`` marker channel."""
    if new is None:
        return existing or {}
    if existing is None:
        return dict(new)
    return {**existing, **new}


class DrAgentState(AgentState):
    """Middleware-contributed state schema merged into the nested lead-agent graph.

    ``create_agent`` merges each middleware's ``state_schema`` class attribute
    into the compiled graph's state (confirmed factory.py:1019-1024 /
    _resolve_schema 412-445), so declaring the channels here is sufficient for
    ``DrLedgerMiddleware`` to contribute them without touching ``ThreadState``.
    """

    dr_sources: Annotated[dict[str, dict], merge_ledger]
    dr_claims: Annotated[dict[str, dict], merge_ledger]
    dr_requirements: Annotated[dict[str, dict], merge_ledger]
    dr_coverage: Annotated[dict[str, dict], merge_ledger]
    dr_run: Annotated[dict, merge_run]


class DrOuterState(AgentState):
    """Outer-graph state: mirrors the same 5 channels + the same reducers.

    Per D5-B, LangGraph propagates shared keys parent-ward from the nested
    subgraph node ("research") to this outer graph, which is safe because the
    reducers above are keyed/idempotent. ``messages`` is inherited from
    ``AgentState`` so it coexists with the dr_* channels.

    ``title`` mirrors ThreadState's channel so TitleMiddleware's write inside
    the nested subgraph propagates to the outer checkpoint, where
    ``runtime/runs/worker.py`` reads it for thread display-name sync (without
    this the title feature is silently inert for dr_agent runs). ``goal`` is
    deliberately NOT mirrored: the goal-continuation loop re-enters the graph
    with hidden turns, which would interact with the gate's corrective loop;
    add it only after that interaction is designed.
    """

    title: str

    dr_sources: Annotated[dict[str, dict], merge_ledger]
    dr_claims: Annotated[dict[str, dict], merge_ledger]
    dr_requirements: Annotated[dict[str, dict], merge_ledger]
    dr_coverage: Annotated[dict[str, dict], merge_ledger]
    dr_run: Annotated[dict, merge_run]

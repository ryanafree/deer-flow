"""Unit tests for the strict D2 keyed-monotonic reducer
(``dr_core.graph.state.merge_ledger``).

Channel payloads are built the same way real nodes will: ``dr_core.models``
instances dumped via ``model_dump(mode="json")`` -- never hand-rolled dicts --
so a schema drift in ``ledger.py`` would break these tests too, not silently
pass a stale fixture.
"""

from __future__ import annotations

import copy

import pytest
from dr_core.graph.state import merge_ledger
from dr_core.models import CitationStatus, Claim, DataProvenance, SupportRecord, SupportRelation, VerificationRecord, VerificationStatus


def _claim(claim_id: str = "c1", *, status: VerificationStatus = VerificationStatus.PENDING, **overrides) -> dict:
    payload = dict(
        claim_id=claim_id,
        text="the sky is blue",
        importance=3,
        source_id="s1",
        verification=VerificationRecord(status=status),
    )
    payload.update(overrides)
    return Claim(**payload).model_dump(mode="json")


class TestMergeLedgerNoneHandling:
    def test_new_none_keeps_existing(self):
        existing = {"c1": _claim()}
        assert merge_ledger(existing, None) == existing

    def test_existing_none_returns_new(self):
        new = {"c1": _claim()}
        assert merge_ledger(None, new) == new


class TestMergeLedgerDistinctIdsCommute:
    def test_disjoint_ids_union(self):
        existing = {"c1": _claim("c1")}
        new = {"c2": _claim("c2")}
        merged = merge_ledger(existing, new)
        assert merged == {"c1": existing["c1"], "c2": new["c2"]}


class TestMergeLedgerVerificationTransition:
    def test_legal_forward_transition_merges(self):
        existing = {"c1": _claim("c1", status=VerificationStatus.PENDING)}
        new = {"c1": _claim("c1", status=VerificationStatus.SUPPORTED)}
        merged = merge_ledger(existing, new)
        assert merged["c1"]["verification"]["status"] == VerificationStatus.SUPPORTED.value

    def test_illegal_backward_transition_raises(self):
        existing = {"c1": _claim("c1", status=VerificationStatus.KILLED_ON_REFUTE)}
        new = {"c1": _claim("c1", status=VerificationStatus.PENDING)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)


class TestMergeLedgerSupportReview:
    def test_missing_reviewer_can_advance_once(self):
        support = SupportRecord(quote="the sky is blue", relation_extractor=SupportRelation.SUPPORTS_DIRECTLY)
        reviewed = support.model_copy(update={"relation_reviewer": SupportRelation.SUPPORTS_DIRECTLY, "reviewer_note": "Exact match."})

        merged = merge_ledger({"c1": _claim(support=support)}, {"c1": _claim(support=reviewed)})

        assert merged["c1"]["support"]["relation_reviewer"] == "supports_directly"

    def test_completed_reviewer_cannot_be_rewritten(self):
        first = SupportRecord(
            quote="the sky is blue",
            relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
            relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY,
        )
        changed = first.model_copy(update={"relation_reviewer": SupportRelation.CONTEXT_ONLY})

        with pytest.raises(ValueError, match="divergent support review"):
            merge_ledger({"c1": _claim(support=first)}, {"c1": _claim(support=changed)})


class TestMergeLedgerIdempotent:
    def test_identical_reapplication_is_noop(self):
        record = _claim("c1", status=VerificationStatus.SUPPORTED)
        existing = {"c1": copy.deepcopy(record)}
        new = {"c1": copy.deepcopy(record)}
        merged = merge_ledger(existing, new)
        assert merged == existing


class TestMergeLedgerDivergentRaise:
    def test_same_field_divergent_non_monotonic_write_raises(self):
        existing = {"c1": _claim("c1", importance=2)}
        new = {"c1": _claim("c1", importance=4)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)

    def test_legal_citation_advance_merges(self):
        # unresolved -> not_found is a legal ladder advance (D2), not a divergent
        # write -- see TestMergeLedgerCitationTransition / TestMergeLedgerProvenanceTransition below.
        existing = {"c1": _claim("c1", citation_status=CitationStatus.UNRESOLVED)}
        new = {"c1": _claim("c1", citation_status=CitationStatus.NOT_FOUND)}
        merged = merge_ledger(existing, new)
        assert merged["c1"]["citation_status"] == CitationStatus.NOT_FOUND.value

    def test_divergent_citation_status_backward_raises(self):
        existing = {"c1": _claim("c1", citation_status=CitationStatus.NOT_FOUND)}
        new = {"c1": _claim("c1", citation_status=CitationStatus.UNRESOLVED)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)


class TestMergeLedgerCitationTransition:
    def test_legal_forward_transition_merges(self):
        existing = {"c1": _claim("c1", citation_status=CitationStatus.UNRESOLVED)}
        new = {"c1": _claim("c1", citation_status=CitationStatus.RESOLVED)}
        merged = merge_ledger(existing, new)
        assert merged["c1"]["citation_status"] == CitationStatus.RESOLVED.value

    def test_illegal_transition_out_of_terminal_raises(self):
        existing = {"c1": _claim("c1", citation_status=CitationStatus.RESOLVED)}
        new = {"c1": _claim("c1", citation_status=CitationStatus.NOT_FOUND)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)

    def test_idempotent_reapplication_is_noop(self):
        record = _claim("c1", citation_status=CitationStatus.NOT_FOUND)
        existing = {"c1": copy.deepcopy(record)}
        new = {"c1": copy.deepcopy(record)}
        merged = merge_ledger(existing, new)
        assert merged == existing


class TestMergeLedgerProvenanceTransition:
    def test_legal_forward_transition_merges(self):
        existing = {"c1": _claim("c1", data_provenance=DataProvenance.UNAUDITED)}
        new = {"c1": _claim("c1", data_provenance=DataProvenance.MATCHED)}
        merged = merge_ledger(existing, new)
        assert merged["c1"]["data_provenance"] == DataProvenance.MATCHED.value

    def test_illegal_transition_out_of_terminal_raises(self):
        existing = {"c1": _claim("c1", data_provenance=DataProvenance.MATCHED)}
        new = {"c1": _claim("c1", data_provenance=DataProvenance.MISMATCH)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)

    def test_idempotent_reapplication_is_noop(self):
        record = _claim("c1", data_provenance=DataProvenance.MISMATCH)
        existing = {"c1": copy.deepcopy(record)}
        new = {"c1": copy.deepcopy(record)}
        merged = merge_ledger(existing, new)
        assert merged == existing

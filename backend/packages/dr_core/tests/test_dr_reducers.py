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
from dr_core.models import Claim, CitationStatus, VerificationRecord, VerificationStatus


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

    def test_divergent_citation_status_raises(self):
        existing = {"c1": _claim("c1", citation_status=CitationStatus.UNRESOLVED)}
        new = {"c1": _claim("c1", citation_status=CitationStatus.NOT_FOUND)}
        with pytest.raises(ValueError):
            merge_ledger(existing, new)

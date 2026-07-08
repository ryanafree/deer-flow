"""Tests for dr_core.verify.selection (D8 claim selection). Pure, no I/O."""

from dr_core.models.enums import CoverageRelation, VerificationStatus
from dr_core.models.ledger import Claim, VerificationRecord
from dr_core.verify.selection import MAX_VERIFY, max_verify_for_depth, select_verification_claim_ids


def _claim(claim_id: str, *, importance: int = 3, source_id: str = "s1", **overrides) -> Claim:
    defaults = dict(claim_id=claim_id, text=f"claim {claim_id}", importance=importance, source_id=source_id)
    defaults.update(overrides)
    return Claim(**defaults)


def _source(source_id: str, *, authority_tier: int = 2, source_system: str = "web") -> dict:
    return {"id": source_id, "url_or_id": f"https://example.com/{source_id}", "source_system": source_system, "authority_tier": authority_tier, "retrieved_at": "2026-07-06T00:00:00+00:00"}


class TestMaxVerifyForDepth:
    def test_known_depths(self):
        assert max_verify_for_depth("quick") == MAX_VERIFY["quick"] == 5
        assert max_verify_for_depth("standard") == MAX_VERIFY["standard"] == 12
        assert max_verify_for_depth("full") == MAX_VERIFY["full"] == 25

    def test_unknown_or_missing_depth_defaults_to_standard(self):
        assert max_verify_for_depth(None) == MAX_VERIFY["standard"]
        assert max_verify_for_depth("weird") == MAX_VERIFY["standard"]


class TestRankFillOrder:
    def test_importance_desc_breaks_ties_by_authority_then_claim_id(self):
        claims = {
            "c3": _claim("c3", importance=3, source_id="s1"),
            "c1": _claim("c1", importance=5, source_id="s1"),
            "c2": _claim("c2", importance=5, source_id="s2"),
        }
        sources = {"s1": _source("s1", authority_tier=2), "s2": _source("s2", authority_tier=1)}
        selected = select_verification_claim_ids(claims, sources, {}, {}, max_verify=10)
        # c2 (importance 5, tier 1) beats c1 (importance 5, tier 2) beats c3 (importance 3).
        assert selected == ["c2", "c1", "c3"]

    def test_missing_source_sorts_as_worst_tier(self):
        claims = {"c1": _claim("c1", importance=4, source_id="s-missing"), "c2": _claim("c2", importance=4, source_id="s1")}
        sources = {"s1": _source("s1", authority_tier=1)}
        selected = select_verification_claim_ids(claims, sources, {}, {}, max_verify=10)
        assert selected == ["c2", "c1"]

    def test_depth_cap_truncates_rank_fill(self):
        claims = {cid: _claim(cid, importance=3) for cid in ("c1", "c2", "c3")}
        sources = {"s1": _source("s1")}
        selected = select_verification_claim_ids(claims, sources, {}, {}, max_verify=2)
        assert len(selected) == 2


class TestMustCoverFirst:
    def test_must_cover_direct_supporter_selected_before_rank_fill(self):
        claims = {
            "c1": _claim("c1", importance=1),
            "c2": _claim("c2", importance=5),
        }
        sources = {"s1": _source("s1")}
        requirements = {"r1": {"id": "r1", "must_cover": True}}
        coverage = {"m1": {"requirement_id": "r1", "claim_id": "c1", "relation": CoverageRelation.DIRECT.value}}
        selected = select_verification_claim_ids(claims, sources, requirements, coverage, max_verify=10)
        assert selected[0] == "c1"
        assert "c2" in selected

    def test_degrades_gracefully_with_empty_requirements_and_coverage(self):
        claims = {"c1": _claim("c1")}
        sources = {"s1": _source("s1")}
        assert select_verification_claim_ids(claims, sources, {}, {}, max_verify=10) == ["c1"]
        assert select_verification_claim_ids(claims, sources, None, None, max_verify=10) == ["c1"]


class TestSeededTrapAlwaysIncluded:
    def test_seeded_trap_claim_included_even_beyond_cap(self):
        claims = {
            "c1": _claim("c1", importance=5, source_id="s1"),
            "c2": _claim("c2", importance=5, source_id="s1"),
            "trap": _claim("trap", importance=1, source_id="s-trap"),
        }
        sources = {"s1": _source("s1"), "s-trap": _source("s-trap", source_system="seeded-trap")}
        selected = select_verification_claim_ids(claims, sources, {}, {}, max_verify=2)
        assert "trap" in selected
        assert len(selected) == 3  # cap of 2 rank-filled + the always-included trap


class TestCompleteTerminalNeverReselected:
    def test_complete_claim_excluded(self):
        claims = {
            "c1": _claim("c1", verification=VerificationRecord(complete=True, status=VerificationStatus.SUPPORTED)),
            "c2": _claim("c2"),
        }
        sources = {"s1": _source("s1")}
        assert select_verification_claim_ids(claims, sources, {}, {}, max_verify=10) == ["c2"]

    def test_killed_on_refute_excluded_even_if_not_marked_complete(self):
        claims = {
            "c1": _claim("c1", verification=VerificationRecord(complete=False, status=VerificationStatus.KILLED_ON_REFUTE)),
            "c2": _claim("c2"),
        }
        sources = {"s1": _source("s1")}
        assert select_verification_claim_ids(claims, sources, {}, {}, max_verify=10) == ["c2"]

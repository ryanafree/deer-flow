"""Tests for the D9 requirement-coverage extension to eligibility_gate (gate.py).
Hermetic: no model/network, drives the node function directly with synthetic
state dicts. Companion to test_dr_gate.py (D6 eligibility-only predicate),
which stays green unmodified -- these tests exercise ONLY the added
must-cover-coverage predicate, breaker-signature extension, and count
freezing.
"""

from dr_core.graph.agent import route_after_gate
from dr_core.graph.gate import eligibility_gate
from dr_core.models.enums import CoverageRelation, SupportRelation
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, SupportRecord


def _claim(claim_id: str, source_id: str = "s1", **overrides) -> dict:
    # Grounded by default (a real supporting quote) so evidence_state's
    # grounded-claims-only filter (derive.py:241) doesn't silently drop this
    # claim's coverage mappings from every requirement's evidence.
    defaults = dict(
        claim_id=claim_id,
        text=f"claim {claim_id}",
        importance=4,
        source_id=source_id,
        # relation_reviewer set (not just the extractor) so grounding survives
        # even at HIGH materiality (a direct-must-cover mapping bump): an
        # unreviewed claim's effective_relation degrades to "unreviewed" at
        # HIGH materiality (D2 ruling divergence 3) regardless of quote length.
        support=SupportRecord(
            quote="the primary source states this fact directly and unambiguously",
            relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
            relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY,
        ),
    )
    defaults.update(overrides)
    return Claim(**defaults).model_dump(mode="json")


def _source(source_id: str) -> dict:
    return {"id": source_id, "url_or_id": f"https://example.com/{source_id}", "source_system": "web", "authority_tier": 3, "retrieved_at": "2026-07-07T00:00:00+00:00"}


def _requirement(req_id: str, text: str, *, must_cover: bool = True, kind: str = "entity") -> dict:
    return Requirement(id=req_id, kind=kind, text=text, must_cover=must_cover).model_dump(mode="json")


def _mapping(req_id: str, claim_id: str, relation: str) -> dict:
    return CoverageMapping(requirement_id=req_id, claim_id=claim_id, relation=relation).model_dump(mode="json")


class TestUncoveredMustCoverBounces:
    def test_uncovered_must_cover_forces_retry_with_verbatim_text(self):
        state = {
            "dr_claims": {"c1": _claim("c1")},
            "dr_sources": {"s1": _source("s1")},
            "dr_requirements": {"req1": _requirement("req1", "Name the current CEO of the company.")},
            "dr_coverage": {},
            "dr_run": {"gate_retries": 0, "active_requirement_ids": ["req1"]},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "research"
        assert route_after_gate({"dr_run": dr_run}) == "research"
        corrective = result["messages"][0]
        assert "Name the current CEO of the company." in corrective.content
        assert dr_run["first_pass_requirements_covered"] == 0
        assert dr_run["first_pass_requirements_must_cover"] == 1
        assert dr_run["first_pass_must_cover_states"] == {"req1": "uncovered"}


class TestCoveredProceeds:
    def test_direct_mapping_to_eligible_claim_covers_and_proceeds(self):
        state = {
            "dr_claims": {"c1": _claim("c1")},
            "dr_sources": {"s1": _source("s1")},
            "dr_requirements": {"req1": _requirement("req1", "Name the current CEO of the company.")},
            "dr_coverage": {"req1:c1": _mapping("req1", "c1", CoverageRelation.DIRECT.value)},
            "dr_run": {"gate_retries": 0, "active_requirement_ids": ["req1"]},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert "stop_reason" not in dr_run
        assert dr_run["requirements_covered"] == 1
        assert dr_run["requirements_must_cover"] == 1
        assert dr_run["must_cover_states"] == {"req1": "covered"}
        assert route_after_gate({"dr_run": dr_run}) == "render"


class TestCountsFrozenOnProceedWithOpenRequirement:
    def test_partial_must_cover_forces_open_stop_reason_after_cap(self):
        # Retries exhausted forces PROCEED even though req1 never reaches COVERED
        # (no direct mapping at all -- stays UNCOVERED).
        state = {
            "dr_claims": {"c1": _claim("c1")},
            "dr_sources": {"s1": _source("s1")},
            "dr_requirements": {"req1": _requirement("req1", "Name the current CEO of the company.")},
            "dr_coverage": {},
            "dr_run": {"gate_retries": 2, "active_requirement_ids": ["req1"]},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["requirements_covered"] == 0
        assert dr_run["requirements_must_cover"] == 1
        from dr_core.models.enums import StopReason

        assert dr_run["stop_reason"] == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value


class TestBreakerSeesCoverageProgress:
    def test_retry_adding_coverage_without_new_claims_does_not_fire_breaker(self):
        """Two must-cover requirements: req_a never gets any mapping (stays
        UNCOVERED throughout, so RETRY keeps firing correctly); req_b starts
        UNCOVERED and gains a PARTIAL mapping between the two gate calls with
        NO new claim added. The signature must see req_b's state change, so
        the one-strike breaker must NOT fire on the second call even though
        gate_retries > 0 and nothing about the claims/sources changed."""
        dr_claims = {"c1": _claim("c1")}
        dr_sources = {"s1": _source("s1")}
        dr_requirements = {
            "req_a": _requirement("req_a", "Requirement A never gets evidence."),
            "req_b": _requirement("req_b", "Requirement B gains partial evidence."),
        }

        state1 = {
            "dr_claims": dr_claims,
            "dr_sources": dr_sources,
            "dr_requirements": dr_requirements,
            "dr_coverage": {},
            "dr_run": {"gate_retries": 0, "active_requirement_ids": ["req_a", "req_b"]},
        }
        result1 = eligibility_gate(state1)
        dr_run1 = result1["dr_run"]
        assert dr_run1["gate_decision"] == "research"

        # Second pass: req_b now has a PARTIAL-relation mapping (progress),
        # req_a still has none. Same claims/sources, retries carried forward.
        state2 = {
            "dr_claims": dr_claims,
            "dr_sources": dr_sources,
            "dr_requirements": dr_requirements,
            "dr_coverage": {"req_b:c1": _mapping("req_b", "c1", CoverageRelation.PARTIAL.value)},
            "dr_run": {"gate_retries": dr_run1["gate_retries"], "gate_ledger_sig": dr_run1["gate_ledger_sig"], "active_requirement_ids": ["req_a", "req_b"]},
        }
        result2 = eligibility_gate(state2)
        dr_run2 = result2["dr_run"]
        # Still retrying (req_a is UNCOVERED), not forced to render by the breaker.
        assert dr_run2["gate_decision"] == "research"
        assert dr_run2["gate_ledger_sig"] != dr_run1["gate_ledger_sig"]


class TestActiveScopingExcludesStaleRequirement:
    def test_stale_prior_turn_must_cover_never_bounces(self):
        """req1 is must_cover and UNCOVERED, but NOT named in this turn's
        active_requirement_ids -- a leftover from a prior turn. It must not
        hold this turn's gate hostage."""
        state = {
            "dr_claims": {"c1": _claim("c1")},
            "dr_sources": {"s1": _source("s1")},
            "dr_requirements": {"req1": _requirement("req1", "Stale requirement from a prior turn.")},
            "dr_coverage": {},
            "dr_run": {"gate_retries": 0, "active_requirement_ids": []},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert "requirements_covered" not in dr_run
        assert "stop_reason" not in dr_run

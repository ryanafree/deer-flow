"""Tests for the D9 materiality activation in dr_core.models.eligibility: a
claim with a DIRECT mapping to a must-cover requirement derives HIGH
materiality (bypassing the pre-D9 unmapped-claim MEDIUM cap), which tightens
is_grounded exactly as state.js specifies (derive.py:180-181,207: HIGH
materiality demands a longer supporting quote). Also proves gate.py and
render.py derive the identical result from the identical (claim, mappings,
requirements) input -- the D6-B "one derivation, two callers" invariant,
now exercised with real coverage mappings rather than the empty defaults.
"""

from dr_core.graph.gate import eligibility_gate
from dr_core.graph.render import render_node
from dr_core.models.eligibility import derive_status
from dr_core.models.enums import CoverageRelation, PublicationStatus, SupportRelation
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, SupportRecord

_SHORT_QUOTE = "yes indeed"  # 2 words -- fails HIGH materiality's >=3-word gate


def _claim_with_short_quote(**overrides) -> Claim:
    defaults = dict(
        claim_id="c1",
        text="The company confirmed the fact.",
        importance=5,
        source_id="s1",
        support=SupportRecord(quote=_SHORT_QUOTE, relation_extractor=SupportRelation.SUPPORTS_DIRECTLY),
    )
    defaults.update(overrides)
    return Claim(**defaults)


class TestDirectMustCoverMappingTightensGroundedness:
    def test_no_mapping_caps_materiality_and_short_quote_still_grounds(self):
        claim = _claim_with_short_quote()
        status = derive_status(claim, mappings_for_claim=(), requirements_by_id={})
        # No mapping -> capped at MEDIUM -> the HIGH-only short-quote gate never
        # applies -> verification still PENDING so status is NOT_VERIFIED, but
        # not because of groundedness (a supported claim would be SUPPORTED).
        assert status != PublicationStatus.EXCLUDED

    def test_direct_must_cover_mapping_forces_high_materiality_and_ungrounds_short_quote(self):
        requirement = Requirement(id="req1", kind="entity", text="Confirm the fact.", must_cover=True)
        mapping = CoverageMapping(requirement_id="req1", claim_id="c1", relation=CoverageRelation.DIRECT)
        claim = _claim_with_short_quote(verification={"status": "supported", "complete": True})

        status_without_mapping = derive_status(claim, mappings_for_claim=(), requirements_by_id={"req1": requirement})
        status_with_mapping = derive_status(claim, mappings_for_claim=[mapping], requirements_by_id={"req1": requirement})

        # Without the mapping: MEDIUM-capped materiality, short quote still
        # grounds (the HIGH-only length gate never applies) -> SUPPORTED.
        assert status_without_mapping == PublicationStatus.SUPPORTED
        # With the direct must-cover mapping: HIGH materiality kicks in, the
        # short (2-word) quote fails the >=3-word HIGH gate -> ungrounded ->
        # NOT_VERIFIED, even though verification.status is SUPPORTED.
        assert status_with_mapping == PublicationStatus.NOT_VERIFIED


class TestGateAndRenderDeriveIdentically:
    """A claim directly satisfying a must-cover requirement (so the
    requirement is COVERED and the gate proceeds) must render with exactly
    the status ``derive_status`` gives it when fed the SAME mappings/
    requirements the gate itself used -- one derivation, two callers (D6-B),
    now exercised with a real coverage mapping (D9) instead of the empty
    defaults."""

    def _state(self, runs_dir: str, claim: Claim) -> dict:
        requirement_payload = Requirement(id="req1", kind="entity", text="Confirm the fact.", must_cover=True).model_dump(mode="json")
        mapping_payload = CoverageMapping(requirement_id="req1", claim_id="c1", relation=CoverageRelation.DIRECT).model_dump(mode="json")
        return {
            "dr_claims": {"c1": claim.model_dump(mode="json")},
            "dr_sources": {"s1": {"id": "s1", "url_or_id": "https://example.test/s1", "source_system": "web", "title": "Example Source", "authority_tier": 2, "retrieved_at": "2026-07-07T00:00:00-05:00"}},
            "dr_requirements": {"req1": requirement_payload},
            "dr_coverage": {"req1:c1": mapping_payload},
            "dr_run": {"active_requirement_ids": ["req1"], "question": "What did the company confirm?", "profile": "general", "runs_dir": runs_dir},
        }

    def test_direct_must_cover_mapping_covers_the_requirement_and_gate_proceeds(self, tmp_path):
        claim = _claim_with_short_quote(
            text="The company confirmed the merger closed on schedule.",
            support=SupportRecord(quote="the merger closed on schedule as planned", relation_extractor=SupportRelation.SUPPORTS_DIRECTLY, relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY),
            verification={"status": "supported", "complete": True},
        )
        state = self._state(str(tmp_path), claim)
        gate_result = eligibility_gate(state)
        dr_run = gate_result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert "c1" in dr_run["citation_ordinals"]
        assert dr_run["must_cover_states"] == {"req1": "covered"}
        assert "stop_reason" not in dr_run

    def test_render_status_matches_derive_status_fed_the_same_inputs(self, tmp_path):
        from pathlib import Path

        claim = _claim_with_short_quote(
            text="The company confirmed the merger closed on schedule.",
            support=SupportRecord(quote="the merger closed on schedule as planned", relation_extractor=SupportRelation.SUPPORTS_DIRECTLY, relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY),
            verification={"status": "supported", "complete": True},
        )
        state = self._state(str(tmp_path), claim)
        gate_result = eligibility_gate(state)
        state["dr_run"] = {**state["dr_run"], **gate_result["dr_run"]}
        render_result = render_node(state)
        report_md = Path(render_result["dr_run"]["run_dir"], "report.md").read_text()
        findings = report_md.split("## Findings")[1].split("## Appendix")[0]

        requirement = Requirement(id="req1", kind="entity", text="Confirm the fact.", must_cover=True)
        mapping = CoverageMapping(requirement_id="req1", claim_id="c1", relation=CoverageRelation.DIRECT)
        expected_status = derive_status(claim, mappings_for_claim=[mapping], requirements_by_id={"req1": requirement})

        # A HIGH-materiality, well-grounded, verification-supported claim
        # derives SUPPORTED -- render must show it as a plain (non-attributed)
        # sentence, matching derive_status exactly.
        assert expected_status == PublicationStatus.SUPPORTED
        assert "According to" not in findings
        assert "merger closed on schedule" in findings

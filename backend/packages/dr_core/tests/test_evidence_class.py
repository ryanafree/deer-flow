"""SPEC_evidence_routing_2026-07-10.md Stage 2 tests (section B, tests 5-8):
evidence-class schema, resolution, and coverage. Hermetic: no model/network.
Companion to test_gate_coverage.py (D9 coverage predicate) and
test_plan_mapping.py (coverage-mapping extraction), both of which stay green
unmodified -- this file exercises ONLY the evidence-class additions.
"""

import pytest
from dr_core.connectors.registry import by_name, load_connectors
from dr_core.graph.gate import eligibility_gate
from dr_core.models.derive import ClaimEvidence, evidence_state
from dr_core.models.enums import CoverageRelation, RequirementState
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, SupportRecord
from dr_core.plan.evidence_class import resolve_evidence_class
from dr_core.profiles import load_profile


class TestRequirementEvidenceClassSchema:
    """Test 5: Requirement parses evidence_class/freshness; missing or
    invalid values coerce to "any"."""

    @pytest.mark.parametrize("value", ["academic", "primary_data", "practitioner", "news", "any"])
    def test_valid_evidence_class_values_parse(self, value):
        req = Requirement(id="r1", kind="entity", text="t", evidence_class=value)
        assert req.evidence_class == value

    @pytest.mark.parametrize("value", ["foundational", "frontier", "any"])
    def test_valid_freshness_values_parse(self, value):
        req = Requirement(id="r1", kind="entity", text="t", freshness=value)
        assert req.freshness == value

    def test_missing_values_default_to_any(self):
        req = Requirement(id="r1", kind="entity", text="t")
        assert req.evidence_class == "any"
        assert req.freshness == "any"

    def test_invalid_values_coerce_to_any(self):
        req = Requirement(id="r1", kind="entity", text="t", evidence_class="bogus", freshness="whenever")
        assert req.evidence_class == "any"
        assert req.freshness == "any"

    def test_explicit_none_coerces_to_any(self):
        req = Requirement(id="r1", kind="entity", text="t", evidence_class=None, freshness=None)
        assert req.evidence_class == "any"
        assert req.freshness == "any"


class TestEvidenceClassResolution:
    """Test 6: connector source_system resolves via the registry; a web
    source resolves via the domain fallback; an unknown domain is news."""

    def test_connector_source_system_resolves_primary_data(self):
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("fred", "https://api.stlouisfed.org/fred/series?id=X", connectors_by_name) == "primary_data"

    def test_connector_source_system_resolves_academic(self):
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("openalex", "https://api.openalex.org/works/W123", connectors_by_name) == "academic"

    def test_web_source_arxiv_domain_resolves_academic(self):
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("web", "https://arxiv.org/abs/2601.00001", connectors_by_name) == "academic"

    def test_web_source_ssrn_subdomain_resolves_academic(self):
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("web", "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1", connectors_by_name) == "academic"

    def test_unknown_web_domain_resolves_news(self):
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("web", "https://randomblog.example.com/post-1", connectors_by_name) == "news"

    def test_connector_without_evidence_class_field_resolves_news(self):
        # tavily is Tier 0 and carries no evidence_class row -- falls through
        # to the default rather than raising.
        connectors_by_name = by_name(load_connectors())
        assert resolve_evidence_class("tavily", "https://api.tavily.com/result/1", connectors_by_name) == "news"


class TestEvidenceClassCoverage:
    """Test 7: a DIRECT mapping citing only wrong-class sources yields at
    most PARTIAL for a class-tagged requirement; a right-class DIRECT
    mapping yields COVERED; class "any" behaves exactly as today."""

    def _mapping(self):
        return [CoverageMapping(requirement_id="r1", claim_id="c1", relation=CoverageRelation.DIRECT)]

    def test_wrong_class_direct_yields_at_most_partial(self):
        req = Requirement(id="r1", kind="entity", text="t", must_cover=True, evidence_class="academic")
        claims_by_id = {"c1": ClaimEvidence(grounded=True, source_id="s1", evidence_class="news")}
        assert evidence_state(req, self._mapping(), claims_by_id) == RequirementState.PARTIAL

    def test_right_class_direct_yields_covered(self):
        req = Requirement(id="r1", kind="entity", text="t", must_cover=True, evidence_class="academic")
        claims_by_id = {"c1": ClaimEvidence(grounded=True, source_id="s1", evidence_class="academic")}
        assert evidence_state(req, self._mapping(), claims_by_id) == RequirementState.COVERED

    def test_no_evidence_at_all_is_uncovered_regardless_of_class(self):
        req = Requirement(id="r1", kind="entity", text="t", must_cover=True, evidence_class="academic")
        assert evidence_state(req, [], {}) == RequirementState.UNCOVERED

    def test_any_class_requirement_is_unaffected_by_source_class(self):
        # Regression: evidence_class == "any" (the default) must reduce to
        # exactly the pre-Stage-2 evidence_state behavior.
        req = Requirement(id="r1", kind="entity", text="t", must_cover=True)
        claims_by_id = {"c1": ClaimEvidence(grounded=True, source_id="s1", evidence_class="news")}
        assert evidence_state(req, self._mapping(), claims_by_id) == RequirementState.COVERED


class TestRetryPromptNamesClassAllowlistedTool:
    """Test 8, amended by D11 item 4: the corrective message for an uncovered
    academic-class requirement under the financial profile names the ACTUAL
    bound academic tool (academic_search -- directives must use bound tool
    names, the root-cause track), and no tool absent from the profile's
    allowlist."""

    def _claim(self, claim_id: str, source_id: str = "s1") -> dict:
        return Claim(
            claim_id=claim_id,
            text=f"claim {claim_id}",
            importance=4,
            source_id=source_id,
            support=SupportRecord(quote="a supporting quote", relation_extractor="supports_directly", relation_reviewer="supports_directly"),
        ).model_dump(mode="json")

    def _source(self, source_id: str) -> dict:
        return {"id": source_id, "url_or_id": f"https://example.com/{source_id}", "source_system": "web", "authority_tier": 3, "retrieved_at": "2026-07-10T00:00:00+00:00"}

    def _state(self) -> dict:
        requirement = Requirement(id="req1", kind="entity", text="Name the seminal paper establishing this result.", must_cover=True, evidence_class="academic").model_dump(mode="json")
        return {
            "dr_claims": {"c1": self._claim("c1")},
            "dr_sources": {"s1": self._source("s1")},
            "dr_requirements": {"req1": requirement},
            "dr_coverage": {},
            "dr_run": {"gate_retries": 0, "active_requirement_ids": ["req1"], "profile": "financial"},
        }

    def test_corrective_message_names_an_allowlisted_academic_tool(self):
        result = eligibility_gate(self._state())
        corrective = result["messages"][0].content
        # D11: the bound tool name, not the underlying connector ids, is what
        # a directive must tell the model to call.
        assert "academic_search" in corrective

    def test_corrective_message_names_no_tool_outside_the_profile_allowlist(self):
        result = eligibility_gate(self._state())
        corrective = result["messages"][0].content
        allowlist = set(load_profile("financial")["tool_allowlist"])
        all_connector_names = {c.name for c in load_connectors()}
        mentioned = {name for name in all_connector_names if name in corrective}
        assert mentioned <= allowlist
        # Sanity: at least one connector name IS mentioned, so the assertion
        # above is exercising something rather than vacuously passing.
        assert mentioned

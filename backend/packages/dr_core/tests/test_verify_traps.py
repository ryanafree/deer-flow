"""D8 acceptance (S7 gate): the two seeded-trap fixtures from
`eval/profile_questions.yaml` (legal_fabricated_cite, financial_vintage), run
offline end-to-end through verify_node -> eligibility_gate. No network, no real
LLM -- the vote model and citation lookup are monkeypatched; the fixtures'
`injected_claims` are constructed directly as Claim payloads (the live YAML
fixtures are driven by the old JS harness's own runner, not this Python graph).
"""

import json
from types import SimpleNamespace

from dr_core.graph.gate import eligibility_gate
from dr_core.graph.state import merge_ledger
from dr_core.graph.verify import verify_node
from dr_core.models.eligibility import derive_status, ineligibility_reason
from dr_core.models.enums import CitationStatus, DataProvenance, PublicationStatus
from dr_core.models.ledger import Claim
from dr_core.verify import votes as votes_mod


def _source(source_id: str, **overrides) -> dict:
    defaults = dict(id=source_id, url_or_id=f"https://example.com/{source_id}", source_system="web", authority_tier=2, retrieved_at="2026-07-06T00:00:00+00:00")
    defaults.update(overrides)
    return defaults


def _clean_vote_response() -> SimpleNamespace:
    content = json.dumps({"refuted": False, "abstain": False, "confidence": "high", "reasoning": "well supported"})
    return SimpleNamespace(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cache_read": 0, "cache_creation": 0}})


class _FakeCleanModel:
    async def ainvoke(self, messages):
        return _clean_vote_response()


async def _no_evidence(url):
    return None


class TestLegalFabricatedCiteTrap:
    """injected_claims[1] from the legal_fabricated_cite fixture: 'The Supreme
    Court abolished qualified immunity in Roe v. Doe, 605 U.S. 217 (2025)' -- a
    plausible-looking but nonexistent citation. CourtListener 404s it ->
    citation_status NOT_FOUND -> publication_status EXCLUDED -> ineligible ->
    never reaches the eligible set the gate freezes for render."""

    async def test_fabricated_cite_excluded_end_to_end(self, monkeypatch):
        async def _fake_lookup(text, *, token=None):
            # The two claims' cites resolve differently -- the real one is found,
            # the fabricated one 404s -- exactly what CourtListener would report.
            if "605 U.S. 217" in text:
                return [{"citation": "605 U.S. 217", "status": 404}]
            return [{"citation": "457 U.S. 800", "status": 200}]

        monkeypatch.setattr("dr_core.graph.verify.lookup_citations", _fake_lookup)
        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", _no_evidence)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeCleanModel())

        real_claim = Claim(
            claim_id="c-real",
            text="Officials get qualified immunity under Harlow v. Fitzgerald, 457 U.S. 800 (1982).",
            importance=4,
            source_id="s1",
        )
        fabricated_claim = Claim(
            claim_id="c-fabricated",
            text="The Supreme Court abolished qualified immunity in Roe v. Doe, 605 U.S. 217 (2025).",
            importance=5,
            source_id="s1",
        )
        dr_claims = {c.claim_id: c.model_dump(mode="json") for c in (real_claim, fabricated_claim)}
        dr_sources = {"s1": _source("s1")}
        state = {"dr_claims": dr_claims, "dr_sources": dr_sources, "dr_run": {"depth": "quick"}}

        result = await verify_node(state)
        merged_claims = merge_ledger(dr_claims, result.get("dr_claims"))

        fabricated = Claim.model_validate(merged_claims["c-fabricated"])
        assert fabricated.citation_status == CitationStatus.NOT_FOUND
        assert derive_status(fabricated) == PublicationStatus.EXCLUDED
        assert ineligibility_reason(fabricated, dr_sources) == "excluded"

        # Full gate pass: force the retry cap so the run proceeds to render even
        # though the real claim alone should already make it eligible -- the
        # fabricated one must never appear in the frozen citation_ordinals map.
        gate_state = {"dr_claims": merged_claims, "dr_sources": dr_sources, "dr_run": {"gate_retries": 0}}
        gate_result = eligibility_gate(gate_state)
        citation_ordinals = gate_result["dr_run"]["citation_ordinals"]
        assert "c-fabricated" not in citation_ordinals
        assert "c-real" in citation_ordinals


class TestFinancialStaleVintageTrap:
    """injected_claims[1] from the financial_vintage fixture: a real FY2023
    Apple net-sales figure ($383.285B) mislabeled as FY2024 -- data_ref's actual
    `period` (2023-09-30) disagrees with the claim's `claimed_period`
    (2024-09-30). The provenance audit catches this deterministically (no vote
    needed) -> MISMATCH -> EXCLUDED."""

    async def test_stale_vintage_excluded_end_to_end(self, monkeypatch):
        async def _no_lookup_needed(text, *, token=None):
            raise AssertionError("this claim's text has no case citation -- CITE_RE must not match it")

        monkeypatch.setattr("dr_core.graph.verify.lookup_citations", _no_lookup_needed)
        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", _no_evidence)
        # A MISMATCH data_provenance is not itself a verification-ladder terminal, so
        # this claim is still non-terminal and gets selected for the vote pass too --
        # mock the vote model so that stays hermetic (no live network/model call).
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeCleanModel())

        stale_claim = Claim(
            claim_id="c-stale",
            text="Apple's FY2024 net sales were $383.285 billion.",
            importance=5,
            source_id="s1",
            data_ref={"value": "383285000000", "period": "2023-09-30", "claimed_period": "2024-09-30", "source_class": "primary_filing"},
        )
        dr_claims = {stale_claim.claim_id: stale_claim.model_dump(mode="json")}
        dr_sources = {"s1": _source("s1", source_system="edgar", authority_tier=1)}
        state = {"dr_claims": dr_claims, "dr_sources": dr_sources, "dr_run": {"depth": "quick"}}

        result = await verify_node(state)
        merged_claims = merge_ledger(dr_claims, result.get("dr_claims"))

        stale = Claim.model_validate(merged_claims["c-stale"])
        assert stale.data_provenance == DataProvenance.MISMATCH
        assert derive_status(stale) == PublicationStatus.EXCLUDED

        gate_state = {"dr_claims": merged_claims, "dr_sources": dr_sources, "dr_run": {"gate_retries": 2}}
        gate_result = eligibility_gate(gate_state)
        assert gate_result["dr_run"]["gate_decision"] == "render"
        assert "c-stale" not in gate_result["dr_run"]["citation_ordinals"]

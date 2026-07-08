"""Tests for dr_core.graph.verify.verify_node (D8) and its wiring into make_dr_agent.
Hermetic: the vote model and evidence/citation fetchers are monkeypatched; no
network, no real LLM. Folds real merge_ledger over the node's output to prove the
write discipline (advance-only, idempotent re-entry).
"""

import json
from types import SimpleNamespace

from dr_core.graph.agent import make_dr_agent
from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.graph.state import merge_ledger
from dr_core.graph.verify import verify_node
from dr_core.models.enums import CitationStatus, VerificationStatus
from dr_core.models.ledger import Claim
from dr_core.verify import votes as votes_mod
from langchain.agents import AgentState
from langgraph.graph import END, StateGraph


def _claim(**overrides) -> dict:
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults).model_dump(mode="json")


def _source(source_id: str = "s1", **overrides) -> dict:
    defaults = dict(id=source_id, url_or_id=f"https://example.com/{source_id}", source_system="web", authority_tier=1, retrieved_at="2026-07-06T00:00:00+00:00")
    defaults.update(overrides)
    return defaults


def _clean_vote_response() -> SimpleNamespace:
    content = json.dumps({"refuted": False, "abstain": False, "confidence": "high", "reasoning": "ok"})
    return SimpleNamespace(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cache_read": 0, "cache_creation": 0}})


class _FakeModel:
    async def ainvoke(self, messages):
        return _clean_vote_response()


async def _no_evidence(url):
    return None


class TestCitationGateSkipsVotes:
    async def test_not_found_citation_claim_never_reaches_the_vote_model(self, monkeypatch):
        def _boom_get_model():
            raise AssertionError("vote model must not be called for a NOT_FOUND claim")

        monkeypatch.setattr(votes_mod, "_get_vote_model", _boom_get_model)
        monkeypatch.setattr("dr_core.graph.verify.lookup_citations", _fake_lookup_not_found)
        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", _no_evidence)

        state = {
            "dr_claims": {"c1": _claim(text="The Court held in Roe v. Doe, 605 U.S. 217 (2025) that ...")},
            "dr_sources": {"s1": _source()},
            "dr_run": {"depth": "quick"},
        }
        result = await verify_node(state)
        touched = result["dr_claims"]["c1"]
        assert touched["citation_status"] == CitationStatus.NOT_FOUND.value
        assert touched["verification"]["status"] == VerificationStatus.PENDING.value  # untouched by votes


async def _fake_lookup_not_found(text, *, token=None):
    return [{"citation": "605 U.S. 217", "status": 404}]


class TestVerifyNodeVotesAndMergesLegally:
    async def test_clean_claim_votes_supported_and_folds_through_merge_ledger(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel())
        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", lambda url: _return_excerpt())

        dr_claims = {"c1": _claim()}
        state = {"dr_claims": dr_claims, "dr_sources": {"s1": _source()}, "dr_run": {"depth": "quick"}}
        result = await verify_node(state)

        assert result["dr_run"]["verify_mode"] == "adaptive_1_3"
        assert result["dr_run"]["verify_usage"]["calls"] == 1

        merged = merge_ledger(dr_claims, result.get("dr_claims"))
        assert merged["c1"]["verification"]["status"] == VerificationStatus.SUPPORTED.value
        assert merged["c1"]["verification"]["complete"] is True

    async def test_idempotent_rerun_is_a_no_op(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel())
        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", lambda url: _return_excerpt())

        dr_claims = {"c1": _claim()}
        state = {"dr_claims": dr_claims, "dr_sources": {"s1": _source()}, "dr_run": {"depth": "quick"}}
        result1 = await verify_node(state)
        merged = merge_ledger(dr_claims, result1.get("dr_claims"))
        assert merged["c1"]["verification"]["complete"] is True

        state2 = {"dr_claims": merged, "dr_sources": {"s1": _source()}, "dr_run": {"depth": "quick"}}
        result2 = await verify_node(state2)
        assert result2.get("dr_claims", {}) == {}  # complete claim was filtered out of `active` entirely
        merged2 = merge_ledger(merged, result2.get("dr_claims"))
        assert merged2 == merged  # no-op, and merge_ledger did not raise


async def _return_excerpt():
    return "some fetched excerpt text"


class TestVerifyNodeGraphWiring:
    def test_research_routes_through_verify_to_eligibility_gate(self, monkeypatch):
        class _Stub(AgentState):
            pass

        def _passthrough(state):
            return {}

        inner = StateGraph(_Stub)
        inner.add_node("noop", _passthrough)
        inner.set_entry_point("noop")
        inner.add_edge("noop", END)
        stub_subgraph = inner.compile()

        def _fake_make_lead_agent(config, *, app_config, extra_middlewares=None):
            assert isinstance(extra_middlewares[0], DrLedgerMiddleware)
            return stub_subgraph

        monkeypatch.setattr("dr_core.graph.agent._make_lead_agent", _fake_make_lead_agent)

        compiled = make_dr_agent(config={}, app_config=object())

        assert "verify" in compiled.nodes

        branches = compiled.builder.branches.get("research", {})
        assert branches, "expected a conditional edge registered on research"
        branch_spec = next(iter(branches.values()))
        assert branch_spec.ends.get("gate") == "verify"

        assert ("verify", "eligibility_gate") in compiled.builder.edges

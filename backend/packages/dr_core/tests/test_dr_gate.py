"""Tests for the real eligibility_gate + corrective loop + route_after_gate (D6
rulings B, C, D, E). Hermetic: no model/network, drives the node functions
directly with synthetic state dicts.
"""

import hashlib
import json

from langchain.agents import AgentState
from langgraph.graph import END, StateGraph

from dr_core.graph.agent import make_dr_agent, route_after_gate
from dr_core.graph.gate import eligibility_gate
from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.models.enums import CitationStatus, StopReason
from dr_core.models.ledger import Claim


def _sig(eligible_ids: list[str], n_claims: int, n_sources: int) -> str:
    """Mirrors gate.py's own sig formula, for tests that need to pre-seed it."""
    return hashlib.sha256(json.dumps([sorted(eligible_ids), n_claims, n_sources], sort_keys=True).encode()).hexdigest()


def _claim(claim_id: str, source_id: str = "s1", **overrides) -> dict:
    defaults = dict(claim_id=claim_id, text=f"claim {claim_id}", importance=4, source_id=source_id)
    defaults.update(overrides)
    return Claim(**defaults).model_dump(mode="json")


def _source(source_id: str) -> dict:
    return {"id": source_id, "url_or_id": f"https://example.com/{source_id}", "source_system": "web", "authority_tier": 3, "retrieved_at": "2026-07-06T00:00:00+00:00"}


class TestEligibleClaimProceeds:
    def test_single_eligible_claim_renders_with_frozen_ordinal(self):
        state = {
            "dr_claims": {"c1": _claim("c1", source_id="s1")},
            "dr_sources": {"s1": _source("s1")},
            "dr_run": {},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["citation_ordinals"] == {"c1": 1}
        assert "stop_reason" not in dr_run
        assert route_after_gate({"dr_run": dr_run}) == "render"


class TestOnlyIneligibleRetries:
    def test_excluded_claim_bounces_a_corrective_retry(self):
        excluded_claim = _claim("c1", source_id="s1", citation_status=CitationStatus.NOT_FOUND)
        state = {
            "dr_claims": {"c1": excluded_claim},
            "dr_sources": {"s1": _source("s1")},
            "dr_run": {"gate_retries": 0},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "research"
        assert dr_run["gate_retries"] == 1
        assert "gate_ledger_sig" in dr_run
        messages = result["messages"]
        assert len(messages) == 1
        corrective = messages[0]
        assert corrective.additional_kwargs.get("hide_from_ui") is True
        assert "<dr_corrective>" in corrective.content
        assert route_after_gate({"dr_run": dr_run}) == "research"


class TestCapHitForcesRender:
    def test_retries_exhausted_forces_render_with_open_requirements(self):
        state = {
            "dr_claims": {"c1": _claim("c1", source_id="s1", citation_status=CitationStatus.NOT_FOUND)},
            "dr_sources": {"s1": _source("s1")},
            "dr_run": {"gate_retries": 2},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["stop_reason"] == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        assert route_after_gate({"dr_run": dr_run}) == "render"


class TestBreakerFiresOnUnchangedSignature:
    def test_one_strike_breaker_forces_render_before_cap(self):
        dr_claims = {"c1": _claim("c1", source_id="s1", citation_status=CitationStatus.NOT_FOUND)}
        dr_sources = {"s1": _source("s1")}
        # Eligible set is empty (the claim is excluded) -- precompute the sig the
        # gate itself will compute for this exact state, so retries==1 with a
        # matching prior sig proves the ONE-STRIKE breaker, not the retries==2 cap.
        prior_sig = _sig([], len(dr_claims), len(dr_sources))
        state = {
            "dr_claims": dr_claims,
            "dr_sources": dr_sources,
            "dr_run": {"gate_retries": 1, "gate_ledger_sig": prior_sig},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["stop_reason"] == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        assert route_after_gate({"dr_run": dr_run}) == "render"


class TestOrdinalFreezeOrder:
    def test_three_eligible_claims_freeze_in_insertion_order(self):
        state = {
            "dr_claims": {
                "c1": _claim("c1", source_id="s1"),
                "c2": _claim("c2", source_id="s1"),
                "c3": _claim("c3", source_id="s1"),
            },
            "dr_sources": {"s1": _source("s1")},
            "dr_run": {},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["citation_ordinals"] == {"c1": 1, "c2": 2, "c3": 3}


class TestMissingSourceBackstop:
    def test_eligible_claim_with_missing_source_is_treated_ineligible(self):
        state = {
            "dr_claims": {"c1": _claim("c1", source_id="s-missing")},
            "dr_sources": {},
            "dr_run": {"gate_retries": 0},
        }
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "research"
        assert dr_run["gate_retries"] == 1


class TestMakeDrAgentGateWiring:
    def test_gate_conditional_edge_routes_to_research_and_render(self, monkeypatch):
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

        branches = compiled.builder.branches.get("eligibility_gate", {})
        assert branches, "expected a conditional edge registered on eligibility_gate"
        branch_spec = next(iter(branches.values()))
        assert branch_spec.ends == {"research": "research", "render": "render"}

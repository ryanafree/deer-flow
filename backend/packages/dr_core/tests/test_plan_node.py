"""Tests for dr_core.graph.plan.plan_coverage_node (D9). Hermetic: the plan
model is a fake injected via monkeypatching the module-level factories in
dr_core.plan.extraction / dr_core.plan.mapping; no network, no real LLM.
Folds real merge_ledger/merge_run over the node's output to prove turn
scoping and idempotent re-merge.
"""

import json
from types import SimpleNamespace

from dr_core.graph.plan import plan_coverage_node
from dr_core.graph.state import merge_ledger, merge_run
from dr_core.models.ledger import Claim
from dr_core.plan import extraction as extraction_mod
from dr_core.plan import mapping as mapping_mod
from langchain_core.messages import HumanMessage


def _extract_response(payload) -> SimpleNamespace:
    return SimpleNamespace(content=json.dumps(payload), usage_metadata={"input_tokens": 3, "output_tokens": 2, "input_token_details": {}})


class _FakeModel:
    def __init__(self, response):
        self._response = response

    async def ainvoke(self, messages):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _claim_payload(claim_id: str) -> dict:
    return Claim(claim_id=claim_id, text=f"claim {claim_id}", importance=3, source_id="s1").model_dump(mode="json")


class TestTurnScopedExtraction:
    async def test_first_pass_extracts_and_sets_active_ids(self, monkeypatch):
        payload = [{"kind": "entity", "text": "Who is the CEO?", "must_cover": True}]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_extract_response(payload)))
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(SimpleNamespace(content="[]", usage_metadata={})))

        state = {
            "messages": [HumanMessage(content="Who is the CEO?")],
            "dr_run": {},
            "dr_requirements": {},
            "dr_claims": {},
            "dr_coverage": {},
        }
        result = await plan_coverage_node(state)
        assert result["dr_run"]["active_requirement_ids"]
        assert len(result["dr_requirements"]) == 1
        assert "plan_usage" in result["dr_run"]

    async def test_corrective_reentry_skips_extraction_and_preserves_active_ids(self, monkeypatch):
        def _boom():
            raise AssertionError("extraction must not run on a corrective re-entry")

        monkeypatch.setattr(extraction_mod, "_get_plan_model", _boom)
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(SimpleNamespace(content="[]", usage_metadata={})))

        state = {
            "messages": [HumanMessage(content="Who is the CEO?")],
            "dr_run": {"gate_decision": "research", "active_requirement_ids": ["req-abc"]},
            "dr_requirements": {"req-abc": {"id": "req-abc", "kind": "entity", "text": "Who is the CEO?", "must_cover": True, "entities": [], "attempts": 0, "terminal_state": None, "window": None}},
            "dr_claims": {},
            "dr_coverage": {},
        }
        result = await plan_coverage_node(state)
        # active_requirement_ids must be ABSENT from the returned dr_run update
        # (never overwritten), not merely equal -- merge_run is last-writer-wins
        # per key, so presence would still clobber a differently-ordered value.
        assert "active_requirement_ids" not in result["dr_run"]
        merged_run = merge_run(state["dr_run"], result["dr_run"])
        assert merged_run["active_requirement_ids"] == ["req-abc"]
        assert "dr_requirements" not in result


class TestMappingRunsEveryPass:
    async def test_mapping_runs_on_corrective_reentry_too(self, monkeypatch):
        def _boom():
            raise AssertionError("extraction must not run on a corrective re-entry")

        monkeypatch.setattr(extraction_mod, "_get_plan_model", _boom)
        mapping_payload = [{"requirement_id": "req-abc", "claim_id": "c1", "relation": "direct", "elements_satisfied": [], "relationship_stated": None}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_extract_response(mapping_payload)))

        state = {
            "messages": [],
            "dr_run": {"gate_decision": "research", "active_requirement_ids": ["req-abc"]},
            "dr_requirements": {"req-abc": {"id": "req-abc", "kind": "entity", "text": "who", "must_cover": True, "entities": [], "attempts": 0, "terminal_state": None, "window": None}},
            "dr_claims": {"c1": _claim_payload("c1")},
            "dr_coverage": {},
        }
        result = await plan_coverage_node(state)
        assert "req-abc:c1" in result["dr_coverage"]


class TestMergeLedgerFold:
    async def test_extraction_is_idempotent_through_merge_ledger(self, monkeypatch):
        payload = [{"kind": "entity", "text": "Who is the CEO?", "must_cover": True}]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_extract_response(payload)))
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(SimpleNamespace(content="[]", usage_metadata={})))

        state = {"messages": [HumanMessage(content="Who is the CEO?")], "dr_run": {}, "dr_requirements": {}, "dr_claims": {}, "dr_coverage": {}}
        result1 = await plan_coverage_node(state)
        merged_requirements = merge_ledger(state["dr_requirements"], result1.get("dr_requirements"))

        # Re-run the SAME question against the merged ledger (simulating a
        # later turn asking the identical thing): same ids, and folding
        # again must be a true no-op, never raise.
        state2 = {"messages": [HumanMessage(content="Who is the CEO?")], "dr_run": {}, "dr_requirements": merged_requirements, "dr_claims": {}, "dr_coverage": {}}
        result2 = await plan_coverage_node(state2)
        merged_requirements_2 = merge_ledger(merged_requirements, result2.get("dr_requirements"))
        assert merged_requirements_2 == merged_requirements


class TestGraphWiring:
    def test_research_routes_through_plan_coverage_then_verify_then_gate(self, monkeypatch):
        from dr_core.graph.agent import make_dr_agent
        from dr_core.graph.middleware import DrLedgerMiddleware
        from langchain.agents import AgentState
        from langgraph.graph import END, StateGraph

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
        node_names = set(compiled.nodes.keys())
        assert {"research", "plan_coverage", "verify", "eligibility_gate", "render"} <= node_names

        branches = compiled.builder.branches.get("research", {})
        branch_spec = next(iter(branches.values()))
        assert branch_spec.ends.get("gate") == "plan_coverage"
        assert ("plan_coverage", "verify") in compiled.builder.edges
        assert ("verify", "eligibility_gate") in compiled.builder.edges

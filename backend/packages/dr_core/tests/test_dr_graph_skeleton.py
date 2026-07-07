"""Structural (walking-skeleton) tests for dr_core.graph — D5 / Fable consult 03.

HERMETIC: no live model, no config.yaml, no network. ``make_dr_agent`` needs a
real ``_make_lead_agent`` subgraph (live config), so its compile test
monkeypatches ``dr_core.graph.agent._make_lead_agent`` with a trivial 1-node
passthrough compiled graph instead of building the real lead agent. The goal
is to prove the OUTER graph's wiring (nodes, entry point, conditional edge),
not to exercise the lead agent itself.
"""

import typing

from langchain.agents import AgentState
from langgraph.graph import END, StateGraph

from dr_core.graph.agent import make_dr_agent, route_after_research
from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.graph.state import DrAgentState, merge_by_id


class TestMergeById:
    def test_merges_disjoint_keys(self):
        assert merge_by_id({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_none_new_keeps_existing(self):
        assert merge_by_id({"a": 1}, None) == {"a": 1}

    def test_none_existing_returns_new(self):
        assert merge_by_id(None, {"a": 1}) == {"a": 1}

    def test_later_write_wins_same_key(self):
        assert merge_by_id({"a": 1}, {"a": 2}) == {"a": 2}


class TestRouteAfterResearch:
    """D7: turn-scoped routing. Gates iff dr_run["deliverable"] is true OR
    dr_run["gate_decision"] == "research" -- never on dr_claims, since claims
    persist across turns in thread state."""

    def test_empty_state_routes_to_end(self):
        assert route_after_research({}) == END

    def test_deliverable_marker_routes_to_gate(self):
        assert route_after_research({"dr_run": {"deliverable": True}}) == "gate"

    def test_empty_claims_and_no_marker_routes_to_end(self):
        assert route_after_research({"dr_claims": {}, "dr_run": {"deliverable": False}}) == END

    def test_zero_claim_search_turn_with_deliverable_marker_routes_to_gate(self):
        """B2(a): a research turn that searched but recorded zero claims must
        still reach the gate so its no-claims corrective is reachable."""
        assert route_after_research({"dr_claims": {}, "dr_run": {"deliverable": True}}) == "gate"

    def test_stale_claims_without_deliverable_marker_route_to_end(self):
        """B2(b): dr_claims persisting from a prior research turn must NOT
        drag a later ordinary chat turn (no deliverable this turn) into the
        gate -- this is the post-render chat-turn leak D7 closes."""
        state = {"dr_claims": {"c1": {"id": "c1"}}, "dr_run": {"deliverable": False, "gate_decision": "render"}}
        assert route_after_research(state) == END

    def test_mid_corrective_loop_routes_to_gate_regardless_of_deliverable(self):
        """B2/D6: gate_decision == "research" is the corrective re-entry
        marker and must gate even if deliverable is absent or false."""
        assert route_after_research({"dr_run": {"gate_decision": "research"}}) == "gate"
        assert route_after_research({"dr_run": {"gate_decision": "research", "deliverable": False}}) == "gate"


class TestDrLedgerMiddleware:
    def test_state_schema_is_dr_agent_state(self):
        assert DrLedgerMiddleware().state_schema is DrAgentState

    def test_dr_channels_are_annotated_with_a_reducer(self):
        hints = typing.get_type_hints(DrAgentState, include_extras=True)
        dr_channels = {"dr_sources", "dr_claims", "dr_requirements", "dr_coverage", "dr_run"}
        assert dr_channels <= hints.keys()
        for name in dr_channels:
            hint = hints[name]
            assert typing.get_origin(hint) is typing.Annotated
            # Annotated[<type>, <reducer>, ...] — at least one non-type metadata entry (the reducer).
            metadata = hint.__metadata__
            assert len(metadata) >= 1
            assert callable(metadata[0])

    def test_dr_agent_state_extends_agent_state(self):
        # TypedDict inheritance doesn't support issubclass()/MRO checks; the
        # structural signal that matters is that AgentState's own fields
        # (e.g. "messages") merge into DrAgentState's type hints.
        hints = typing.get_type_hints(DrAgentState, include_extras=True)
        assert "messages" in hints


def _make_trivial_compiled_subgraph():
    """A 1-node passthrough compiled StateGraph standing in for the real lead agent."""

    class _Stub(AgentState):
        pass

    def _passthrough(state):
        return {}

    inner = StateGraph(_Stub)
    inner.add_node("noop", _passthrough)
    inner.set_entry_point("noop")
    inner.add_edge("noop", END)
    return inner.compile()


class TestMakeDrAgentCompiles:
    def test_outer_graph_compiles_with_stubbed_research_subgraph(self, monkeypatch):
        stub_subgraph = _make_trivial_compiled_subgraph()

        def _fake_make_lead_agent(config, *, app_config, extra_middlewares=None):
            # Confirms make_dr_agent threads extra_middlewares (D5-B) without
            # requiring a live app_config/model.
            assert extra_middlewares is not None
            assert isinstance(extra_middlewares[0], DrLedgerMiddleware)
            return stub_subgraph

        monkeypatch.setattr("dr_core.graph.agent._make_lead_agent", _fake_make_lead_agent)

        compiled = make_dr_agent(config={}, app_config=object())

        node_names = set(compiled.nodes.keys())
        assert {"research", "eligibility_gate", "render"} <= node_names

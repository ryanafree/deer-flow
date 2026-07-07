"""make_dr_agent — the D5 gated outer-graph wrap, now with the D6 corrective loop.

A thin ``StateGraph(DrOuterState)``: node "research" (the DeerFlow lead-agent
subgraph, carrying ``DrLedgerMiddleware``) -> conditional edge -> node
"eligibility_gate" -> conditional edge -> node "research" (corrective retry)
or node "render" (stub) -> END. See D5 in DECISIONS.md for the
research-routing design caveat and D6 for the gate/loop policy this graph
implements.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from deerflow.agents.lead_agent.agent import _make_lead_agent
from deerflow.config.app_config import get_app_config

from dr_core.graph.gate import eligibility_gate
from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.graph.render import render_node
from dr_core.graph.state import DrOuterState


def route_after_research(state) -> str:
    """Route to the gate only when the turn produced a research deliverable.

    Clarification / ordinary chat turns (no dr_claims, no dr_run deliverable
    marker) MUST reach END directly — this is the mandatory D5 design caveat:
    ClarificationMiddleware's ``Command(goto=END)`` ends the subgraph, and the
    outer graph must not route that into gate+render.
    """
    dr_claims = state.get("dr_claims") or {}
    dr_run = state.get("dr_run") or {}
    if dr_claims or dr_run.get("deliverable"):
        return "gate"
    return END


def route_after_gate(state) -> str:
    """Route on the gate's own decision (D6: the decision belongs to the gate,
    written into ``dr_run`` and visible in checkpoints). Never recompute
    eligibility here -- read what ``eligibility_gate`` decided.
    """
    decision = (state.get("dr_run") or {}).get("gate_decision")
    if decision == "research":
        return "research"
    return "render"


def make_dr_agent(config, app_config=None):
    """LangGraph graph factory for the dr-assurance agent.

    ``app_config`` is an explicit kwarg (not inferred) because
    ``runtime/runs/worker.py`` inspects the factory signature for it.
    """
    resolved_app_config = app_config or get_app_config()

    research_agent = _make_lead_agent(
        config,
        app_config=resolved_app_config,
        extra_middlewares=[DrLedgerMiddleware()],
    )

    graph = StateGraph(DrOuterState)
    graph.add_node("research", research_agent)
    graph.add_node("eligibility_gate", eligibility_gate)
    graph.add_node("render", render_node)

    graph.set_entry_point("research")
    graph.add_conditional_edges("research", route_after_research, {"gate": "eligibility_gate", END: END})
    graph.add_conditional_edges("eligibility_gate", route_after_gate, {"research": "research", "render": "render"})
    graph.add_edge("render", END)

    return graph.compile()

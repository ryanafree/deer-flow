"""make_dr_agent — the D5 gated outer-graph wrap, now with the D6 corrective loop,
the D8 verification layer, the D9 requirement/coverage planning layer, and the
D11 initialize/entry-router layer.

A thin ``StateGraph(DrOuterState)``: node "initialize" (the D11 run-scoping
entry node; deterministic, no LLM) -> conditional edge: a preset
``deliverable`` routes "plan_requirements" (extract + schedule BEFORE any
retrieval -- the benchmark/headless wiring) -> "research"; otherwise straight
to "research" (the DeerFlow lead-agent subgraph, carrying
``DrLedgerMiddleware`` -- the D7 interactive wiring, unchanged) ->
conditional edge -> node "plan_coverage" (the D9 requirement extraction /
coverage mapping pass) -> node "verify" (the D8 citation gate / provenance
audit / adaptive vote pass) -> node "eligibility_gate" -> conditional edge ->
node "research" (planned corrective retry) or node "render" -> END. See D5 in
DECISIONS.md for the research-routing design caveat, D6 for the gate/loop
policy this graph implements, D7 for the turn-scoped ``deliverable`` marker
that decides ``route_after_research``, D8 for the verify node's topology, D9
for the planning node's turn-scoped extraction + every-pass mapping (the
corrective loop re-enters through "plan_coverage" -> "verify" too -- safe
because both skip already-settled work), and D11 for the entry router / run
scoping / planned-retry rulings.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from deerflow.agents.lead_agent.agent import _make_lead_agent
from deerflow.config.app_config import get_app_config
from dr_core.connectors.tools import DrConnectorToolsMiddleware
from dr_core.graph.directive_middleware import DrResearchDirectiveMiddleware
from dr_core.graph.gate import eligibility_gate
from dr_core.graph.initialize import initialize_node, route_after_initialize
from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.graph.plan import plan_coverage_node, plan_requirements_node
from dr_core.graph.profile_middleware import DrProfileToolMiddleware
from dr_core.graph.render import render_node
from dr_core.graph.state import DrOuterState
from dr_core.graph.verify import verify_node


def route_after_research(state) -> str:
    """Route to the gate only when the turn produced a research deliverable.

    D7 (turn-scoped routing): routes to "gate" iff ``dr_run["deliverable"]``
    is true (set by ``DrLedgerMiddleware``'s C2 source scan or
    ``record_claim``'s success Command) OR ``dr_run["gate_decision"] ==
    "research"`` (mid-corrective-loop re-entry). This does NOT key on
    dr_claims at all: dr_claims persists across turns in thread state, so a
    claims-based check would drag every later ordinary chat turn on the same
    thread back into the gate. Clarification / ordinary chat turns (neither
    marker set) MUST reach END directly — this is the mandatory D5 design
    caveat: ClarificationMiddleware's ``Command(goto=END)`` ends the
    subgraph, and the outer graph must not route that into gate+render.
    """
    dr_run = state.get("dr_run") or {}
    if dr_run.get("deliverable") or dr_run.get("gate_decision") == "research":
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
        # DrConnectorToolsMiddleware contributes Stage-C typed tools (WRDS
        # first) the same way DrLedgerMiddleware contributes record_claim --
        # via its `tools` class attribute, no hooks, so its position in this
        # list doesn't affect wrap_tool_call composition. DrProfileToolMiddleware
        # last: langchain's factory composes wrap_tool_call handlers so
        # earlier entries end up outer / later entries closer to the actual
        # tool call -- profile enforcement should sit as close to execution
        # as this list lets it (S9-B), and it must see the Stage-C tools
        # already bound so it can filter/deny them by profile.
        # DrResearchDirectiveMiddleware (SPEC_evidence_routing_2026-07-10.md
        # Stage 3) only reads dr_run/profile state and merges its directive
        # straight into request.system_message -- it neither depends on nor
        # affects tool-call wrapping order, so its position here is arbitrary
        # relative to the other three; appended last (rather than first) so
        # extra_middlewares[0] stays DrLedgerMiddleware, which existing tests
        # (test_dr_gate.py, test_dr_graph_skeleton.py, test_plan_node.py,
        # test_verify_node.py) assert on directly.
        extra_middlewares=[
            DrLedgerMiddleware(),
            DrConnectorToolsMiddleware(),
            DrProfileToolMiddleware(),
            DrResearchDirectiveMiddleware(),
        ],
    )

    graph = StateGraph(DrOuterState)
    graph.add_node("initialize", initialize_node)
    graph.add_node("plan_requirements", plan_requirements_node)
    graph.add_node("research", research_agent)
    graph.add_node("plan_coverage", plan_coverage_node)
    graph.add_node("verify", verify_node)
    graph.add_node("eligibility_gate", eligibility_gate)
    graph.add_node("render", render_node)

    # D11 item 4: initialize is the entry on EVERY invocation (run scoping),
    # and hosts the entry decision -- a preset deliverable plans before any
    # retrieval; the interactive path enters research-first exactly as D7/D9
    # ruled.
    graph.set_entry_point("initialize")
    graph.add_conditional_edges("initialize", route_after_initialize, {"plan": "plan_requirements", "research": "research"})
    graph.add_edge("plan_requirements", "research")
    # route_after_research's returned keys are unchanged ("gate" | END); the
    # wiring target for "gate" is now plan_coverage (D9), ahead of verify (D8).
    graph.add_conditional_edges("research", route_after_research, {"gate": "plan_coverage", END: END})
    graph.add_edge("plan_coverage", "verify")
    graph.add_edge("verify", "eligibility_gate")
    graph.add_conditional_edges("eligibility_gate", route_after_gate, {"research": "research", "render": "render"})
    graph.add_edge("render", END)

    return graph.compile()

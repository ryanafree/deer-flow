import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "dr_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "harness"))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402

from deerflow.config.app_config import get_app_config  # noqa: E402
from deerflow.runtime.runs.worker import _build_runtime_context, _install_runtime_context  # noqa: E402
from dr_core.graph.agent import make_dr_agent  # noqa: E402

QUESTION = (
    "Provide a research report investigating the current SOTA/edge of understanding regarding "
    "volatility, including equity price and options volatility, beta vs IV correlations, "
    "anticipated volatility disagreements as signals, AI/ML models of the character/driver/"
    "sources-of-alpha, etc. Include both the academic finance literature as well as well-respected "
    "thinkers and investment/finance professionals in the area. What are people investigating, what "
    "are they building/testing, and what new insights are being gained at the frontier right now?"
)


async def main():
    app_config = get_app_config()
    agent = make_dr_agent(config={}, app_config=app_config)
    initial_state = {
        "messages": [HumanMessage(content=QUESTION)],
        "dr_run": {"profile": "financial"},
    }
    thread_id = "q4-diag-" + str(uuid.uuid4())[:8]
    run_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id, "model_name": "or-mid"},
        "recursion_limit": 200,
    }
    runtime_ctx = _build_runtime_context(thread_id, run_id, None, app_config)
    _install_runtime_context(config, runtime_ctx)
    runtime = Runtime(context=runtime_ctx, store=None)
    config["configurable"]["__pregel_runtime"] = runtime

    result = await agent.ainvoke(initial_state, config=config)
    for m in result.get("messages", []):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                print(f"TOOL_CALL name={tc.get('name')} args={str(tc.get('args'))[:300]}")
        if isinstance(m, ToolMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"TOOL_MSG name={m.name} content_head={content[:400]!r}")
    print("dr_run:", result.get("dr_run"))


if __name__ == "__main__":
    asyncio.run(main())

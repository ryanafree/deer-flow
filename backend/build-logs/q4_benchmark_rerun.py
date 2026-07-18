"""q.4 volatility benchmark re-run (2026-07-10), headless, mirrors the S6/S10 pattern:
make_dr_agent + real live connectors, single ainvoke, financial profile.
Not committed; scratch script for the benchmark stage contract.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "dr_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "harness"))

import uuid  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402
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
    runs_dir = "/Users/ryanfree/Documents/Projects/deerflow-dr/runs"
    before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()

    app_config = get_app_config()
    thread_id = "q4-benchmark-rerun-2026-07-10b"
    run_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id, "model_name": "or-mid"},
        "recursion_limit": 200,
    }
    # make_dr_agent resolves+freezes the research model from THIS config at
    # construction time (mirrors runtime/runs/worker.py calling the factory
    # with the real per-run config) -- passing config={} here (as the first
    # attempt did) silently falls back to config.yaml's first model.
    agent = make_dr_agent(config=config, app_config=app_config)
    initial_state = {
        "messages": [HumanMessage(content=QUESTION)],
        "dr_run": {"profile": "financial"},
    }
    runtime_ctx = _build_runtime_context(thread_id, run_id, None, app_config)
    _install_runtime_context(config, runtime_ctx)
    runtime = Runtime(context=runtime_ctx, store=None)
    config["configurable"]["__pregel_runtime"] = runtime

    result = await agent.ainvoke(initial_state, config=config)

    after = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    new_run_folders = sorted(after - before)
    dr_run = result.get("dr_run") or {}
    summary = {
        "new_run_folders": new_run_folders,
        "run_dir": os.path.join(runs_dir, new_run_folders[0]) if new_run_folders else None,
        "gate_decision": dr_run.get("gate_decision"),
        "stop_reason": dr_run.get("stop_reason"),
        "gate_retries": dr_run.get("gate_retries"),
        "n_sources": len(result.get("dr_sources") or {}),
        "n_claims": len(result.get("dr_claims") or {}),
        "n_requirements": len(result.get("dr_requirements") or {}),
        "requirements_covered": (dr_run.get("loop") or {}).get("requirements_covered"),
        "requirements_must_cover": (dr_run.get("loop") or {}).get("requirements_must_cover"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

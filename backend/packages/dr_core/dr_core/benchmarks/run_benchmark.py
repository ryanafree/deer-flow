"""Headless dr_core benchmark runner (spec Stage 4, `SPEC_evidence_routing_2026-07-10.md`).

Promoted from the scratch script `build-logs/q4_benchmark_rerun.py`, retaining its
model-freeze fix: `make_dr_agent` resolves and freezes the research model from
`config` at graph-construction time, not at `ainvoke()` time, so `model_name` must
already be set in `config["configurable"]` before `make_dr_agent()` is called —
passing `config={}` silently falls back to config.yaml's first model
(see BENCHMARK_COMPARISON_2026-07-10.md's discarded-run writeup).

`dr_run["deliverable"]` is force-set True here as a benchmark-harness workaround.
No headless or Gateway dr-invocation path sets it today, so a first-turn run that
makes zero tool calls never reaches the eligibility gate (BUILD_LEDGER.md's Stage 1
entry, "Investigation: why the 2026-07-10 diagnostic run bypassed the gate"). This
script sets the precondition explicitly so the gate/coverage-retry feature under
test is actually exercised; it does not fix the production gap.
"""

import argparse
import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "packages", "dr_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "packages", "harness"))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402

from deerflow.config.app_config import get_app_config  # noqa: E402
from deerflow.runtime.runs.worker import _build_runtime_context, _install_runtime_context  # noqa: E402
from dr_core.graph.agent import make_dr_agent  # noqa: E402

DEFAULT_RUNS_DIR = os.environ.get("DR_RUNS_DIR") or os.path.expanduser("~/Documents/Projects/deerflow-dr/runs")


async def run_benchmark(
    question: str,
    profile: str = "financial",
    model_name: str = "or-mid",
    depth: str = "standard",
    thread_id: str | None = None,
    runs_dir: str = DEFAULT_RUNS_DIR,
) -> dict:
    thread_id = thread_id or f"benchmark-{uuid.uuid4().hex[:8]}"
    before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()

    app_config = get_app_config()
    run_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id, "model_name": model_name},
        "recursion_limit": 200,
    }
    agent = make_dr_agent(config=config, app_config=app_config)
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "dr_run": {"deliverable": True, "profile": profile, "depth": depth},
    }
    runtime_ctx = _build_runtime_context(thread_id, run_id, None, app_config)
    _install_runtime_context(config, runtime_ctx)
    runtime = Runtime(context=runtime_ctx, store=None)
    config["configurable"]["__pregel_runtime"] = runtime

    result = await agent.ainvoke(initial_state, config=config)

    after = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    new_run_folders = sorted(after - before)
    dr_run = result.get("dr_run") or {}
    return {
        "thread_id": thread_id,
        "depth": depth,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--profile", default="financial")
    parser.add_argument("--model", default="or-mid")
    parser.add_argument("--depth", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()

    summary = asyncio.run(
        run_benchmark(
            question=args.question,
            profile=args.profile,
            model_name=args.model,
            depth=args.depth,
            thread_id=args.thread_id,
            runs_dir=args.runs_dir,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Headless dr_core benchmark runner (spec Stage 4, `SPEC_evidence_routing_2026-07-10.md`).

Promoted from the scratch script `build-logs/q4_benchmark_rerun.py`, retaining its
model-freeze fix: `make_dr_agent` resolves and freezes the research model from
`config` at graph-construction time, not at `ainvoke()` time, so `model_name` must
already be set in `config["configurable"]` before `make_dr_agent()` is called —
passing `config={}` silently falls back to config.yaml's first model
(see BENCHMARK_COMPARISON_2026-07-10.md's discarded-run writeup).

`dr_run["deliverable"]` is set explicitly because every benchmark cell is a
deliverable run. The production Gateway can set the equivalent
`configurable["dr_deliverable"]`; initialization normalizes both paths.
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


def _load_manifest(run_dir: str | None) -> dict:
    if not run_dir:
        return {}
    path = os.path.join(run_dir, "manifest.json")
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def build_summary(*, result: dict, thread_id: str, depth: str, runs_dir: str, new_run_folders: list[str]) -> dict:
    dr_run = result.get("dr_run") or {}
    run_dir = dr_run.get("run_dir")
    if not run_dir and len(new_run_folders) == 1:
        run_dir = os.path.join(runs_dir, new_run_folders[0])
    manifest = _load_manifest(run_dir)
    coverage = manifest.get("coverage") or {}
    final_coverage = coverage.get("final") or {}
    tool_calls = manifest.get("tool_calls") or {}
    return {
        "thread_id": thread_id,
        "depth": depth,
        "new_run_folders": new_run_folders,
        "run_dir": run_dir,
        "gate_decision": dr_run.get("gate_decision"),
        "stop_reason": dr_run.get("stop_reason"),
        "gate_retries": manifest.get("gate_retries", dr_run.get("gate_retries")),
        "tool_calls_total": tool_calls.get("total"),
        "tool_calls_by_name": tool_calls.get("by_name") or {},
        "n_sources": len(result.get("dr_sources") or {}),
        "n_claims": len(result.get("dr_claims") or {}),
        "n_requirements": len(result.get("dr_requirements") or {}),
        "requirements_covered": final_coverage.get("requirements_covered", dr_run.get("requirements_covered")),
        "requirements_must_cover": final_coverage.get("requirements_must_cover", dr_run.get("requirements_must_cover")),
        "first_pass_coverage": coverage.get("first_pass") or {},
        "final_coverage": final_coverage,
        "sources_by_evidence_class": manifest.get("sources_by_evidence_class") or {},
    }


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
        "dr_run": {"deliverable": True, "profile": profile, "depth": depth, "runs_dir": runs_dir},
    }
    runtime_ctx = _build_runtime_context(thread_id, run_id, None, app_config)
    _install_runtime_context(config, runtime_ctx)
    runtime = Runtime(context=runtime_ctx, store=None)
    config["configurable"]["__pregel_runtime"] = runtime

    result = await agent.ainvoke(initial_state, config=config)

    after = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    new_run_folders = sorted(after - before)
    return build_summary(
        result=result,
        thread_id=thread_id,
        depth=depth,
        runs_dir=runs_dir,
        new_run_folders=new_run_folders,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument("--question")
    question_group.add_argument("--question-file")
    parser.add_argument("--profile", default="financial")
    parser.add_argument("--model", default="or-mid")
    parser.add_argument("--depth", choices=("quick", "standard", "full"), default="standard")
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    args = parser.parse_args()
    question = args.question
    if args.question_file:
        with open(args.question_file) as handle:
            question = handle.read().strip()

    summary = asyncio.run(
        run_benchmark(
            question=question,
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

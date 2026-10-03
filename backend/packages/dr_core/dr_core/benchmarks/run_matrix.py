#!/usr/bin/env python3
"""Persisted multi-cell benchmark driver for the dr_core benchmark matrix.

The 2026-08-19 B1/B2 matrix was driven by a hand-written shell loop that was never
kept; only its `driver.log` survived. This module is that loop, persisted: it runs a
list of cells through `run_benchmark.run_benchmark`, writes a per-cell log, a per-cell
summary, and a per-cell lint capture, and appends the same
`=== START <cell> <ts> ===` / `EXIT=<code> cell=<cell> <ts>` / `ALL DONE` lines to
`driver.log` that the 2026-08-19 log carries.

Three things the old loop did not do, added here because the benchmark readings needed
them and had to be reconstructed by hand:
  - lint is run against each cell's run folder and persisted to `<cell>-lint.json`
    (`report_lint.py` is otherwise a stdout-only CLI, so lint verdicts were never kept);
  - the summary carries the manifest's accounting (tokens and dollar cost) and the model
    the run actually recorded, neither of which `build_summary` reports;
  - a `zero_plan_verify_tokens` flag surfaces cells that passed on the deterministic
    path without spending plan or verify tokens (the B2 case).

Cells run in-process: `run_benchmark` builds its own agent per cell with the model
frozen into `config["configurable"]` before construction, which is the only ordering
constraint that mattered, so no process isolation is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import traceback
from datetime import UTC, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "packages", "dr_core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "packages", "harness"))

from dr_core.benchmarks.run_benchmark import DEFAULT_RUNS_DIR, _load_manifest  # noqa: E402
from dr_core.lint.report_lint import lint_run  # noqa: E402

DEFAULT_MODEL = "or-mid"
DEFAULT_PROFILE = "financial"
DEPTHS = ("quick", "standard", "full")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_name(question_file: str, depth: str) -> str:
    """`.../b1-question.txt` + `standard` -> `b1-standard`, the 2026-08-19 log's names."""
    stem = os.path.splitext(os.path.basename(question_file))[0]
    if stem.endswith("-question"):
        stem = stem[: -len("-question")]
    return f"{stem}-{depth}"


def parse_cell_arg(value: str, *, default_model: str = DEFAULT_MODEL) -> dict:
    """`question_file:depth[:model]` -> a resolved cell dict."""
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"--cell must be question_file:depth[:model], got {value!r}")
    question_file, depth = parts[0], parts[1]
    model = parts[2] if len(parts) == 3 and parts[2] else default_model
    if depth not in DEPTHS:
        raise ValueError(f"--cell depth must be one of {DEPTHS}, got {depth!r}")
    if not question_file:
        raise ValueError(f"--cell needs a question file, got {value!r}")
    return {"name": cell_name(question_file, depth), "question_file": question_file, "depth": depth, "model": model}


def load_spec(path: str, *, default_model: str = DEFAULT_MODEL) -> list[dict]:
    """A spec file is `{"cells": [{"question_file", "depth", "model"?, "name"?}, ...]}`."""
    with open(path) as handle:
        payload = json.load(handle)
    raw_cells = payload.get("cells") if isinstance(payload, dict) else payload
    if not isinstance(raw_cells, list) or not raw_cells:
        raise ValueError(f"spec {path!r} has no cells")
    cells = []
    for entry in raw_cells:
        question_file = entry["question_file"]
        depth = entry.get("depth", "standard")
        if depth not in DEPTHS:
            raise ValueError(f"spec cell depth must be one of {DEPTHS}, got {depth!r}")
        cells.append(
            {
                "name": entry.get("name") or cell_name(question_file, depth),
                "question_file": question_file,
                "depth": depth,
                "model": entry.get("model") or default_model,
            }
        )
    return cells


def resolve_cells(*, cell_args: list[str] | None, spec: str | None, default_model: str = DEFAULT_MODEL) -> list[dict]:
    cells: list[dict] = []
    if spec:
        cells.extend(load_spec(spec, default_model=default_model))
    for value in cell_args or []:
        cells.append(parse_cell_arg(value, default_model=default_model))
    if not cells:
        raise ValueError("no cells: pass --cell question_file:depth[:model] or --spec spec.json")
    return cells


def format_dry_run(cells: list[dict], *, runs_dir: str, logs_dir: str, profile: str) -> str:
    lines = []
    for cell in cells:
        lines.append(
            "cell={name} question_file={question_file} depth={depth} model={model} profile={profile} runs_dir={runs_dir} logs_dir={logs_dir}".format(
                profile=profile,
                runs_dir=runs_dir,
                logs_dir=logs_dir,
                **cell,
            )
        )
    return "\n".join(lines)


def capture_lint(run_dir: str | None) -> dict:
    """Run the deterministic report lint over a run folder and return it as data."""
    if not run_dir or not os.path.isfile(os.path.join(run_dir, "report.md")):
        return {"run_dir": run_dir, "exit_code": None, "status": "no-report", "findings": [], "warnings": [], "n_findings": 0, "n_warnings": 0}
    try:
        findings, warnings = lint_run(run_dir)
    except Exception as exc:  # a lint crash must not lose the cell's result
        return {"run_dir": run_dir, "exit_code": None, "status": f"error: {exc}", "findings": [], "warnings": [], "n_findings": 0, "n_warnings": 0}
    return {
        "run_dir": run_dir,
        "exit_code": 1 if findings else 0,
        "status": "fail" if findings else "pass",
        "findings": findings,
        "warnings": warnings,
        "n_findings": len(findings),
        "n_warnings": len(warnings),
    }


def _phase_tokens(accounting: dict, phase: str) -> int:
    phase_totals = ((accounting.get("phases") or {}).get(phase)) or {}
    return sum(value for value in phase_totals.values() if isinstance(value, (int, float)))


def enrich_summary(summary: dict, *, cell: dict, manifest: dict, lint: dict) -> dict:
    """`build_summary` output plus what the benchmark readings actually needed:
    accounting, the model the manifest recorded, and the lint verdict."""
    accounting = manifest.get("accounting") or {}
    totals = accounting.get("totals") or {}
    enriched = dict(summary)
    enriched.update(
        {
            "cell": cell["name"],
            "question_file": cell["question_file"],
            "requested_model": cell["model"],
            "manifest_model": manifest.get("model"),
            "accounting": accounting,
            "tokens_in": totals.get("input_tokens"),
            "tokens_out": totals.get("output_tokens"),
            "dollar_cost": accounting.get("dollar_cost"),
            "zero_plan_verify_tokens": _phase_tokens(accounting, "plan") == 0 and _phase_tokens(accounting, "verify") == 0,
            "lint_exit_code": lint.get("exit_code"),
            "lint_findings": lint.get("n_findings"),
            "lint_warnings": lint.get("n_warnings"),
        }
    )
    return enriched


def _append(path: str, line: str) -> None:
    with open(path, "a") as handle:
        handle.write(line + "\n")


def run_cell(cell: dict, *, runs_dir: str, logs_dir: str, profile: str, runner=None) -> tuple[int, dict]:
    """Run one cell, persisting `<cell>.log`, `<cell>.json`, and `<cell>-lint.json`.

    `runner` is injected by tests; it defaults to the real single-cell benchmark
    coroutine and is called with the same keyword arguments its CLI passes.
    """
    if runner is None:
        from dr_core.benchmarks.run_benchmark import run_benchmark as runner  # local import: keeps --dry-run agent-free

    with open(cell["question_file"]) as handle:
        question = handle.read().strip()

    log_path = os.path.join(logs_dir, f"{cell['name']}.log")
    exit_code = 0
    summary: dict = {}
    with open(log_path, "w") as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
        try:
            summary = asyncio.run(
                runner(
                    question=question,
                    profile=profile,
                    model_name=cell["model"],
                    depth=cell["depth"],
                    runs_dir=runs_dir,
                )
            )
        except Exception:
            traceback.print_exc()
            exit_code = 1

    manifest = _load_manifest(summary.get("run_dir"))
    lint = capture_lint(summary.get("run_dir"))
    enriched = enrich_summary(summary, cell=cell, manifest=manifest, lint=lint)
    enriched["exit_code"] = exit_code
    with open(os.path.join(logs_dir, f"{cell['name']}-lint.json"), "w") as handle:
        json.dump(lint, handle, indent=2)
    with open(os.path.join(logs_dir, f"{cell['name']}.json"), "w") as handle:
        json.dump(enriched, handle, indent=2)
    return exit_code, enriched


def run_matrix(cells: list[dict], *, runs_dir: str, logs_dir: str, profile: str, runner=None) -> list[dict]:
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)
    driver_log = os.path.join(logs_dir, "driver.log")
    summaries = []
    for cell in cells:
        _append(driver_log, f"=== START {cell['name']} {_now()} ===")
        exit_code, summary = run_cell(cell, runs_dir=runs_dir, logs_dir=logs_dir, profile=profile, runner=runner)
        _append(driver_log, f"EXIT={exit_code} cell={cell['name']} {_now()}")
        summaries.append(summary)
    _append(driver_log, "ALL DONE")
    return summaries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a dr_core benchmark matrix and persist logs, summaries, and lint captures.")
    parser.add_argument("--cell", action="append", default=[], metavar="QUESTION_FILE:DEPTH[:MODEL]", help="repeatable; depth is one of quick|standard|full")
    parser.add_argument("--spec", default=None, help='JSON matrix spec: {"cells": [{"question_file": ..., "depth": ..., "model": ...}]}')
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="default research model for cells that do not name one")
    parser.add_argument("--dry-run", action="store_true", help="print each cell's resolved parameters and exit; constructs no agent and makes no network call")
    args = parser.parse_args(argv)

    cells = resolve_cells(cell_args=args.cell, spec=args.spec, default_model=args.model)
    if args.dry_run:
        print(format_dry_run(cells, runs_dir=args.runs_dir, logs_dir=args.logs_dir, profile=args.profile))
        return 0

    summaries = run_matrix(cells, runs_dir=args.runs_dir, logs_dir=args.logs_dir, profile=args.profile)
    print(json.dumps(summaries, indent=2))
    return 1 if any(s.get("exit_code") for s in summaries) else 0


if __name__ == "__main__":
    sys.exit(main())

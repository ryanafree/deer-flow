"""Tests for dr_core.run.write_run — the run-folder / manifest writer.

Ported from the retired dr.js-era harness/evals/write_run_test.py (DeepResearch project).
The envelope-metric tests are kept close to the original (still exercised via the CLI,
since that unwrapping only happens in main()); the dr.js-hash assertions are replaced
with dr_core-engine-identity assertions, and a new test round-trips a small synthetic
dr_core ledger through write_run() directly, since claims/sources/requirements are now
dr_core.models instances rather than dr.js's plain-object shapes.
"""

import json
import os
import subprocess
import sys

import pytest
from dr_core.models import Claim, Requirement, Source
from dr_core.models.enums import RequirementKind
from dr_core.run.write_run import write_run

WRITER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dr_core", "run", "write_run.py")


def make_source(**overrides):
    defaults = dict(id="s1", url_or_id="https://one.test", source_system="web", title="Source one", authority_tier=1, retrieved_at="2026-07-04T00:00:00-05:00")
    defaults.update(overrides)
    return Source(**defaults)


def make_claim(**overrides):
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


def make_requirement(**overrides):
    defaults = dict(id="r1", kind=RequirementKind.ENTITY, text="Cover entity X", must_cover=True)
    defaults.update(overrides)
    return Requirement(**defaults)


def _cli_env():
    # A bare `uv run pytest` has dr_core/harness on sys.path only via conftest.py's
    # editable-.pth workaround, which a subprocess.run child does not inherit. Set
    # PYTHONPATH explicitly so the CLI is importable regardless of how the outer
    # pytest process was invoked (see conftest.py's UF_HIDDEN .pth note).
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # packages/dr_core
    harness_root = os.path.join(os.path.dirname(pkg_root), "harness")
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (pkg_root, harness_root, existing) if p)
    return env


def run_cli(payload, runs_dir):
    result_path = os.path.join(runs_dir, "result.json")
    with open(result_path, "w") as f:
        json.dump(payload, f)
    proc = subprocess.run(
        [sys.executable, WRITER, "--result", result_path, "--runs-dir", runs_dir, "--model", "test-model"],
        check=True,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    out_dir = proc.stdout.strip().splitlines()[-1]
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    return manifest, out_dir


# ---- claim order is the frozen citation ordinal --------------------------------


def test_claims_jsonl_order_is_frozen_citation_ordinal_and_reloads(tmp_path):
    claims = [make_claim(claim_id="c3", text="third"), make_claim(claim_id="c1", text="first"), make_claim(claim_id="c2", text="second")]
    folder = write_run(
        report_body="# Report\n\nBody [1][2][3].",
        sources=[make_source()],
        claims=claims,
        requirements=[],
        profile="general",
        question="Order test?",
        runs_dir=str(tmp_path),
    )
    lines = open(os.path.join(folder, "claims.jsonl")).read().splitlines()
    assert len(lines) == 3
    # line index (1-based) is the citation ordinal, in the caller's given order — never resorted
    reloaded = [Claim(**json.loads(line)) for line in lines]
    assert [c.claim_id for c in reloaded] == ["c3", "c1", "c2"]
    assert reloaded[0].text == "third"


def test_sources_and_requirements_jsonl_round_trip(tmp_path):
    folder = write_run(
        report_body="report",
        sources=[make_source()],
        claims=[make_claim()],
        requirements=[make_requirement()],
        profile="general",
        question="Round trip?",
        runs_dir=str(tmp_path),
    )
    src_lines = open(os.path.join(folder, "sources.jsonl")).read().splitlines()
    assert Source(**json.loads(src_lines[0])).id == "s1"
    req_lines = open(os.path.join(folder, "requirements.jsonl")).read().splitlines()
    assert Requirement(**json.loads(req_lines[0])).id == "r1"


def test_requirements_jsonl_omitted_when_no_requirements(tmp_path):
    folder = write_run(report_body="report", sources=[], claims=[], requirements=[], profile="general", question="No reqs?", runs_dir=str(tmp_path))
    assert not os.path.exists(os.path.join(folder, "requirements.jsonl"))


def test_sources_fill_missing_retrieved_at_from_dict_input(tmp_path):
    folder = write_run(
        report_body="report",
        sources=[{"id": "s1", "url_or_id": "https://one.test", "source_system": "web", "authority_tier": 2}],
        claims=[],
        requirements=[],
        profile="general",
        question="Fill retrieved_at?",
        runs_dir=str(tmp_path),
    )
    (line,) = open(os.path.join(folder, "sources.jsonl")).read().splitlines()
    assert json.loads(line)["retrieved_at"]


# ---- engine identity: no dr.js coupling -----------------------------------------


def test_manifest_run_env_has_no_dr_js_coupling(tmp_path):
    folder = write_run(report_body="report", sources=[], claims=[], requirements=[], profile="general", question="Engine identity?", runs_dir=str(tmp_path))
    manifest = json.load(open(os.path.join(folder, "manifest.json")))
    run_env = manifest["run_env"]
    assert run_env["engine"] == "dr_core"
    assert "deployed_engine_path" not in run_env
    assert "engine_version_mismatch" not in run_env
    assert "build_stamp" not in run_env
    assert "writer_sha256" in run_env


# ---- runs_dir is a parameter, not a hardcoded DeepResearch path -----------------


def test_runs_dir_param_is_honored(tmp_path):
    runs_dir = tmp_path / "custom-runs"
    folder = write_run(report_body="report", sources=[], claims=[], requirements=[], profile="general", question="Custom runs dir?", runs_dir=str(runs_dir))
    assert str(runs_dir) in folder
    assert "DeepResearch" not in folder


# ---- CLI envelope-unwrap behavior (kept from the original suite) ---------------


def test_cli_actual_metrics_captured_from_workflow_envelope(tmp_path):
    envelope = {
        "result": {
            "report": "report",
            "sources": [],
            "claims": [],
            "manifest": {"profile": "general", "question": "What happened?", "runtime": {"budget": {"agent_call_limit": 80}}},
        },
        "agentCount": 7,
        "totalTokens": 1234,
        "durationMs": 456,
    }
    manifest, _ = run_cli(envelope, str(tmp_path))
    actual = (manifest.get("runtime") or {}).get("actual") or {}
    assert actual.get("agentCount") == 7
    assert actual.get("totalTokens") == 1234
    assert actual.get("totalToolCalls") is None
    assert actual.get("durationMs") == 456
    assert manifest["runtime"]["budget"]["agent_call_limit"] == 80


def test_cli_raw_result_does_not_fabricate_actual_metrics(tmp_path):
    raw_result = {"report": "report", "sources": [], "claims": [], "manifest": {"profile": "general", "question": "Raw result?"}}
    manifest, _ = run_cli(raw_result, str(tmp_path))
    assert "actual" not in (manifest.get("runtime") or {})


def test_missing_profile_and_question_exits(tmp_path):
    result_path = tmp_path / "bad.json"
    result_path.write_text(json.dumps({"report": "x", "manifest": {}}))
    proc = subprocess.run([sys.executable, WRITER, "--result", str(result_path), "--runs-dir", str(tmp_path)], capture_output=True, text=True, env=_cli_env())
    assert proc.returncode != 0
    assert "--profile and --question" in proc.stderr

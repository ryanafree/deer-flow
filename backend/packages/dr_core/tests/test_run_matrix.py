"""Tests for the persisted multi-cell benchmark driver (`dr_core.benchmarks.run_matrix`).

Hermetic: no agent is ever constructed and no network call is made. Cell execution is
driven through the injected `runner` seam with fixture run folders on disk, so the log,
lint, and summary plumbing is exercised against real manifests written by the test.
"""

import json
import os

import pytest
from dr_core.benchmarks import run_matrix
from dr_core.models.enums import VerificationStatus
from dr_core.models.ledger import Claim, SupportRecord, VerificationRecord


def _manifest(**overrides) -> dict:
    manifest = {
        "model": "or-mid",
        "accounting": {
            "phases": {
                "research": {"input_tokens": 100, "output_tokens": 10},
                "verify": {"input_tokens": 0, "output_tokens": 0},
                "plan": {"input_tokens": 0, "output_tokens": 0},
            },
            "totals": {"input_tokens": 100, "output_tokens": 10},
            "dollar_cost": 0.0125,
        },
    }
    manifest.update(overrides)
    return manifest


def _run_folder(tmp_path, name: str, *, report: str, manifest: dict) -> str:
    """A minimal but real run folder: report.md plus a ledger-schema claims.jsonl, so
    the linter's eligibility pass (PART 2) actually activates over the fixture."""
    folder = tmp_path / "runs" / name
    folder.mkdir(parents=True)
    (folder / "report.md").write_text(report)
    (folder / "manifest.json").write_text(json.dumps(manifest))
    claim = Claim(
        claim_id="c1",
        text="The observatory recorded a twelve percent increase in nightly visitors during 2025.",
        importance=3,
        source_id="s1",
        support=SupportRecord(quote="the primary source states this directly", relation_extractor="supports_directly"),
        verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
    )
    (folder / "claims.jsonl").write_text(json.dumps(claim.model_dump(mode="json")) + "\n")
    return str(folder)


CLEAN_REPORT = """# Observatory findings

## Executive summary

The observatory recorded a twelve percent increase in nightly visitors during 2025 [1].

## Conclusion

The increase held across the year.
"""

DIRTY_REPORT = """# Observatory findings

## Executive summary

The observatory recorded a twelve percent increase in nightly visitors during 2025.

## Conclusion

The increase held across the year.
"""


def _question_file(tmp_path, stem: str) -> str:
    path = tmp_path / f"{stem}-question.txt"
    path.write_text("Provide a research report investigating volatility.\n")
    return str(path)


class TestCellResolution:
    def test_parse_cell_arg_derives_the_driver_log_cell_name(self):
        cell = run_matrix.parse_cell_arg("/b/b1-question.txt:standard")
        assert cell == {"name": "b1-standard", "question_file": "/b/b1-question.txt", "depth": "standard", "model": "or-mid"}

    def test_parse_cell_arg_accepts_a_per_cell_model(self):
        assert run_matrix.parse_cell_arg("/b/b2-question.txt:full:claude-top")["model"] == "claude-top"

    def test_parse_cell_arg_rejects_an_unknown_depth(self):
        with pytest.raises(ValueError):
            run_matrix.parse_cell_arg("/b/b1-question.txt:deep")

    def test_spec_file_and_cell_args_both_resolve(self, tmp_path):
        spec = tmp_path / "spec.json"
        spec.write_text(json.dumps({"cells": [{"question_file": "/b/b1-question.txt", "depth": "full"}]}))
        cells = run_matrix.resolve_cells(cell_args=["/b/b2-question.txt:standard"], spec=str(spec), default_model="or-sonnet")
        assert [c["name"] for c in cells] == ["b1-full", "b2-standard"]
        assert {c["model"] for c in cells} == {"or-sonnet"}

    def test_no_cells_is_an_error(self):
        with pytest.raises(ValueError):
            run_matrix.resolve_cells(cell_args=[], spec=None)


class TestDryRun:
    def test_dry_run_prints_resolved_parameters_and_runs_nothing(self, tmp_path, capsys):
        code = run_matrix.main(
            [
                "--cell",
                "/b/b1-question.txt:standard",
                "--cell",
                "/b/b2-question.txt:full:claude-top",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--logs-dir",
                str(tmp_path / "logs"),
                "--dry-run",
            ]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "cell=b1-standard question_file=/b/b1-question.txt depth=standard model=or-mid profile=financial" in out
        assert "cell=b2-full question_file=/b/b2-question.txt depth=full model=claude-top profile=financial" in out
        assert not (tmp_path / "logs").exists()


class TestLintCapture:
    def test_clean_run_folder_captures_exit_zero(self, tmp_path):
        run_dir = _run_folder(tmp_path, "clean", report=CLEAN_REPORT, manifest=_manifest())
        lint = run_matrix.capture_lint(run_dir)
        assert lint["exit_code"] == 0
        assert lint["n_findings"] == 0

    def test_uncited_factual_sentence_captures_exit_one_with_findings(self, tmp_path):
        run_dir = _run_folder(tmp_path, "dirty", report=DIRTY_REPORT, manifest=_manifest())
        lint = run_matrix.capture_lint(run_dir)
        assert lint["exit_code"] == 1
        assert lint["n_findings"] >= 1
        assert any("cite-required" in finding for finding in lint["findings"])

    def test_missing_run_folder_is_recorded_not_raised(self):
        lint = run_matrix.capture_lint(None)
        assert lint["status"] == "no-report"
        assert lint["exit_code"] is None


class TestSummaryEnrichment:
    def test_summary_carries_accounting_model_and_lint(self, tmp_path):
        manifest = _manifest()
        lint = {"exit_code": 1, "n_findings": 2, "n_warnings": 3}
        cell = {"name": "b1-full", "question_file": "/b/b1-question.txt", "depth": "full", "model": "or-mid"}
        enriched = run_matrix.enrich_summary({"run_dir": "/tmp/x"}, cell=cell, manifest=manifest, lint=lint)
        assert enriched["tokens_in"] == 100
        assert enriched["tokens_out"] == 10
        assert enriched["dollar_cost"] == 0.0125
        assert enriched["manifest_model"] == "or-mid"
        assert enriched["requested_model"] == "or-mid"
        assert enriched["lint_exit_code"] == 1
        assert enriched["lint_findings"] == 2
        assert enriched["lint_warnings"] == 3

    def test_zero_plan_and_verify_tokens_flag_surfaces_the_deterministic_path(self, tmp_path):
        cell = {"name": "b2-standard", "question_file": "/b/b2-question.txt", "depth": "standard", "model": "or-mid"}
        deterministic = run_matrix.enrich_summary({}, cell=cell, manifest=_manifest(), lint={})
        assert deterministic["zero_plan_verify_tokens"] is True

        spent = _manifest()
        spent["accounting"]["phases"]["verify"] = {"input_tokens": 500, "output_tokens": 20}
        assert run_matrix.enrich_summary({}, cell=cell, manifest=spent, lint={})["zero_plan_verify_tokens"] is False


class TestMatrixDriver:
    def _cells(self, tmp_path):
        return [
            {"name": "b1-standard", "question_file": _question_file(tmp_path, "b1"), "depth": "standard", "model": "or-mid"},
            {"name": "b2-standard", "question_file": _question_file(tmp_path, "b2"), "depth": "standard", "model": "or-mid"},
        ]

    def test_driver_writes_logs_summaries_and_lint_per_cell(self, tmp_path):
        run_dirs = {
            "b1-standard": _run_folder(tmp_path, "b1", report=DIRTY_REPORT, manifest=_manifest()),
            "b2-standard": _run_folder(tmp_path, "b2", report=CLEAN_REPORT, manifest=_manifest()),
        }
        calls = []

        async def fake_runner(*, question, profile, model_name, depth, runs_dir):
            calls.append({"question": question, "profile": profile, "model_name": model_name, "depth": depth, "runs_dir": runs_dir})
            print("cell chatter")
            name = "b1-standard" if len(calls) == 1 else "b2-standard"
            return {"thread_id": f"benchmark-{len(calls)}", "depth": depth, "run_dir": run_dirs[name]}

        logs_dir = tmp_path / "logs"
        summaries = run_matrix.run_matrix(self._cells(tmp_path), runs_dir=str(tmp_path / "runs"), logs_dir=str(logs_dir), profile="financial", runner=fake_runner)

        assert [c["model_name"] for c in calls] == ["or-mid", "or-mid"]
        assert [c["depth"] for c in calls] == ["standard", "standard"]
        assert [s["lint_exit_code"] for s in summaries] == [1, 0]
        assert summaries[0]["dollar_cost"] == 0.0125

        for name in ("b1-standard", "b2-standard"):
            assert os.path.isfile(logs_dir / f"{name}.log")
            assert json.loads((logs_dir / f"{name}.json").read_text())["cell"] == name
            assert "findings" in json.loads((logs_dir / f"{name}-lint.json").read_text())
        assert "cell chatter" in (logs_dir / "b1-standard.log").read_text()

    def test_driver_log_matches_the_2026_08_19_format(self, tmp_path):
        async def fake_runner(**kwargs):
            return {"run_dir": None}

        logs_dir = tmp_path / "logs"
        run_matrix.run_matrix(self._cells(tmp_path), runs_dir=str(tmp_path / "runs"), logs_dir=str(logs_dir), profile="financial", runner=fake_runner)
        lines = (logs_dir / "driver.log").read_text().splitlines()

        assert lines[-1] == "ALL DONE"
        assert lines[0].startswith("=== START b1-standard 20") and lines[0].endswith("===")
        assert lines[1].startswith("EXIT=0 cell=b1-standard 20")
        assert lines[2].startswith("=== START b2-standard 20")
        assert len(lines) == 5

    def test_a_failing_cell_records_a_nonzero_exit_and_keeps_going(self, tmp_path):
        async def fake_runner(**kwargs):
            raise RuntimeError("model plane refused the call")

        logs_dir = tmp_path / "logs"
        summaries = run_matrix.run_matrix(self._cells(tmp_path), runs_dir=str(tmp_path / "runs"), logs_dir=str(logs_dir), profile="financial", runner=fake_runner)
        driver_log = (logs_dir / "driver.log").read_text()

        assert [s["exit_code"] for s in summaries] == [1, 1]
        assert driver_log.count("EXIT=1 cell=") == 2
        assert "model plane refused the call" in (logs_dir / "b1-standard.log").read_text()

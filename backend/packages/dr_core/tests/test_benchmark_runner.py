import json

from dr_core.benchmarks.run_benchmark import build_summary


def test_build_summary_reads_persisted_manifest_instrumentation(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "gate_retries": 2,
                "tool_calls": {"total": 7, "by_name": {"fred_series": 2, "record_claim": 5}},
                "coverage": {
                    "first_pass": {
                        "requirements_covered": 0,
                        "requirements_must_cover": 2,
                        "must_cover_states": {"r1": "uncovered", "r2": "uncovered"},
                    },
                    "final": {
                        "requirements_covered": 2,
                        "requirements_must_cover": 2,
                        "must_cover_states": {"r1": "covered", "r2": "covered"},
                    },
                },
                "sources_by_evidence_class": {"academic": 5, "primary_data": 2},
            }
        )
    )
    result = {
        "dr_run": {"run_dir": str(run_dir), "gate_decision": "render"},
        "dr_sources": {"s1": {}},
        "dr_claims": {"c1": {}},
        "dr_requirements": {"r1": {}, "r2": {}},
    }

    summary = build_summary(result=result, thread_id="bench-1", depth="standard", runs_dir=str(tmp_path), new_run_folders=["run-1"])

    assert summary["tool_calls_total"] == 7
    assert summary["gate_retries"] == 2
    assert summary["requirements_covered"] == 2
    assert summary["first_pass_coverage"]["requirements_covered"] == 0
    assert summary["sources_by_evidence_class"] == {"academic": 5, "primary_data": 2}


def test_runner_summary_prefers_state_run_dir_over_directory_diff(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "manifest.json").write_text("{}")
    result = {"dr_run": {"run_dir": str(actual)}}

    summary = build_summary(
        result=result,
        thread_id="bench-2",
        depth="full",
        runs_dir=str(tmp_path),
        new_run_folders=["ambiguous-a", "ambiguous-b"],
    )

    assert summary["run_dir"] == str(actual)

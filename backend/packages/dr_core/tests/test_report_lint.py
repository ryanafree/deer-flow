"""Ported from harness/evals/report_lint_test.py (offline positive/negative checks for
report_lint.py). Same fixtures and assertions as the original; adapted to pytest and to
call dr_core.lint.report_lint.main() in-process (via monkeypatched sys.argv) instead of
via subprocess, since report_lint.py is now a package module rather than a standalone
script. Test names below stand in for the original's single `main()` check, split into
the two cases it exercised: the clean fixture passing lint, and the bad fixture failing
with all five expected findings.
"""

import json
import os

from dr_core.lint.report_lint import main
from dr_core.models import Claim

GOOD_REPORT = """# Test report

## Executive summary

The available evidence supports the answer [1].

## What does the evidence show?

The main finding is stable across the available source [1].

## Conclusion

The evidence supports proceeding cautiously.
"""

BAD_REPORT = "# Test report\n\nThis run has a claim [1][2][3].\n"


def _write_fixture(folder, report, claims):
    with open(os.path.join(folder, "report.md"), "w") as fh:
        fh.write(report)
    with open(os.path.join(folder, "claims.jsonl"), "w") as fh:
        for claim in claims:
            fh.write(json.dumps({"claim": claim}) + "\n")


def test_clean_report_passes_lint(tmp_path, monkeypatch):
    _write_fixture(tmp_path, GOOD_REPORT, ["A qualitative claim"])
    monkeypatch.setattr("sys.argv", ["report_lint.py", str(tmp_path)])
    assert main() == 0


def test_bad_report_fails_lint_with_expected_findings(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, BAD_REPORT, [f"Numeric claim {i}" for i in range(8)])
    monkeypatch.setattr("sys.argv", ["report_lint.py", str(tmp_path)])
    assert main() == 1
    out = capsys.readouterr().out
    for expected in ("no executive summary", "no conclusion", "banned phrase", "3+ citation", "no body table"):
        assert expected in out


def test_numeric_claims_without_table_fires_on_real_claim_dumps(tmp_path, monkeypatch, capsys):
    """task-phase1-consolidate.md item 3: real dr_core dumps key the claim text under
    `text`, not the legacy `claim` field used above. This must still fire against that
    shape, not just the legacy fixture."""
    with open(tmp_path / "report.md", "w") as fh:
        fh.write(BAD_REPORT)
    with open(tmp_path / "claims.jsonl", "w") as fh:
        for i in range(8):
            claim = Claim(claim_id=f"c{i}", text=f"Numeric claim {i}", importance=2, source_id="s1")
            fh.write(json.dumps(claim.model_dump(mode="json")) + "\n")
    monkeypatch.setattr("sys.argv", ["report_lint.py", str(tmp_path)])
    assert main() == 1
    assert "8 numeric claims but no body table" in capsys.readouterr().out

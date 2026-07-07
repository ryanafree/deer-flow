"""Gate evidence for the PART 2 eligibility extension (task-port-linter.md PART 3): a
synthetic claim ledger built from the real dr_core.models schema, plus a tiny report.md
citing it with claim-keyed [n] markers, exercised through dr_core.lint.report_lint.main().

The ledger has one claim of each kind the D2 ruling's publication_status precedence
distinguishes: c1 supported+grounded, c2 contested (a gate flag), c3 excluded
(killed_on_refute), c4 not_verified (unsupported, ungrounded). Ordinals are the 1-based
claims.jsonl line index (c1->[1] ... c4->[4]) per build-logs/PHASE1-SHARED-CONTRACT.md.
"""

import json

from dr_core.lint.report_lint import main
from dr_core.models import (
    Claim,
    CitationStatus,
    GateFlag,
    SupportRecord,
    SupportRelation,
    VerificationRecord,
    VerificationStatus,
)


def _synthetic_claims():
    supported = Claim(
        claim_id="c1",
        text="Revenue grew twenty percent.",
        importance=4,
        source_id="s1",
        support=SupportRecord(
            quote="revenue grew twenty percent",
            relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
            relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY,
        ),
        citation_status=CitationStatus.RESOLVED,
        verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
    )
    contested = Claim(
        claim_id="c2",
        text="The period covered does not align with the filing date.",
        importance=3,
        source_id="s2",
        citation_status=CitationStatus.RESOLVED,
        gate_flags=[GateFlag.PERIOD_MISMATCH],
        verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
    )
    excluded = Claim(
        claim_id="c3",
        text="A refuted claim.",
        importance=3,
        source_id="s3",
        citation_status=CitationStatus.RESOLVED,
        verification=VerificationRecord(status=VerificationStatus.KILLED_ON_REFUTE, complete=True),
    )
    not_verified = Claim(
        claim_id="c4",
        text="The trend may continue.",
        importance=2,
        source_id="s4",
        citation_status=CitationStatus.RESOLVED,
    )
    return [supported, contested, excluded, not_verified]


def _write_run_folder(folder, report, claims):
    (folder / "report.md").write_text(report)
    with open(folder / "claims.jsonl", "w") as fh:
        for claim in claims:
            fh.write(json.dumps(claim.model_dump(mode="json")) + "\n")


CLEAN_REPORT = """# Eligibility test report

## Executive summary

Revenue grew twenty percent according to the filing [1].

## Findings

The period covered does not align with the filing date, the source period does not match the claimed period [2]. According to analysts, the trend may continue [4].

## Conclusion

The evidence supports cautious optimism.
"""


def _run(tmp_path, monkeypatch, report, claims=None):
    _write_run_folder(tmp_path, report, claims if claims is not None else _synthetic_claims())
    monkeypatch.setattr("sys.argv", ["report_lint.py", str(tmp_path)])
    return main()


def test_clean_report_passes_eligibility(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, CLEAN_REPORT) == 0


def test_excluded_claim_cited_in_body_hard_fails(tmp_path, monkeypatch, capsys):
    report = CLEAN_REPORT.replace("[2]", "[2][3]")
    assert _run(tmp_path, monkeypatch, report) == 1
    assert "excluded claim [3]" in capsys.readouterr().out


def test_contested_claim_missing_caveat_fails(tmp_path, monkeypatch, capsys):
    report = CLEAN_REPORT.replace("the source period does not match the claimed period ", "")
    assert _run(tmp_path, monkeypatch, report) == 1
    out = capsys.readouterr().out
    assert "missing-caveat" in out
    assert "missing caveat" in out


def test_contested_claim_with_caveat_present_passes(tmp_path, monkeypatch):
    # CLEAN_REPORT already carries claim_caveat(c2)'s exact string at [2]'s first
    # (only) citation, so this is the positive twin of the missing-caveat case above.
    assert _run(tmp_path, monkeypatch, CLEAN_REPORT) == 0


def test_not_verified_claim_without_attribution_warns(tmp_path, monkeypatch, capsys):
    report = CLEAN_REPORT.replace(
        "According to analysts, the trend may continue [4].",
        "The market trend clearly continues in a positive direction [4].",
    )
    assert _run(tmp_path, monkeypatch, report) == 0  # WARN is soft, not a hard fail
    out = capsys.readouterr().out
    assert "LINT-WARN" in out
    assert "unattributed-not-verified" in out

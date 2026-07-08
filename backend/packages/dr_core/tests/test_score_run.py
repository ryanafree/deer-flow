"""dr_core.eval.score_run against real write_run-produced run folders
(build-logs/task-port-evalfixtures.md). Exercises two of the five ported seeded-trap
fixtures in dr_core/eval/profile_questions.yaml: general_smoke (a clean run that must
pass) and legal_fabricated_cite (a planted trap the scorer must catch — a resolved real
citation alongside a citation_status=NOT_FOUND fabricated one)."""

from dr_core.eval.score_run import load_fixtures, load_run_from_folder, score
from dr_core.models import CitationStatus, Claim, GateFlag, Source, SupportRecord, SupportRelation, VerificationStatus
from dr_core.models.ledger import VerificationRecord
from dr_core.run.write_run import write_run

FIXTURES = load_fixtures()


def _source(id, **kw):
    kw.setdefault("url_or_id", f"https://example.test/{id}")
    kw.setdefault("source_system", "web")
    kw.setdefault("authority_tier", 2)
    kw.setdefault("retrieved_at", "2026-07-06T00:00:00-05:00")
    return Source(id=id, **kw)


def test_general_smoke_fixture_passes_against_a_clean_run(tmp_path):
    fx = FIXTURES["general_smoke"]
    sources = [_source(f"s{i}") for i in range(1, 4)]
    claims = [
        Claim(
            claim_id=f"c{i}",
            text=f"Homicide rates rose after 2020 per source {i}.",
            importance=3,
            source_id=f"s{i}",
            support=SupportRecord(
                quote="rates rose after 2020",
                relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
                relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY,
            ),
            citation_status=CitationStatus.RESOLVED,
        )
        for i in range(1, 4)
    ]
    body = "# Report\n\n## Executive summary\n\n" + ("Homicide rates rose sharply after 2020 across multiple jurisdictions. " * 20) + "\n\n## Conclusion\n\nThe rise appears durable.\n"
    folder = write_run(report_body=body, sources=sources, claims=claims, requirements=[], profile="general", question=fx["question"], runs_dir=str(tmp_path))

    manifest, loaded_claims, loaded_sources, report = load_run_from_folder(folder)
    checks = score("general_smoke", fx["expect"], manifest, loaded_claims, loaded_sources, report)
    assert checks, "expected at least one assertion"
    assert all(ok for ok, _ in checks), checks


def test_legal_fabricated_cite_fixture_catches_the_planted_trap(tmp_path):
    fx = FIXTURES["legal_fabricated_cite"]
    sources = [_source("s1", source_system="courtlistener"), _source("s2", source_system="courtlistener")]
    claims = [
        Claim(  # real cite: Harlow v. Fitzgerald resolves
            claim_id="c1",
            text="Officials get qualified immunity under Harlow v. Fitzgerald, 457 U.S. 800 (1982).",
            importance=4,
            source_id="s1",
            citation_status=CitationStatus.RESOLVED,
        ),
        Claim(  # fabricated cite: Eyecite lookup 404s against CourtListener
            claim_id="c2",
            text="The Supreme Court abolished qualified immunity in Roe v. Doe, 605 U.S. 217 (2025).",
            importance=4,
            source_id="s2",
            citation_status=CitationStatus.NOT_FOUND,
        ),
    ]
    body = "# Report\n\n## Executive summary\n\nQualified immunity shields officials from suit under most circumstances.\n\n## Conclusion\n\nThe doctrine remains intact.\n"
    folder = write_run(report_body=body, sources=sources, claims=claims, requirements=[], profile="legal", question=fx["question"], runs_dir=str(tmp_path))

    manifest, loaded_claims, loaded_sources, report = load_run_from_folder(folder)
    checks = score("legal_fabricated_cite", fx["expect"], manifest, loaded_claims, loaded_sources, report)
    assert checks, "expected at least one assertion"
    assert all(ok for ok, _ in checks), checks


def test_financial_vintage_fixture_counts_killed_claims_from_the_ledger_not_manifest_counts(tmp_path):
    """Regression for the S10 re-acceptance finding: `claims_killed_min` used to read
    manifest['counts']['killed_on_refute'], which write_run computes over the
    ELIGIBLE-only claim set — a killed_on_refute claim is by definition never eligible,
    so that manifest field is structurally always 0 and the assertion could never pass.
    `_derive_count` now counts `claim.verification.status == KILLED_ON_REFUTE` straight
    off the loaded (ledger-preferring) claims, same pattern as `citations_not_found`."""
    fx = FIXTURES["financial_vintage"]
    sources = [_source("s1", source_system="press"), _source("s2", source_system="press", authority_tier=1)]
    claims = [
        Claim(  # fabricated magnitude, no data_ref -- killed on refutation
            claim_id="c1",
            text="Apple's FY2024 net sales were $450 billion.",
            importance=5,
            source_id="s1",
            gate_flags=[GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE],
            verification=VerificationRecord(status=VerificationStatus.KILLED_ON_REFUTE, complete=True),
        ),
        Claim(  # wrong period vs. its own data_ref -- caught deterministically by provenance
            claim_id="c2",
            text="Apple's FY2024 net sales were $383.285 billion.",
            importance=5,
            source_id="s2",
            data_ref={"value": "383285000000", "period": "2023-09-30", "claimed_period": "2024-09-30", "source_class": "primary_filing"},
        ),
    ]
    body = "# Report\n\n## Executive summary\n\nNo recorded claim met the eligibility bar for inclusion in this report.\n\n## Conclusion\n\nNo claim in this pass met the eligibility bar for inclusion.\n"
    folder = write_run(report_body=body, sources=sources, claims=[], ledger_claims=claims, requirements=[], profile="financial", question=fx["question"], runs_dir=str(tmp_path))

    manifest, loaded_claims, loaded_sources, report = load_run_from_folder(folder)
    checks = score("financial_vintage", fx["expect"], manifest, loaded_claims, loaded_sources, report)
    assert checks, "expected at least one assertion"
    assert all(ok for ok, _ in checks), checks

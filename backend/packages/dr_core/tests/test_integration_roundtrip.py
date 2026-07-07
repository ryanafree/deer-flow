"""Phase 1 GATE test (task-phase1-consolidate.md item 5): a synthetic full ledger, built
straight from dr_core.models, round-tripped through write_run -> render -> report_lint.

Covers one claim of each publication-eligibility class the D2 ruling's precedence
distinguishes (dr_core.models.derive.publication_status):
  c1 supported+grounded, c2 contested via a gate flag, c3 contested via Conflict
  membership (no gate flag, so it carries no claim_caveat), c4 excluded
  (killed_on_refute), c5 not_verified.
Ordinals are the 1-based claims.jsonl line index (c1 -> [1] ... c5 -> [5]), per
PHASE1-SHARED-CONTRACT.md. Sources, a couple of Requirements, one Conflict, and two
CoverageMappings round out the ledger.
"""

import json
import os

from dr_core.lint.report_lint import _eligibility_findings, _load_claims, _load_conflicts, body_of
from dr_core.models import (
    Claim,
    CitationStatus,
    Conflict,
    ConflictOutcome,
    CoverageMapping,
    CoverageRelation,
    GateFlag,
    Requirement,
    RequirementKind,
    RequirementState,
    Source,
    SupportRecord,
    SupportRelation,
    VerificationRecord,
    VerificationStatus,
)
from dr_core.render.render_report import render
from dr_core.run.write_run import write_run

CAVEAT_C2 = "vendor-reported and not independently reproduced"  # claim_caveat(c2) exact string


def _synthetic_ledger():
    claims = [
        Claim(  # c1: supported + grounded
            claim_id="c1",
            text="Revenue grew 20 percent in fiscal Q3 according to the filing.",
            importance=4,
            source_id="s1",
            support=SupportRecord(
                quote="revenue grew twenty percent",
                relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
                relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY,
            ),
            citation_status=CitationStatus.RESOLVED,
            verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        ),
        Claim(  # c2: contested via gate flag (caveat = CAVEAT_C2)
            claim_id="c2",
            text="The vendor reported a 45 percent efficiency gain in Q2.",
            importance=3,
            source_id="s2",
            citation_status=CitationStatus.RESOLVED,
            gate_flags=[GateFlag.VENDOR_REPORTED],
            verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        ),
        Claim(  # c3: contested via Conflict membership only (no gate flag -> no caveat)
            claim_id="c3",
            text="Sources disagree on the exact launch date, some citing March and others April.",
            importance=3,
            source_id="s3",
            support=SupportRecord(quote="launch date disagreement", relation_extractor=SupportRelation.SUPPORTS_DIRECTLY, relation_reviewer=SupportRelation.SUPPORTS_DIRECTLY),
            citation_status=CitationStatus.RESOLVED,
            verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        ),
        Claim(  # c4: excluded (killed_on_refute)
            claim_id="c4",
            text="The product achieved 99 percent uptime per an internal memo that was later refuted.",
            importance=3,
            source_id="s4",
            citation_status=CitationStatus.RESOLVED,
            verification=VerificationRecord(status=VerificationStatus.KILLED_ON_REFUTE, complete=True),
        ),
        Claim(  # c5: not_verified (default pending, unsupported, ungrounded)
            claim_id="c5",
            text="The positive trend may continue into next year.",
            importance=2,
            source_id="s5",
        ),
    ]
    sources = [Source(id=f"s{i}", url_or_id=f"https://example.test/{i}", source_system="web", title=f"Source {i}", authority_tier=2, retrieved_at="2026-07-06T00:00:00-05:00") for i in range(1, 6)]
    requirements = [
        Requirement(id="r1", kind=RequirementKind.ENTITY, text="Explain the revenue growth driver", must_cover=True, terminal_state=RequirementState.COVERED),
        Requirement(id="r2", kind=RequirementKind.ENTITY, text="Explain the contested launch date discrepancy", must_cover=True, terminal_state=RequirementState.PARTIAL),
    ]
    conflicts = [Conflict(id="cf1", claim_ids=["c3"], description="conflicting accounts of the launch date", outcome=ConflictOutcome.UNRESOLVED_PERSISTENT)]
    coverage = [
        CoverageMapping(requirement_id="r1", claim_id="c1", relation=CoverageRelation.DIRECT),
        CoverageMapping(requirement_id="r2", claim_id="c3", relation=CoverageRelation.PARTIAL),
    ]
    return claims, sources, requirements, conflicts, coverage


CLEAN_BODY = """# Integration Roundtrip Report

## Executive summary

Revenue grew twenty percent in fiscal Q3 according to the filing [1]. The vendor reported a 45 percent efficiency gain, vendor-reported and not independently reproduced [2]. Sources disagree on the exact launch date, a conflict that remains unresolved [3].

## Findings

According to observers, the positive trend may continue into next year [5].

## Conclusion

The evidence supports cautious optimism about the underlying trend.
"""

# Cites the EXCLUDED claim (c4, ordinal 4) in the body -> must HARD-FAIL.
DOCTORED_BODY = CLEAN_BODY.replace(
    "## Conclusion",
    "The product achieved 99 percent uptime per an internal memo [4].\n\n## Conclusion",
)

# Drops c2's caveat text at its first (only) citation but keeps the citation -> must HARD-FAIL.
MISSING_CAVEAT_BODY = CLEAN_BODY.replace(", vendor-reported and not independently reproduced", "")


def _write(tmp_path, name, report_body, claims, sources, requirements, conflicts, coverage):
    return write_run(
        report_body=report_body,
        sources=sources,
        claims=claims,
        requirements=requirements,
        conflicts=conflicts,
        coverage=coverage,
        profile="general",
        question=f"Integration roundtrip {name}?",
        runs_dir=str(tmp_path / name),
    )


# ---- 1. write_run: complete artifact set, frozen ordinal, every artifact reloads ----------------


def test_write_run_produces_the_complete_artifact_set_and_round_trips(tmp_path):
    claims, sources, requirements, conflicts, coverage = _synthetic_ledger()
    folder = _write(tmp_path, "clean", CLEAN_BODY, claims, sources, requirements, conflicts, coverage)

    for name in ("report.md", "claims.jsonl", "sources.jsonl", "requirements.jsonl", "conflicts.jsonl", "coverage.jsonl", "manifest.json"):
        assert os.path.isfile(os.path.join(folder, name)), f"missing artifact: {name}"

    reloaded_claims = [Claim(**json.loads(line)) for line in open(os.path.join(folder, "claims.jsonl"))]
    assert [c.claim_id for c in reloaded_claims] == ["c1", "c2", "c3", "c4", "c5"]  # line order IS the citation ordinal

    reloaded_sources = [Source(**json.loads(line)) for line in open(os.path.join(folder, "sources.jsonl"))]
    assert {s.id for s in reloaded_sources} == {"s1", "s2", "s3", "s4", "s5"}

    reloaded_requirements = [Requirement(**json.loads(line)) for line in open(os.path.join(folder, "requirements.jsonl"))]
    assert {r.id for r in reloaded_requirements} == {"r1", "r2"}

    reloaded_conflicts = [Conflict(**json.loads(line)) for line in open(os.path.join(folder, "conflicts.jsonl"))]
    assert reloaded_conflicts[0].id == "cf1"

    reloaded_coverage = [CoverageMapping(**json.loads(line)) for line in open(os.path.join(folder, "coverage.jsonl"))]
    assert {(m.requirement_id, m.claim_id) for m in reloaded_coverage} == {("r1", "c1"), ("r2", "c3")}


# ---- 2. render(): caveats surface, excluded claim absent from references, no mutated text -------


def test_render_surfaces_caveats_drops_excluded_from_references_and_mutates_no_claim_text(tmp_path):
    claims, sources, requirements, conflicts, coverage = _synthetic_ledger()
    folder = _write(tmp_path, "render", CLEAN_BODY, claims, sources, requirements, conflicts, coverage)
    out = render(folder)

    assert CAVEAT_C2 in out  # c2's contested caveat, verbatim (claim_caveat's fixed table)

    # c4 (excluded) is never cited [4] in CLEAN_BODY, so it must not surface in the rendered
    # narrative body -- distinct from the appendix's evidence ledger, which deliberately lists
    # every claim regardless of eligibility (an audit trail, not the reader-facing narrative).
    narrative = out.split("<main>", 1)[1].split('<section class="references-section">', 1)[0]
    assert "99 percent uptime" not in narrative
    assert "[4]" not in narrative

    for claim in claims:
        first_clause = claim.text.split(",")[0].split(".")[0]
        assert first_clause in out, f"claim text mutated or dropped: {claim.claim_id!r}"


# ---- 3. linter: clean passes, excluded-in-body hard-fails, missing-caveat hard-fails -------------


def test_linter_clean_report_has_no_hard_findings(tmp_path):
    claims, sources, requirements, conflicts, coverage = _synthetic_ledger()
    folder = _write(tmp_path, "lint-clean", CLEAN_BODY, claims, sources, requirements, conflicts, coverage)

    loaded_claims = _load_claims(folder)
    loaded_conflicts = _load_conflicts(folder)
    assert loaded_claims is not None  # real ledger schema must parse, not fall back to legacy

    body = body_of(open(os.path.join(folder, "report.md")).read())
    hard, _warn = _eligibility_findings(body, loaded_claims, loaded_conflicts)
    assert hard == []


def test_linter_hard_fails_when_excluded_claim_cited_in_body(tmp_path):
    claims, sources, requirements, conflicts, coverage = _synthetic_ledger()
    folder = _write(tmp_path, "lint-doctored", DOCTORED_BODY, claims, sources, requirements, conflicts, coverage)

    loaded_claims = _load_claims(folder)
    loaded_conflicts = _load_conflicts(folder)
    body = body_of(open(os.path.join(folder, "report.md")).read())
    hard, _warn = _eligibility_findings(body, loaded_claims, loaded_conflicts)
    assert any("excluded-in-body" in f and "[4]" in f for f in hard), hard


def test_linter_hard_fails_when_contested_claim_cited_without_its_caveat(tmp_path):
    claims, sources, requirements, conflicts, coverage = _synthetic_ledger()
    folder = _write(tmp_path, "lint-missing-caveat", MISSING_CAVEAT_BODY, claims, sources, requirements, conflicts, coverage)

    loaded_claims = _load_claims(folder)
    loaded_conflicts = _load_conflicts(folder)
    body = body_of(open(os.path.join(folder, "report.md")).read())
    hard, _warn = _eligibility_findings(body, loaded_claims, loaded_conflicts)
    assert any("missing-caveat" in f and "[2]" in f for f in hard), hard

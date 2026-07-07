"""Round-trip regression tests: write_run() -> render_report.render() through the real
run-folder contract, not the hand-built fixtures in test_render_report.py. B3 (the
appendix-marker desync) and M1 (the counts key mismatch) only show up once real
write_run() output is fed to the renderer -- both were "reproduced live" per
REVIEW_FINISH_PLAN_2026-07-06.md Part I.
"""

import json
import os

from dr_core.models import Claim, Source
from dr_core.models.enums import VerificationStatus
from dr_core.render.render_report import render
from dr_core.run.write_run import write_run


def _source(source_id, **overrides):
    defaults = dict(id=source_id, url_or_id=f"https://example.test/{source_id}", source_system="web", title=f"Source {source_id}", authority_tier=2, retrieved_at="2026-07-06T00:00:00-05:00")
    defaults.update(overrides)
    return Source(**defaults)


def _claim(claim_id, **overrides):
    defaults = dict(claim_id=claim_id, text=f"Claim text for {claim_id}.", importance=3, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


def _write_and_render(tmp_path, claims):
    folder = write_run(
        report_body="# Round-trip Report\n\n## Direct answer\n\nSome findings here [1].\n",
        sources=[_source("s1")],
        claims=claims,
        requirements=[],
        profile="general",
        question="Does the roundtrip work?",
        runs_dir=str(tmp_path),
    )
    return folder, render(folder)


# ---- B3: appendix marker must not leak into the rendered narrative body -----------


def test_appendix_heading_does_not_leak_into_the_rendered_body(tmp_path):
    folder, html = _write_and_render(tmp_path, [_claim("c1")])
    report_md = open(os.path.join(folder, "report.md")).read()
    # Sanity: the writer's real appendix header text (not a hand-copied string), so this
    # test itself breaks loudly if the two ever drift out of sync again.
    assert "## Appendix — Methodology, Verification & Requirement Coverage" in report_md

    body_html = html.split('<section class="appendix-section">')[0]
    # The bug: split_body's exact-string match silently failed against write_run's real
    # header, so the whole report.md -- appendix included -- became "body", and the
    # writer's raw markdown appendix rendered a second time as an ordinary body section.
    assert "Methodology, Verification & Requirement Coverage" not in body_html
    assert "Provenance and validation detail" not in body_html

    # The renderer's own (JSON-derived) appendix section still renders exactly once.
    assert html.count("Appendix — Methodology, Sources &amp; Validation") == 1


# ---- M1: manifest counts keys the renderer reads must come from the real writer ---


def test_killed_on_refute_claim_renders_a_nonzero_killed_callout(tmp_path):
    claims = [
        _claim("c1"),
        _claim("c2", verification={"status": VerificationStatus.KILLED_ON_REFUTE, "complete": True}),
    ]
    folder, html = _write_and_render(tmp_path, claims)
    manifest = json.load(open(os.path.join(folder, "manifest.json")))

    assert manifest["counts"]["claims_killed"] == 1
    assert manifest["counts"]["killed_on_refute"] == 1  # the aggregate key stays intact
    assert "Killed on refutation" in html
    assert "1 claim failed adversarial verification" in html

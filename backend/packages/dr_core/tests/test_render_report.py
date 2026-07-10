"""Ported from `~/Documents/Projects/DeepResearch/harness/evals/render_report_test.py`.

Structural invariants over hand-built fixtures (no model calls, no run folder needed on disk
beyond what the test writes itself). Checks the deterministic-renderer contract (PLAN_v3.md
locked decision 8): every citation resolves, key-stat cards only appear for real supported
numeric claims, callouts match the manifest/requirements data exactly, and no claim text is
altered — only reorganized/styled.

Citation-related assertions are updated for D2 divergence #2 (claim-keyed citations,
build-logs/PHASE1-SHARED-CONTRACT.md): in-body `[n]` is the claims.jsonl ordinal, not a source
number. The fixture's markers already used claim positions (the harness's own "legacy" path),
so the resolution and dedup assertions carry over unchanged. The harness's second test block —
covering its "0.2.0+" `manifest.source_order` dense-source-numbering variant — is dropped: that
scheme has no equivalent under claim-keyed citations (there is exactly one scheme now, and
`source_order` is not read at all).
"""

import json
import re

import pytest
from dr_core.render.render_report import render

REPORT_MD = """# Test Report Title

## Direct answer

The market moved sharply today because of a rate decision. VIX closed at 18.89 [1] and a contested claim appears here [2].

## Second section

More findings here with a repeated citation [1][1] and one more [3].

---

## Appendix — Methodology, Sources & Validation

*ignored by the renderer; rebuilt from JSON instead*
"""

MANIFEST = {
    "profile": "financial",
    "question": "What happened to volatility today?",
    "restated_question": "What happened to volatility today?",
    "engine_version": "0.2.0-dev",
    "depth": "standard",
    "verify_mode": True,
    "timestamp": "2026-06-30T12:00:00-05:00",
    "counts": {"claims": 3, "claims_verified": 1, "claims_killed": 1, "claims_flagged": 1},
    "loop": {"conflicts_open": 1, "requirements_covered": 1, "requirements_must_cover": 2},
}

# Field names follow dr_core.models.ledger.Claim, not the harness's ad hoc dict shape:
# claim -> text, source_ref -> source_id, gate.flags -> gate_flags, authority_tier moved to Source.
CLAIMS = [
    {
        "claim_id": "c1",
        "text": "VIX closed at 18.89 on 2026-06-30.",
        "importance": 5,
        "source_id": "s1",
        "data_ref": {"value": "18.89", "period": "2026-06-30", "query_params": "x"},
        "data_provenance": "matched",
        "verification": {"status": "supported", "complete": True},
    },
    {
        "claim_id": "c2",
        "text": "A contested claim that was refuted.",
        "importance": 4,
        "source_id": "s2",
        "support": {"quote": "quote text", "relation_extractor": "supports_directly"},
        "verification": {"status": "killed_on_refute", "complete": True},
        "gate_flags": ["vendor_reported"],
    },
    {
        "claim_id": "c3",
        "text": "A minor supporting claim.",
        "importance": 3,
        "source_id": "s1",
        "support": {"quote": "another quote", "relation_extractor": "supports_directly"},
        "verification": {"status": "not_verified"},
    },
]

SOURCES = [
    {"id": "s1", "title": "FRED VIX series", "source_system": "fred", "authority_tier": 1, "url_or_id": "data_fetch:fred x", "retrieved_at": "2026-06-30T12:00:00-05:00"},
    {"id": "s2", "title": "Some blog", "source_system": "web", "authority_tier": 2, "url_or_id": "https://example.test/blog", "retrieved_at": "2026-06-30T12:00:00-05:00"},
]

# "state" is read from the ported Requirement's terminal_state field; there is no separate live
# requirement-state field on the artifact (see render_report.py's module docstring).
REQUIREMENTS = [
    {"id": "r1", "kind": "entity", "text": "Explain today's VIX move", "must_cover": True, "terminal_state": "covered"},
    {"id": "r2", "kind": "entity", "text": "Explain the contested driver", "must_cover": True, "terminal_state": "partial"},
]


def _write_run_dir(tmp_path):
    (tmp_path / "report.md").write_text(REPORT_MD)
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST))
    for name, rows in (("claims.jsonl", CLAIMS), ("sources.jsonl", SOURCES), ("requirements.jsonl", REQUIREMENTS)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(tmp_path)


@pytest.fixture
def out(tmp_path):
    return render(_write_run_dir(tmp_path))


def test_uses_the_leading_h1_as_the_title_not_restated_question(out):
    assert "Test Report Title" in out.split("<main>")[0]


def test_question_shown_separately_from_the_title(out):
    assert MANIFEST["question"] in out


# ---- citation resolution: every [n] in the body resolves to a #ref-N target that exists ------


def test_every_citation_link_target_has_a_matching_reference_anchor(out):
    cite_targets = {int(m) for m in re.findall(r'href="#ref-(\d+)"', out)}
    ref_ids = {int(m) for m in re.findall(r'id="ref-(\d+)"', out)}
    assert cite_targets and cite_targets.issubset(ref_ids)


def test_no_bare_n_markers_survive_unrewritten_in_the_body(out):
    body_before_cards = out.split('<section class="cards-section"')[0].split("<main>")[-1]
    assert "[1]" not in body_before_cards.replace('href="#ref-1"', "").replace(">[1]<", "REWRITTEN")


def test_citations_to_the_same_source_dedupe_to_one_reference_number(out):
    # claim 1 (source s1) and claim 3 (source s1) both resolve to source s1 -> same ref number.
    assert out.count('href="#ref-1"') >= 2
    assert out.count('<li id="ref-2"') == 1


# ---- key-stat cards: only the one real supported/high-materiality numeric claim qualifies -----


def test_exactly_one_key_stat_card(out):
    body_only = out.split("</style>", 1)[1]
    assert body_only.count('class="card"') == 1


def _cards_block(out):
    body_only = out.split("</style>", 1)[1]
    if '<section class="cards-section">' not in body_only:
        return ""
    return body_only.split('<section class="cards-section">')[1].split("</section>")[0]


def test_the_card_shows_the_claims_actual_value_verbatim(out):
    assert "18.89" in _cards_block(out)


def test_killed_on_refute_claim_c2_does_not_get_a_card(out):
    assert _cards_block(out).count('class="card"') == 1


# ---- callouts reflect the manifest/requirements data exactly, no invented text -----------------


def test_open_must_cover_requirement_r2_produces_a_callout_naming_it(out):
    assert "Explain the contested driver" in out and "Not fully answered" in out


def test_covered_requirement_r1_produces_no_callout(out):
    assert not re.search(r"Not fully answered:\s*Explain today's VIX move", out)


def test_open_conflict_count_is_rendered_from_manifest_loop_conflicts_open(out):
    assert "1 conflicting source span" in out


def test_flag_counts_callout_reflects_the_one_vendor_reported_flag(out):
    assert "1 vendor_reported" in out


def test_killed_count_callout_reflects_manifest_counts_claims_killed(out):
    assert "1 claim failed adversarial verification" in out


# ---- structured table + appendix present the same claim data, not fabricated ------------------


def test_structured_table_includes_the_data_ref_claims_exact_value(out):
    assert "data-table" in out and "18.89" in out


def test_appendix_evidence_ledger_lists_all_3_claims(out):
    # c1 is structured (data:matched); c3 (medium materiality, no reviewer) inherits its
    # extractor label "supports_directly"; c2 (high materiality, no reviewer) resolves to
    # "unreviewed" via effective_relation — a materiality-aware derivation the harness's flat
    # precomputed field never expressed, not a fabricated value (dr_core.models.derive.
    # effective_relation, ruling divergence 3).
    assert out.count("data:matched") == 1
    assert "supports_directly" in out
    assert "unreviewed" in out


def test_requirement_coverage_table_in_the_appendix_lists_both_requirements(out):
    assert "r1" in out and "r2" in out


# ---- no new claim text: every claim's text appears verbatim somewhere in the output -----------


@pytest.mark.parametrize("claim", CLAIMS, ids=[c["claim_id"] for c in CLAIMS])
def test_claim_text_fragment_present_verbatim(out, claim):
    # the renderer may truncate long claims with an ellipsis for takeaways/tables, but the full
    # text must appear verbatim at least once; check the first clause to sidestep truncation.
    first_clause = re.split(r"[.:]", claim["text"])[0]
    assert first_clause in out


# ---- HTML is well-formed enough for a real browser: balanced tags for our own elements --------


@pytest.mark.parametrize("tag", ["section", "table", "div", "ol", "li"])
def test_balanced_tags(out, tag):
    opens = len(re.findall(rf"<{tag}\b", out))
    closes = len(re.findall(rf"</{tag}>", out))
    assert opens == closes

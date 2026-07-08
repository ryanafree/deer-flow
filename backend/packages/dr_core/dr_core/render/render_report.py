"""render_report.py — a deterministic, print-safe HTML renderer for a /dr run folder.

Ported from `~/Documents/Projects/DeepResearch/harness/render_report.py` (PLAN_v3.md locked
decision 8: "The renderer is deterministic. It may reorganize, format, and style; it may not
introduce new factual content."). This script makes no model calls and invents no text: every
word in the output either comes verbatim from report.md, or is a template label around a value
already present in manifest.json / claims.jsonl / sources.jsonl / requirements.jsonl. Where it
needs a "takeaway" for a section, it extracts that section's own first sentence rather than
writing new prose.

Port note (D2 divergence #2, build-logs/PHASE1-SHARED-CONTRACT.md): the harness numbered
SOURCES in first-use order and resolved body citations to source numbers. This port replaces
that with CLAIM-KEYED citations — in-body `[n]` is the claims.jsonl line index (the claim's
frozen ordinal), resolved n -> claim -> claim.source_id -> Source. The reference list is
derived from those resolved sources (deduped, ordered by first appearance); numbering never
changes across regenerations. The harness's dense "source_order" 0.2.0+ variant (manifest-
supplied source numbering) has no equivalent here and is dropped — there is exactly one
citation scheme now.

The harness's ad hoc claim/source dicts are replaced by dr_core.models' pydantic Claim/Source/
Requirement, which reshapes several field names (claim.claim -> claim.text, claim.source_ref ->
claim.source_id, claim.gate.flags -> claim.gate_flags, claim.authority_tier moved onto Source,
claim.materiality is derived-only). Where the harness ad-hoc-computed eligibility (materiality
threshold + status blocklist + gate-flag blocklist) for key-stat cards, this port instead calls
dr_core.models.derive.publication_status per the shared contract's instruction not to reinvent
eligibility; `load_run` also loads `conflicts.jsonl` (empty if absent) and threads it into every
`publication_status(...)` call so contested-via-conflict is reachable, not just contested-via-
gate-flag. Requirement.mapped_claim_ids has no equivalent field on the ported Requirement model
(that linkage lives in CoverageMapping); `load_run` now loads `coverage.jsonl` when present
(Phase 1 closure, PHASE1-SHARED-CONTRACT.md) and the appendix's requirement table restores an
Evidence column of claim ordinals from it, degrading back to the column-less table when the
artifact is absent (older run folders). Requirement "state" is read from the ported model's
`terminal_state` (there is no separate live-state field in the artifact).

Usage:
  uv run --with markdown python3 -m dr_core.render.render_report <run-folder> [--out report.html]
"""

import argparse
import html as htmllib
import json
import os
import re
import sys

try:
    import markdown as md_lib
except ImportError:
    sys.exit("render_report.py: requires the 'markdown' package — run via `uv run --with markdown python3 -m dr_core.render.render_report ...`")

from dr_core.models import (
    Claim,
    Conflict,
    CoverageMapping,
    Materiality,
    PublicationStatus,
    Requirement,
    RequirementKind,
    RequirementState,
    Source,
    VerificationStatus,
)
from dr_core.models.derive import effective_relation, is_grounded, materiality_of, publication_status

# Tolerant match (same pattern as dr_core.lint.report_lint.APPENDIX_RE) rather than an
# exact-string marker: write_run.py's own appendix heading text has drifted from this
# renderer's before (B3) and a literal-string match silently stops catching it again.
APPENDIX_RE = re.compile(r"^##+\s*Appendix", re.M)
# Ported from the harness verbatim in spirit; the harness's own copy of this set (hyphenated
# strings, no source of truth) has been dropped rather than kept out of sync — the blocking-flag
# set now lives once, in dr_core.models.derive, and is applied via is_grounded/publication_status
# below instead of re-checked locally.
TIER_LABEL = {1: "primary/peer-reviewed", 2: "secondary", 3: "press/tertiary", 4: "forum/weak"}


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(run_dir):
    with open(os.path.join(run_dir, "report.md")) as f:
        report_md = f.read()
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    claims = [Claim(**c) for c in load_jsonl(os.path.join(run_dir, "claims.jsonl"))]
    sources = [Source(**s) for s in load_jsonl(os.path.join(run_dir, "sources.jsonl"))]
    requirements = [Requirement(**r) for r in load_jsonl(os.path.join(run_dir, "requirements.jsonl"))]
    # Optional artifacts (Phase 1 closure, PHASE1-SHARED-CONTRACT.md): empty list when
    # the run folder predates them or the caller passed none to write_run().
    conflicts = [Conflict(**c) for c in load_jsonl(os.path.join(run_dir, "conflicts.jsonl"))]
    coverage = [CoverageMapping(**c) for c in load_jsonl(os.path.join(run_dir, "coverage.jsonl"))]
    return report_md, manifest, claims, sources, requirements, conflicts, coverage


def split_body(report_md):
    """The body is everything write_run.py put before the deterministic appendix it appends. That
    appendix stays in report.md verbatim (the portable/plain-text copy); this renderer rebuilds its
    own HTML version from the JSON records instead of re-parsing that markdown."""
    m = APPENDIX_RE.search(report_md)
    body = report_md[: m.start()] if m else report_md
    return body.strip()


def first_sentence(text, max_len=220):
    """A verbatim excerpt, not a new sentence — the deterministic-renderer constraint means a
    'takeaway' can only be text already in the report, promoted visually, not authored fresh."""
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = parts[0] if parts else text
    if len(out) > max_len:
        out = out[:max_len].rsplit(" ", 1)[0] + "…"
    return out


def split_sections(body_md):
    """Split the body on top-level '## ' headings. A leading '# Title' (inconsistent across runs —
    the Synth prompt does not require one) is captured separately and NOT treated as a section."""
    title = None
    m = re.match(r"^#\s+(.+?)\s*\n+", body_md)
    if m:
        title = m.group(1).strip()
        body_md = body_md[m.end() :]
    parts = re.split(r"(?m)^##\s+(.+?)\s*$", body_md)
    # re.split with a capturing group yields: [prefix, heading1, content1, heading2, content2, ...]
    prefix = parts[0].strip()
    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, content))
    return title, prefix, sections


def build_claim_refs(claims, sources):
    """Resolve in-body `[n]` markers to sources, claim-keyed (D2 divergence #2).

    `n` is the 1-based claims.jsonl line index — the claim's frozen ordinal, never renumbered.
    Each ordinal resolves via claim.source_id to its Source; the reference list is built from
    those resolved sources, deduped and ordered by first appearance among the claims (so a claim
    cited later that shares a source with an earlier claim gets the earlier claim's ref number).
    Returns (claim_refnum: {ordinal: ref_num}, ordered_sources: [Source, ...]).
    """
    src_by_id = {s.id: s for s in sources}
    ref_num_by_source_id = {}
    ordered_sources = []
    claim_refnum = {}
    for ordinal, claim in enumerate(claims, start=1):
        src = src_by_id.get(claim.source_id)
        if src is None:
            continue
        if claim.source_id not in ref_num_by_source_id:
            ref_num_by_source_id[claim.source_id] = len(ordered_sources) + 1
            ordered_sources.append(src)
        claim_refnum[ordinal] = ref_num_by_source_id[claim.source_id]
    return claim_refnum, ordered_sources


CITATION_RE = re.compile(r"\[(\d+)\]")


def rewrite_citations(section_html, claim_refnum):
    """Runs AFTER markdown conversion, so any remaining literal '[n]' in the HTML is unambiguously a
    citation marker — a markdown link '[text](url)' has already become an <a> tag by this point, so
    this cannot misfire on real link syntax."""

    def repl(m):
        n = int(m.group(1))
        ref = claim_refnum.get(n)
        if ref is None:
            return m.group(0)
        return f'<a class="cite" href="#ref-{ref}">[{ref}]</a>'

    return CITATION_RE.sub(repl, section_html)


def md_to_html(text):
    return md_lib.markdown(text, extensions=["tables"])


def esc(s):
    return htmllib.escape(str(s if s is not None else ""), quote=True)


def render_key_stat_cards(claims, src_by_id, claim_refnum_by_claim_id, conflicts=()):
    candidates = []
    for c in claims:
        if not c.data_ref:
            continue
        materiality = materiality_of(c.importance)
        if materiality != Materiality.HIGH:
            continue
        grounded = is_grounded(c, materiality)
        if publication_status(c, materiality=materiality, grounded=grounded, conflicts=conflicts) != PublicationStatus.SUPPORTED:
            continue
        candidates.append(c)
    candidates.sort(key=lambda c: (-(c.importance or 0), (src_by_id.get(c.source_id).authority_tier if src_by_id.get(c.source_id) else 9)))
    candidates = candidates[:6]
    if not candidates:
        return ""
    cards = []
    for c in candidates:
        dr = c.data_ref or {}
        label = esc(first_sentence(c.text or "", max_len=110))
        value = esc(dr.get("value"))
        period = esc(dr.get("period")) if dr.get("period") else None
        ref = claim_refnum_by_claim_id.get(c.claim_id)
        cite = f' <a class="cite" href="#ref-{ref}">[{ref}]</a>' if ref else ""
        period_html = f'<div class="card-period">as of {period}</div>' if period else ""
        cards.append(f'<div class="card"><div class="card-value">{value}</div>{period_html}<div class="card-label">{label}{cite}</div></div>')
    return '<section class="cards-section"><h2>Key figures</h2><div class="cards">' + "".join(cards) + "</div></section>"


def render_structured_table(claims, src_by_id, claim_refnum_by_claim_id):
    structured = [c for c in claims if c.data_ref]
    if not structured:
        return ""
    rows = []
    for c in structured:
        dr = c.data_ref or {}
        ref = claim_refnum_by_claim_id.get(c.claim_id)
        cite = f'<a class="cite" href="#ref-{ref}">[{ref}]</a>' if ref else "—"
        status = c.verification.status
        status_cls = {VerificationStatus.SUPPORTED: "ok", VerificationStatus.KILLED_ON_REFUTE: "bad"}.get(status, "neutral")
        rows.append(
            '<tr><td>{claim}</td><td>{value}</td><td>{period}</td><td>{src}</td><td class="status {cls}">{status}</td></tr>'.format(
                claim=esc(first_sentence(c.text or "", max_len=90)),
                value=esc(dr.get("value")),
                period=esc(dr.get("period") or "—"),
                src=cite,
                cls=status_cls,
                status=esc(status),
            )
        )
    return (
        '<section class="structured-section"><h2>Structured data referenced in this report</h2>'
        '<p class="section-intro">Every primary/official data point (EDGAR, FRED, and similar) this '
        "run traced a claim to, independent of whether it made it into the narrative above.</p>"
        '<table class="data-table"><thead><tr><th>Claim</th><th>Value</th><th>Period</th>'
        "<th>Source</th><th>Status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


def render_callouts(requirements, claims, manifest):
    boxes = []
    must_open = [r for r in requirements if r.must_cover and r.kind != RequirementKind.DELIVERABLE and r.terminal_state != RequirementState.COVERED]
    for r in must_open:
        boxes.append(f'<div class="callout warn"><strong>Not fully answered:</strong> {esc(r.text)} <span class="callout-meta">(state: {esc(r.terminal_state or "not_attempted")})</span></div>')

    loop = manifest.get("loop") or {}
    open_conflicts = loop.get("conflicts_open") or 0
    if open_conflicts:
        boxes.append(
            '<div class="callout warn"><strong>Unresolved contradiction'
            f"{'s' if open_conflicts != 1 else ''}:</strong> {open_conflicts} conflicting source span"
            f"{'s remain' if open_conflicts != 1 else ' remains'} unreconciled in the evidence "
            "(see the technical appendix's evidence ledger).</div>"
        )

    flag_counts = {}
    for c in claims:
        for flag in c.gate_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    if flag_counts:
        parts = ", ".join(f"{v} {esc(k)}" for k, v in sorted(flag_counts.items(), key=lambda kv: -kv[1]))
        boxes.append(f'<div class="callout caution"><strong>Caveat flags in the evidence:</strong> {parts}. These claims are not presented as established findings in the narrative above; see the appendix ledger for which ones.</div>')

    killed = (manifest.get("counts") or {}).get("claims_killed") or 0
    if killed:
        boxes.append(f'<div class="callout bad"><strong>Killed on refutation:</strong> {killed} claim{"s" if killed != 1 else ""} failed adversarial verification and were excluded from the narrative above.</div>')

    if not boxes:
        return ""
    return '<section class="callouts-section">' + "".join(boxes) + "</section>"


def render_body(body_md, claim_refnum):
    title, prefix, sections = split_sections(body_md)
    parts = []
    if prefix:
        parts.append(f'<div class="lede">{rewrite_citations(md_to_html(prefix), claim_refnum)}</div>')
    for heading, content in sections:
        takeaway = first_sentence(re.sub(r"[#*`_\[\]()]", "", content)[:400]) if content else ""
        section_html = rewrite_citations(md_to_html(content), claim_refnum)
        takeaway_html = f'<p class="takeaway">{esc(takeaway)}</p>' if takeaway else ""
        css_class = "muted-section" if heading.strip().lower() == "open questions" else ""
        parts.append(f'<section class="body-section {css_class}"><h2>{esc(heading)}</h2>{takeaway_html}{section_html}</section>')
    return title, "\n".join(parts)


def render_references(ordered_sources):
    if not ordered_sources:
        return ""
    items = []
    for i, s in enumerate(ordered_sources, start=1):
        title = s.title or s.url_or_id or s.source_system or "—"
        url = s.url_or_id or ""
        tier = s.authority_tier
        is_real_url = bool(re.match(r"^https?://", url or ""))
        link = f'<a href="{esc(url)}">{esc(title)}</a>' if is_real_url else esc(title)
        meta = " · ".join(
            filter(
                None,
                [
                    esc(s.source_system) if s.source_system else None,
                    f"tier {tier}" if tier else None,
                ],
            )
        )
        meta_html = f' <span class="ref-meta">({meta})</span>' if meta else ""
        items.append(f'<li id="ref-{i}"><span class="ref-num">[{i}]</span> {link}{meta_html}</li>')
    return '<section class="references-section"><h2>References</h2><ol class="references">' + "".join(items) + "</ol></section>"


def render_appendix(manifest, claims, sources, requirements, coverage=()):
    src_by_id = {s.id: s for s in sources}
    counts = manifest.get("counts") or {}
    loop = manifest.get("loop") or {}
    verify_on = "on" if manifest.get("verify_mode") else "off"
    methodology = f"Profile: {esc(manifest.get('profile'))} · depth: {esc(manifest.get('depth'))} · verify: {verify_on} · engine: v{esc(manifest.get('engine_version'))} · generated {esc(manifest.get('timestamp'))}"
    verification = (
        f"{counts.get('claims', 0)} claims · {counts.get('claims_verified', 0)} verified · "
        f"{counts.get('claims_killed', 0)} killed on refutation · {counts.get('claims_flagged', 0)} flagged. "
        f"{loop.get('requirements_covered', 0)}/{loop.get('requirements_must_cover', 0)} must-cover "
        "requirements covered."
    )

    # Evidence column (requirement -> claim ordinals): restored when coverage.jsonl is
    # present (Phase 1 closure, PHASE1-SHARED-CONTRACT.md); the linkage lives in
    # CoverageMapping, so with no coverage this degrades gracefully back to the
    # column-less table (the prior behavior, when that artifact was never written).
    claim_ordinal_by_id = {c.claim_id: i for i, c in enumerate(claims, start=1)}
    mappings_by_req = {}
    for m in coverage:
        mappings_by_req.setdefault(m.requirement_id, []).append(m)

    def _evidence_cell(req_id):
        ordinals = sorted({claim_ordinal_by_id[m.claim_id] for m in mappings_by_req.get(req_id, ()) if m.claim_id in claim_ordinal_by_id})
        return f"<td>{', '.join(f'[{o}]' for o in ordinals) or '—'}</td>"

    evidence_th = "<th>Evidence</th>" if coverage else ""
    req_rows = "".join(
        f"<tr><td>{esc(r.id)}</td><td>{esc(first_sentence(r.text or '', 90))}</td>"
        f"<td>{esc(r.kind)}</td><td>{'yes' if r.must_cover else '—'}</td>"
        f"<td>{esc(r.terminal_state or 'not_attempted')}</td>{_evidence_cell(r.id) if coverage else ''}</tr>"
        for r in requirements
    )
    req_html = ""
    if requirements:
        req_html = f"<h3>Requirement coverage</h3><table class='data-table'><thead><tr><th>Req</th><th>Ask</th><th>Kind</th><th>Must-cover</th><th>State</th>{evidence_th}</tr></thead><tbody>{req_rows}</tbody></table>"

    ledger_rows = []
    for i, c in enumerate(claims, start=1):
        src = src_by_id.get(c.source_id)
        flags = ", ".join(c.gate_flags) or "—"
        if c.data_ref:
            support = "data:" + str(c.data_provenance)
        elif c.support:
            support = effective_relation(c.support.relation_extractor, c.support.relation_reviewer, materiality_of(c.importance))
        else:
            support = "—"
        src_label = (src.title or src.source_system) if src else None
        ledger_rows.append(
            f"<tr><td>{i}</td><td>{esc(first_sentence(c.text or '', 90))}</td>"
            f"<td>{esc(src_label or '—')}</td>"
            f"<td>{esc(src.authority_tier if src else '—')}</td><td>{esc(support)}</td>"
            f"<td>{esc(c.verification.status)}</td><td>{esc(flags)}</td></tr>"
        )
    tier_note = "Tier 1 = primary/peer-reviewed … 4 = weak/forum."

    disclaimers = []
    if not manifest.get("verify_mode"):
        disclaimers.append("Claims here were not independently verified; each is graded by the authority tier of its source.")
    profile = manifest.get("profile")
    if profile == "health":
        disclaimers.append("This is an evidence synthesis for informational purposes, not medical advice.")
    if profile == "legal":
        disclaimers.append("First-pass public sources, not Westlaw/Lexis; verify critical authorities manually.")
    if profile == "financial":
        disclaimers.append("Figures are stated with their data vintage; confirm against the primary filings before relying on them.")

    return (
        '<section class="appendix-section"><h2>Appendix — Methodology, Sources &amp; Validation</h2>'
        f"<p class='methodology'>{methodology}</p>"
        f"<p class='verification'>{verification}</p>"
        f"{req_html}"
        "<h3>Evidence ledger</h3>"
        f"<p class='section-intro'>{tier_note}</p>"
        "<table class='data-table'><thead><tr><th>#</th><th>Claim</th><th>Source</th><th>Tier</th>"
        f"<th>Support</th><th>Status</th><th>Flags</th></tr></thead><tbody>{''.join(ledger_rows)}</tbody></table>" + (f"<p class='disclaimers'>{' '.join(esc(d) for d in disclaimers)}</p>" if disclaimers else "") + "</section>"
    )


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<div class="print-header">{title}</div>
<div class="print-footer">{profile} · {date} · deep-research harness</div>
<header class="page-header">
  <div class="eyebrow">{profile} research report · {date}</div>
  <h1>{title}</h1>
  <p class="question">{question}</p>
</header>
<main>
{cards}
{callouts}
{body}
{structured_table}
{references}
{appendix}
</main>
</body>
</html>
"""

CSS = """
:root { --ink:#1a1a1a; --muted:#5b6470; --line:#dfe3e8; --accent:#1f5f8b; --warn-bg:#fff6e5; --warn-border:#e8b93a;
  --caution-bg:#fdeeea; --caution-border:#d97757; --bad-bg:#fdecec; --bad-border:#c0392b; --ok:#1f7a3f; }
* { box-sizing: border-box; }
body { font-family: Georgia, 'Times New Roman', serif; color: var(--ink); max-width: 860px; margin: 0 auto;
  padding: 2.5rem 1.5rem 6rem; line-height: 1.55; font-size: 16px; }
h1, h2, h3 { font-family: -apple-system, Helvetica, Arial, sans-serif; line-height: 1.25; }
h1 { font-size: 1.9rem; margin: 0.2rem 0 0.4rem; }
h2 { font-size: 1.25rem; margin-top: 2.2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.3rem; }
h3 { font-size: 1.05rem; margin-top: 1.4rem; }
.page-header { border-bottom: 3px solid var(--accent); padding-bottom: 0.8rem; margin-bottom: 1.5rem; }
.eyebrow { font-family: -apple-system, Helvetica, Arial, sans-serif; text-transform: uppercase;
  letter-spacing: 0.06em; font-size: 0.78rem; color: var(--muted); }
.question { color: var(--muted); font-style: italic; }
.lede { font-size: 1.08rem; }
.takeaway { font-family: -apple-system, Helvetica, Arial, sans-serif; font-weight: 600; color: var(--accent);
  background: #f2f7fa; border-left: 3px solid var(--accent); padding: 0.5rem 0.8rem; margin: 0.6rem 0 0.9rem;
  font-size: 0.95rem; break-inside: avoid; }
.muted-section { color: var(--muted); }
a { color: var(--accent); }
a.cite { font-size: 0.82em; text-decoration: none; }
.cards-section { margin-top: 2.2rem; }
.cards { display: flex; flex-wrap: wrap; gap: 0.8rem; font-family: -apple-system, Helvetica, Arial, sans-serif; }
.card { flex: 1 1 200px; border: 1px solid var(--line); border-radius: 6px; padding: 0.9rem 1rem;
  break-inside: avoid; background: #fafbfc; }
.card-value { font-size: 1.6rem; font-weight: 700; color: var(--accent); }
.card-period { font-size: 0.75rem; color: var(--muted); }
.card-label { font-size: 0.85rem; margin-top: 0.3rem; }
.callouts-section { margin: 1.6rem 0; display: flex; flex-direction: column; gap: 0.6rem; }
.callout { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 0.9rem; padding: 0.6rem 0.9rem;
  border-radius: 4px; border: 1px solid; break-inside: avoid; }
.callout.warn { background: var(--warn-bg); border-color: var(--warn-border); }
.callout.caution { background: var(--caution-bg); border-color: var(--caution-border); }
.callout.bad { background: var(--bad-bg); border-color: var(--bad-border); }
.callout-meta { color: var(--muted); }
table.data-table { width: 100%; border-collapse: collapse; font-family: -apple-system, Helvetica, Arial, sans-serif;
  font-size: 0.82rem; margin: 0.8rem 0 1.2rem; }
table.data-table th, table.data-table td { border: 1px solid var(--line); padding: 0.35rem 0.5rem; text-align: left;
  vertical-align: top; }
table.data-table th { background: #f2f4f6; }
table.data-table tr { break-inside: avoid; }
td.status.ok { color: var(--ok); }
td.status.bad { color: var(--bad-border); }
.section-intro { color: var(--muted); font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 0.9rem; }
.references-section, .appendix-section { margin-top: 2.4rem; border-top: 1px solid var(--line); padding-top: 1rem; }
ol.references { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 0.85rem; padding-left: 1.4rem; }
ol.references li { margin-bottom: 0.35rem; }
.ref-meta { color: var(--muted); }
.methodology, .verification, .disclaimers { font-family: -apple-system, Helvetica, Arial, sans-serif;
  font-size: 0.85rem; color: var(--muted); }
.print-header, .print-footer { display: none; }
@media print {
  body { padding-top: 3rem; padding-bottom: 3rem; font-size: 12.5pt; }
  .page-header { display: none; }
  .print-header { display: block; position: fixed; top: 0; left: 0; right: 0; font-family: -apple-system, Helvetica, Arial, sans-serif;
    font-size: 9pt; color: var(--muted); border-bottom: 1px solid var(--line); padding: 0.3rem 1.5rem;
    background: white; }
  .print-footer { display: block; position: fixed; bottom: 0; left: 0; right: 0; font-family: -apple-system, Helvetica, Arial, sans-serif;
    font-size: 8pt; color: var(--muted); border-top: 1px solid var(--line); padding: 0.25rem 1.5rem;
    background: white; }
  /* Only atomic, single-glance elements are kept whole across a page break (cards, callouts, table
     rows, the takeaway strip). A multi-paragraph body/appendix section is left free to flow and split
     normally — forcing a whole long section onto one page just wastes paper with a blank gap before it. */
  h2 { break-after: avoid-page; }
  a { color: var(--ink); text-decoration: none; }
  a.cite { color: var(--accent); }
}
"""


def render(run_dir):
    report_md, manifest, claims, sources, requirements, conflicts, coverage = load_run(run_dir)
    body_md = split_body(report_md)
    src_by_id = {s.id: s for s in sources}
    claim_refnum, ordered_sources = build_claim_refs(claims, sources)
    claim_refnum_by_claim_id = {c.claim_id: claim_refnum[i] for i, c in enumerate(claims, start=1) if i in claim_refnum}

    h1_title, body_html = render_body(body_md, claim_refnum)
    title = h1_title or manifest.get("restated_question") or manifest.get("question") or "Research report"

    return PAGE_TEMPLATE.format(
        title=esc(title),
        css=CSS,
        profile=esc(manifest.get("profile") or "general"),
        date=esc((manifest.get("timestamp") or "")[:10]),
        question=esc(manifest.get("question") or ""),
        cards=render_key_stat_cards(claims, src_by_id, claim_refnum_by_claim_id, conflicts),
        callouts=render_callouts(requirements, claims, manifest),
        body=body_html,
        structured_table=render_structured_table(claims, src_by_id, claim_refnum_by_claim_id),
        references=render_references(ordered_sources),
        appendix=render_appendix(manifest, claims, sources, requirements, coverage),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a run folder produced by write_run.py")
    ap.add_argument("--out", default=None, help="output path (default: <run_dir>/report.html)")
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.run_dir, "report.md")):
        sys.exit(f"render_report.py: {args.run_dir} does not look like a run folder (no report.md)")

    html_out = render(args.run_dir)
    out_path = args.out or os.path.join(args.run_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html_out)
    print(out_path)


if __name__ == "__main__":
    main()

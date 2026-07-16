"""Deterministic report-body generator (D6 ruling D; REVIEW_FINISH_PLAN_2026-07-06.md
Part III). Builds report.md's narrative body directly from the eligible-claim ledger --
no LLM calls, no network, no randomness -- so excluded claims are absent BY CONSTRUCTION
rather than by post-hoc linting (render-body design fork, BUILD_LEDGER.md 2026-07-06).

Acceptance is `dr_core.lint.report_lint` returning zero hard findings over this
generator's own output (the elegant self-consistency spec the ledger already names).

Caller contract: `claims_by_ordinal` MUST already be restricted to the eligible set,
keyed by the frozen `dr_run["citation_ordinals"]` value -- this module never re-derives
eligibility itself (see `dr_core.models.eligibility`); it only renders what it is given.

D10 body hygiene: model-authored claim text is sanitized of em/en dashes at RENDER time
only (this module's output), never on the ledger itself -- claims.jsonl/ledger.jsonl
keep the claim's verbatim text. This exists so a model's own wording can never trip
report_lint's em-dash ban.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from dr_core.models.derive import claim_caveat
from dr_core.models.eligibility import derive_status
from dr_core.models.enums import PublicationStatus, RequirementState
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, Source

_URL_HOST_RE = re.compile(r"^https?://([^/]+)")
_DASH_RE = re.compile(r"[–—]")  # en dash, em dash
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\b[A-Za-z]\.)(?<=[.!?])\s+")


def _sanitize_dashes(text: str) -> str:
    """D10 body hygiene: replace em/en dashes with a plain hyphen at render time only
    (the ledger keeps the claim's verbatim text -- see module docstring)."""
    return _DASH_RE.sub("-", text)


def _clean(text: str | None) -> str:
    return _sanitize_dashes(" ".join((text or "").split()))


def _domain(url: str) -> str:
    m = _URL_HOST_RE.match(url or "")
    return m.group(1) if m else url


def _source_label(source: Source | None) -> str:
    if source is None:
        return "an unspecified source"
    if source.title:
        return _clean(source.title)
    if source.url_or_id:
        return _domain(source.url_or_id)
    return source.source_system or "an unspecified source"


def _claim_sentence(
    ordinal: int,
    claim: Claim,
    source: Source | None,
    mappings_for_claim: Sequence[CoverageMapping] = (),
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> str:
    """One sentence per eligible claim, carrying its `[n]` marker. Contested
    claims carry `claim_caveat(claim)` verbatim (their only citation is this one,
    so it is always the "first citation"); not_verified claims render attributed
    ("According to <source>, ...") per the D6-D render policy."""
    status = derive_status(claim, mappings_for_claim, requirements_by_id)
    text = _clean(claim.text) or "(no claim text recorded)"
    parts = [part.strip().rstrip(".") for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if status == PublicationStatus.NOT_VERIFIED:
            label = _source_label(source)
            body = part[0].lower() + part[1:] if len(part) > 1 else part.lower()
            rendered.append(f"According to {label}, {body} [{ordinal}].")
            continue
        sentence = f"{part} [{ordinal}]."
        if status == PublicationStatus.CONTESTED and index == 0:
            caveat = claim_caveat(claim)
            if caveat:
                sentence = f"{part} [{ordinal}] ({caveat})."
        rendered.append(sentence)
    return " ".join(rendered)


def _table_cell(value: object) -> str:
    return _clean(str(value)).replace("|", "\\|")


def _structured_findings_table(claims_by_ordinal: Mapping[int, Claim]) -> tuple[list[str], set[int]]:
    """Render dense structured evidence as a readable body table.

    Eight numeric claims trigger report_lint's table requirement. Keeping the
    threshold here aligned with the linter prevents deterministic data runs from
    degrading into dozens of one-line paragraphs.
    """
    structured = [
        (ordinal, claim)
        for ordinal, claim in sorted(claims_by_ordinal.items())
        if claim.data_ref and re.search(r"\d", claim.text or "")
    ]
    if len(structured) < 8:
        return [], set()

    lines = ["| Period | Finding | Value | Evidence |", "|---|---|---:|---:|"]
    for ordinal, claim in structured:
        data_ref = claim.data_ref or {}
        lines.append(
            "| {period} | {finding} | {value} | [{ordinal}] |".format(
                period=_table_cell(data_ref.get("period") or "-"),
                finding=_table_cell(claim.text),
                value=_table_cell(data_ref.get("value") if data_ref.get("value") is not None else "-"),
                ordinal=ordinal,
            )
        )
    return lines, {ordinal for ordinal, _ in structured}


def _title(dr_run: Mapping) -> str:
    question = _clean(dr_run.get("question"))
    return f"Research report: {question}" if question else "Research report"


def _executive_summary(
    claims_by_ordinal: Mapping[int, Claim],
    sources: Mapping[str, Source],
    dr_run: Mapping,
    mappings_by_claim: Mapping[str, Sequence[CoverageMapping]] | None = None,
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> list[str]:
    mappings_by_claim = mappings_by_claim or {}
    lines: list[str] = []
    n_claims = len(claims_by_ordinal)
    n_sources = len(sources)
    if n_claims:
        lines.append(f"- {n_claims} substantiated claim(s) drawn from {n_sources} source(s) are presented below.")
    else:
        lines.append("- No recorded claim met the eligibility bar for inclusion in this report.")
        if n_sources:
            titles = ", ".join(_source_label(s) for s in sources.values())
            lines.append(f"- {n_sources} source(s) were consulted: {titles}.")
        else:
            lines.append("- No sources were consulted.")

    def _status(claim: Claim) -> PublicationStatus:
        return derive_status(claim, mappings_by_claim.get(claim.claim_id, ()), requirements_by_id)

    contested = sum(1 for c in claims_by_ordinal.values() if _status(c) == PublicationStatus.CONTESTED)
    not_verified = sum(1 for c in claims_by_ordinal.values() if _status(c) == PublicationStatus.NOT_VERIFIED)
    if contested:
        lines.append(f"- {contested} claim(s) carry an explicit caveat noted at first citation.")
    if not_verified:
        lines.append(f"- {not_verified} claim(s) are presented in attributed form pending independent verification.")
    if dr_run.get("stop_reason"):
        lines.append("- This report stopped short of full coverage; see Unsubstantiated below for the open gaps.")
    return lines


def _unsubstantiated_section(
    ineligible: Sequence[tuple[str, Claim, str]],
    dr_run: Mapping,
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> list[str]:
    """D6-D machinery, extended by D9: on top of the ineligible-claim gaps
    already listed, also names each ACTIVE must-cover requirement the gate
    froze as not fully COVERED (`dr_run["must_cover_states"]`, id + text +
    evidence state) -- the frozen set from the gate's own decision, not a
    re-derivation of "active" at render time."""
    requirements_by_id = requirements_by_id or {}
    lines = ["## Unsubstantiated", ""]
    if not ineligible:
        lines.append("- No additional claims were recorded.")
    else:
        for claim_id, _claim, reason in ineligible:
            lines.append(f"- Claim `{claim_id}` was not included in the findings above ({reason}).")

    must_cover_states: Mapping[str, str] = dr_run.get("must_cover_states") or {}
    open_items = sorted((req_id, state) for req_id, state in must_cover_states.items() if state != RequirementState.COVERED.value)
    for req_id, state in open_items:
        # Bullet-prefixed, like every other line in this section (`-` exempts
        # it from the linter's cite-required rule, same as the ineligible-claim
        # bullets above -- these are gap disclosures, not new factual assertions).
        requirement = requirements_by_id.get(req_id)
        text = requirement.text if requirement is not None else req_id
        lines.append(f"- Required item not yet fully covered: requirement `{req_id}` ({text}) -- evidence state: `{state}`.")
    return lines


def _conclusion(claims_by_ordinal: Mapping[int, Claim]) -> str:
    if claims_by_ordinal:
        return "The findings above reflect only claims that met this report's eligibility bar; contested and attributed items are marked accordingly."
    return "No claim in this pass met the eligibility bar for inclusion; the sources reviewed are listed above for reference."


def generate_body(
    claims_by_ordinal: Mapping[int, Claim],
    sources: Mapping[str, Source],
    dr_run: Mapping,
    *,
    ineligible: Sequence[tuple[str, Claim, str]] = (),
    mappings_by_claim: Mapping[str, Sequence[CoverageMapping]] | None = None,
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> str:
    """Deterministic report.md body: `# Title`, `## Executive summary`, per-claim
    findings under `## Findings` (eligible claims only, ordinal order), an
    optional `## Unsubstantiated` section when `dr_run` carries a `stop_reason`,
    and a citation-exempt `## Conclusion`. Zero-eligible still emits all
    mandatory sections (the linter requires them).

    D9: `mappings_by_claim`/`requirements_by_id` feed the SAME materiality
    derivation the gate used (D6-B), so a claim's rendered status here can
    never disagree with the gate's eligibility decision."""
    dr_run = dr_run or {}
    mappings_by_claim = mappings_by_claim or {}
    lines: list[str] = [f"# {_title(dr_run)}", "", "## Executive summary", ""]
    lines.extend(_executive_summary(claims_by_ordinal, sources, dr_run, mappings_by_claim, requirements_by_id))
    lines.append("")

    if claims_by_ordinal:
        lines.append("## Findings")
        lines.append("")
        table_lines, table_ordinals = _structured_findings_table(claims_by_ordinal)
        if table_lines:
            lines.extend(table_lines)
            lines.append("")
        for ordinal in sorted(claims_by_ordinal):
            if ordinal in table_ordinals:
                continue
            claim = claims_by_ordinal[ordinal]
            source = sources.get(claim.source_id)
            lines.append(_claim_sentence(ordinal, claim, source, mappings_by_claim.get(claim.claim_id, ()), requirements_by_id))
            lines.append("")

    if dr_run.get("stop_reason"):
        lines.extend(_unsubstantiated_section(ineligible, dr_run, requirements_by_id))
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(_conclusion(claims_by_ordinal))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"

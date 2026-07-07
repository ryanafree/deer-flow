"""Deterministic report-body generator (D6 ruling D; REVIEW_FINISH_PLAN_2026-07-06.md
Part III). Builds report.md's narrative body directly from the eligible-claim ledger --
no LLM calls, no network, no randomness -- so excluded claims are absent BY CONSTRUCTION
rather than by post-hoc linting (render-body design fork, BUILD_LEDGER.md 2026-07-06).

Acceptance is `dr_core.lint.report_lint` returning zero hard findings over this
generator's own output (the elegant self-consistency spec the ledger already names).

Caller contract: `claims_by_ordinal` MUST already be restricted to the eligible set,
keyed by the frozen `dr_run["citation_ordinals"]` value -- this module never re-derives
eligibility itself (see `dr_core.models.eligibility`); it only renders what it is given.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from dr_core.models.derive import claim_caveat
from dr_core.models.eligibility import derive_status
from dr_core.models.enums import PublicationStatus
from dr_core.models.ledger import Claim, Source

_URL_HOST_RE = re.compile(r"^https?://([^/]+)")


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


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


def _claim_sentence(ordinal: int, claim: Claim, source: Source | None) -> str:
    """One sentence per eligible claim, carrying its `[n]` marker. Contested
    claims carry `claim_caveat(claim)` verbatim (their only citation is this one,
    so it is always the "first citation"); not_verified claims render attributed
    ("According to <source>, ...") per the D6-D render policy."""
    status = derive_status(claim)
    text = _clean(claim.text).rstrip(".") or "(no claim text recorded)"
    if status == PublicationStatus.NOT_VERIFIED:
        label = _source_label(source)
        body = text[0].lower() + text[1:] if len(text) > 1 else text.lower()
        return f"According to {label}, {body} [{ordinal}]."
    sentence = f"{text} [{ordinal}]."
    if status == PublicationStatus.CONTESTED:
        caveat = claim_caveat(claim)
        if caveat:
            sentence = f"{text} [{ordinal}] ({caveat})."
    return sentence


def _title(dr_run: Mapping) -> str:
    question = _clean(dr_run.get("question"))
    return f"Research report: {question}" if question else "Research report"


def _executive_summary(
    claims_by_ordinal: Mapping[int, Claim],
    sources: Mapping[str, Source],
    dr_run: Mapping,
) -> list[str]:
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

    contested = sum(1 for c in claims_by_ordinal.values() if derive_status(c) == PublicationStatus.CONTESTED)
    not_verified = sum(1 for c in claims_by_ordinal.values() if derive_status(c) == PublicationStatus.NOT_VERIFIED)
    if contested:
        lines.append(f"- {contested} claim(s) carry an explicit caveat noted at first citation.")
    if not_verified:
        lines.append(f"- {not_verified} claim(s) are presented in attributed form pending independent verification.")
    if dr_run.get("stop_reason"):
        lines.append("- This report stopped short of full coverage; see Unsubstantiated below for the open gaps.")
    return lines


def _unsubstantiated_section(ineligible: Sequence[tuple[str, Claim, str]]) -> list[str]:
    lines = ["## Unsubstantiated", ""]
    if not ineligible:
        lines.append("- No additional claims were recorded.")
    else:
        for claim_id, _claim, reason in ineligible:
            lines.append(f"- Claim `{claim_id}` was not included in the findings above ({reason}).")
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
) -> str:
    """Deterministic report.md body: `# Title`, `## Executive summary`, per-claim
    findings under `## Findings` (eligible claims only, ordinal order), an
    optional `## Unsubstantiated` section when `dr_run` carries a `stop_reason`,
    and a citation-exempt `## Conclusion`. Zero-eligible still emits all
    mandatory sections (the linter requires them)."""
    dr_run = dr_run or {}
    lines: list[str] = [f"# {_title(dr_run)}", "", "## Executive summary", ""]
    lines.extend(_executive_summary(claims_by_ordinal, sources, dr_run))
    lines.append("")

    if claims_by_ordinal:
        lines.append("## Findings")
        lines.append("")
        for ordinal in sorted(claims_by_ordinal):
            claim = claims_by_ordinal[ordinal]
            source = sources.get(claim.source_id)
            lines.append(_claim_sentence(ordinal, claim, source))
            lines.append("")

    if dr_run.get("stop_reason"):
        lines.extend(_unsubstantiated_section(ineligible))
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    lines.append(_conclusion(claims_by_ordinal))
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"

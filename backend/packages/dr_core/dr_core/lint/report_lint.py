#!/usr/bin/env python3
"""Deterministic report-quality lint. Usage: report_lint.py <run-folder>

PART 1 is ported VERBATIM from harness/evals/report_lint.py: title/exec-summary/
conclusion presence, the BANNED pipeline-vocab list, the em-dash ban, the 3+-citations-
in-one-sentence check, the paragraph p90 word-count check, and the numeric-claims-
without-table heuristic. That last check originally read claims.jsonl the OLD way -
`json.loads(line).get("claim")` - the harness's plain-text-claim shape; against a real
dr_core dump (field `text`, per PHASE1-SHARED-CONTRACT.md) that silently never matched
anything, so the check never fired (task-phase1-consolidate.md item 3). It now reads
`text` first, falling back to `claim` so the legacy fixture in test_report_lint.py -
ported verbatim from the original harness test and never upgraded to the ledger schema
- still parses.

PART 2 is new: sentence-level publication-eligibility enforcement against the claim
ledger, per the D2 ruling section C ("the linter is the backstop to the eligibility_gate
node"). It only activates when claims.jsonl parses as the full dr_core.models ledger
schema (one Claim.model_dump(mode="json") per line, per build-logs/PHASE1-SHARED-
CONTRACT.md); a claims.jsonl in the legacy `{"claim": text}` shape - like the PART 1
fixtures ported from the original harness test - fails that parse and eligibility is
simply skipped for that run folder, falling back to PART 1 checks only. That is the
reconciliation between the old and new schemas: PART 1's file predates the ledger schema
and its own fixtures cannot be upgraded in place without breaking the ported test.

Eligibility enforcement never re-derives PublicationStatus or the caveat table locally -
it calls dr_core.models.derive.publication_status / claim_caveat. materiality and
grounded (both required by publication_status but not stored on Claim) are derived from
claim.importance alone via materiality_of + is_grounded, since the linter only has
report.md + claims.jsonl (+ an optional conflicts.jsonl) - no requirements.jsonl or
coverage mappings to derive materiality the fuller way derived_materiality does.
"""

import json
import os
import re
import sys

from dr_core.connectors.registry import load_connectors
from dr_core.models import Claim, Conflict, CoverageMapping, PublicationStatus, Requirement, Source, claim_caveat, is_grounded, materiality_of, publication_status
from dr_core.plan.evidence_class import resolve_evidence_class

BANNED = [
    "this run",
    "gate flag",
    "gate_flags",
    "killed on refut",
    "not-verified",
    "claim ledger",
    "seeded-trap",
    "checkpoint",
    "verification apparatus",
    "worth holding in tension",
    "useful corrective",
    "the honest state of",
    "should be weighted accordingly",
    "papering over",
    "worth carrying forward",
]
APPENDIX_RE = re.compile(r"^##+\s*Appendix", re.M)
CONCLUSION_RE = re.compile(r"^##\s+Conclusion\b", re.M)
CITATION_RE = re.compile(r"\[(\d+)\]")

# PART 2 sentence splitter (not the PART 1 verbatim port at line ~221): a bare
# `(?<=[.!?])\s+` treats any single-letter abbreviation ending in a period -- "v."
# in a case citation ("Harlow v. Fitzgerald"), "U.S."/"S." in a reporter citation --
# as a sentence boundary, which fires a false [cite-required] finding on a legal
# claim whose citation marker is at the TRUE end of the sentence (found live on a
# legal-profile trap run, 2026-07-08). The negative lookbehind skips a split
# immediately after a lone letter + period; genuine sentence ends (multi-letter
# words) are unaffected.
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\b[A-Za-z]\.)(?<=[.!?])\s+")

# Best-effort heuristic for PART 2 rule 5 (not_verified claims must read as attributed or
# hedged). Catches common surface forms; not a real NLP classifier - see
# _looks_attributed_or_hedged for its documented limits.
ATTRIBUTION_CUE_RE = re.compile(
    r"\b(according to|reportedly|sources? (?:say|indicate)|is (?:said|reported|claimed) to|"
    r"suggests?|researchers? (?:say|note|report)|(?:the )?(?:study|report|source) (?:suggests?|indicates?|notes?))\b",
    re.I,
)
HEDGE_END_RE = re.compile(r"(?:may|might|possibly|unclear|unconfirmed|uncertain)\.?\s*$", re.I)


def body_of(md):
    match = APPENDIX_RE.search(md)
    return md[: match.start()] if match else md


# ---------------------------------------------------------------------------
# PART 2 helpers
# ---------------------------------------------------------------------------


def _load_claims(folder):
    """Parse claims.jsonl as the full dr_core.models ledger schema, keyed by the 1-based
    line index (the citation ordinal, per PHASE1-SHARED-CONTRACT.md). Returns None if the
    file is absent or ANY line fails to validate as a Claim - the legacy-schema signal
    that lets PART 1's ported fixtures keep passing (see module docstring)."""
    path = os.path.join(folder, "claims.jsonl")
    if not os.path.exists(path):
        return None
    claims: dict[int, Claim] = {}
    with open(path) as fh:
        for i, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                claims[i] = Claim.model_validate(json.loads(line))
            except Exception:
                return None
    return claims or None


def _load_conflicts(folder):
    path = os.path.join(folder, "conflicts.jsonl")
    if not os.path.exists(path):
        return []
    conflicts = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                conflicts.append(Conflict.model_validate(json.loads(line)))
    return conflicts


def _publication_status_of(claim, conflicts):
    materiality = materiality_of(claim.importance)
    grounded = is_grounded(claim, materiality, post_verify=True)
    return publication_status(claim, materiality=materiality, grounded=grounded, conflicts=conflicts)


def _looks_attributed_or_hedged(sentence):
    """PART 2 rule 5's heuristic: an explicit attribution cue, a trailing hedge word, or
    a question mark all read as expressly-attributed / open-question framing. Limits:
    misses attribution phrased outside these surface forms, and can pass a sentence that
    merely contains a cue word without genuinely hedging its claim."""
    s = sentence.strip()
    if not s or s.endswith("?"):
        return True
    return bool(ATTRIBUTION_CUE_RE.search(s) or HEDGE_END_RE.search(s))


def _is_factual_sentence(sentence):
    s = sentence.strip()
    if len(s.split()) <= 5:
        return False
    return not s.startswith(("#", "|", "-", "*", ">"))


def _strip_structural_lines(segment):
    """Drops heading/table/list/blockquote lines before sentence-splitting so a heading
    glued to the next sentence (no terminal punctuation between them) can't glom onto it
    and get misread as a single non-factual, heading-prefixed chunk. Broader than PART 1's
    table-only strip (used only by the 3+-citations rule) because PART 2 needs individual
    sentences, not just citation density."""
    lines = [line for line in segment.splitlines() if not line.lstrip().startswith(("#", "|", "-", "*", ">"))]
    return "\n".join(lines)


def _load_model_jsonl(path, model_cls):
    """Parse a JSONL file as ``model_cls`` instances, one per line. Returns None
    if the file is absent or any line fails to validate -- same skip-the-check
    contract as ``_load_claims``."""
    if not os.path.exists(path):
        return None
    items = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                items.append(model_cls.model_validate(json.loads(line)))
            except Exception:
                return None
    return items


def _source_class_mismatch_findings(folder):
    """PART 3 (Stage 3, SPEC_evidence_routing_2026-07-10.md): a deterministic,
    ledger-level cross-check -- never a sentence-level scan of report.md, unlike
    PART 2 -- flagging a requirement/claim/source triple where the requirement
    names a specific evidence_class but the claim's cited source resolves to a
    different one. WARNING only (settled question 9: this is a lint signal, not
    a verify vote or hard gate); skipped entirely (returns []) unless
    requirements.jsonl, coverage.jsonl, sources.jsonl, and claims.jsonl (keyed by
    claim_id, not citation ordinal) all parse against the dr_core ledger schema."""
    requirements = _load_model_jsonl(os.path.join(folder, "requirements.jsonl"), Requirement)
    coverage = _load_model_jsonl(os.path.join(folder, "coverage.jsonl"), CoverageMapping)
    sources = _load_model_jsonl(os.path.join(folder, "sources.jsonl"), Source)
    claims = _load_model_jsonl(os.path.join(folder, "claims.jsonl"), Claim)
    if requirements is None or coverage is None or sources is None or claims is None:
        return []

    requirements_by_id = {r.id: r for r in requirements}
    sources_by_id = {s.id: s for s in sources}
    claims_by_id = {c.claim_id: c for c in claims}
    connectors_by_name = {c.name: c for c in load_connectors()}

    findings = []
    for mapping in coverage:
        requirement = requirements_by_id.get(mapping.requirement_id)
        if requirement is None or requirement.evidence_class == "any":
            continue
        claim = claims_by_id.get(mapping.claim_id)
        if claim is None:
            continue
        source = sources_by_id.get(claim.source_id)
        if source is None:
            continue
        resolved = resolve_evidence_class(source.source_system, source.url_or_id, connectors_by_name)
        if resolved != requirement.evidence_class:
            findings.append(
                f"[source-class-mismatch] requirement {requirement.id} expects {requirement.evidence_class} "
                f"evidence but claim {claim.claim_id} (via requirement mapping) cites a {resolved}-class "
                f"source ({source.source_system}): {claim.text[:80]!r}"
            )
    return findings


def _split_conclusion(body):
    """Sentences in the Conclusion section synthesize already-cited claims rather than
    asserting new ones, so PART 2 rule 1 (every factual sentence needs >=1 citation)
    exempts them; citations that DO appear there are still checked against rules 2-5."""
    match = CONCLUSION_RE.search(body)
    if not match:
        return body, ""
    return body[: match.start()], body[match.start() :]


def _eligibility_findings(body, claims, conflicts):
    hard, warn = [], []
    status_cache = {ordinal: _publication_status_of(claim, conflicts) for ordinal, claim in claims.items()}
    first_seen: set[int] = set()

    def scan(segment, *, require_citation):
        for raw_sentence in _SENTENCE_SPLIT_RE.split(_strip_structural_lines(segment)):
            s = raw_sentence.strip()
            if not _is_factual_sentence(s):
                continue
            markers = [int(m) for m in CITATION_RE.findall(s)]
            if not markers:
                if require_citation:
                    hard.append(f"[cite-required] factual sentence has no claim-keyed citation: {s[:80]!r}")
                continue
            for ordinal in markers:
                claim = claims.get(ordinal)
                if claim is None:
                    hard.append(f"[unresolved-citation] citation [{ordinal}] does not resolve to a claim in claims.jsonl: {s[:80]!r}")
                    continue
                status = status_cache[ordinal]
                is_first = ordinal not in first_seen
                first_seen.add(ordinal)
                if status == PublicationStatus.EXCLUDED:
                    hard.append(f"[excluded-in-body] excluded claim [{ordinal}] cited in body: {s[:80]!r}")
                elif status == PublicationStatus.CONTESTED and is_first:
                    caveat = claim_caveat(claim)
                    if caveat and caveat not in s:
                        hard.append(f"[missing-caveat] contested claim [{ordinal}] missing caveat {caveat!r} at first citation: {s[:80]!r}")
                elif status == PublicationStatus.NOT_VERIFIED and not _looks_attributed_or_hedged(s):
                    warn.append(f"[unattributed-not-verified] not_verified claim [{ordinal}] cited without attribution/hedging (heuristic, WARN): {s[:80]!r}")

    main_body, conclusion_body = _split_conclusion(body)
    scan(main_body, require_citation=True)
    scan(conclusion_body, require_citation=False)
    return hard, warn


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: report_lint.py <run-folder>")
    folder = sys.argv[1]
    with open(os.path.join(folder, "report.md")) as fh:
        body = body_of(fh.read())
    findings = []
    warnings = []

    # ---- PART 1: structural checks (verbatim from harness/evals/report_lint.py) ----
    if not re.search(r"^#\s+\S", body, re.M):
        findings.append("no top-level title")
    if "## Executive summary" not in body[:2000]:
        findings.append("no executive summary near the top")
    if not re.search(r"^##\s+Conclusion", body, re.M):
        findings.append("no conclusion section")
    low = body.lower()
    for phrase in BANNED:
        if phrase in low:
            findings.append(f"banned phrase in body: {phrase!r}")
    if "—" in body:
        findings.append("em dash in body")
    body_without_tables = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("|"))
    for sentence in re.split(r"(?<=[.!?])\s+", body_without_tables):
        if len(re.findall(r"\[\d+\]", sentence)) >= 3:
            findings.append(f"3+ citation markers in one sentence: {sentence[:80]!r}")
            break
    paragraphs = [p for p in body.split("\n\n") if len(p.split()) > 5 and not p.lstrip().startswith(("#", "|", "-", "*", ">"))]
    if paragraphs:
        lengths = sorted(len(p.split()) for p in paragraphs)
        p90 = lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))]
        if p90 > 120:
            findings.append(f"paragraph p90 = {p90} words (max 120)")
    claims_path = os.path.join(folder, "claims.jsonl")
    if os.path.exists(claims_path):
        # Real dumps (PHASE1-SHARED-CONTRACT.md) serialize Claim with field `text`, not
        # `claim` — the original harness fixture shape. Read `text` first and fall back
        # to `claim` so the legacy fixture in test_report_lint.py still parses; without
        # the fallback this check silently never fires on real run folders.
        with open(claims_path) as fh:
            numeric = 0
            for line in fh:
                obj = json.loads(line)
                if re.search(r"\d", obj.get("text") or obj.get("claim") or ""):
                    numeric += 1
        if numeric >= 8 and not re.search(r"^\|.+\|\s*$", body, re.M):
            findings.append(f"{numeric} numeric claims but no body table")

    # ---- PART 2: sentence-level publication eligibility (D2 ruling section C) ----
    claims = _load_claims(folder)
    if claims is not None:
        conflicts = _load_conflicts(folder)
        hard, warn = _eligibility_findings(body, claims, conflicts)
        findings.extend(hard)
        warnings.extend(warn)

    # ---- PART 3: source-class-mismatch (Stage 3, SPEC_evidence_routing_2026-07-10.md) ----
    # WARNING only -- appended to `warnings`, never `findings`, so this new check
    # cannot change `main()`'s exit code (settled question 9).
    warnings.extend(_source_class_mismatch_findings(folder))

    for finding in findings:
        print(f"LINT: {finding}")
    for warning in warnings:
        print(f"LINT-WARN: {warning}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())

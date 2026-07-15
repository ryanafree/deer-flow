"""verify_node -- the D8 outer-graph verification node (research -> verify -> gate).

Owns the I/O and shared-budget bookkeeping; the actual decisions live in the pure/
independently-tested `dr_core.verify` modules (`citation`, `provenance`, `selection`,
`votes`). Skips claims already complete/terminal (idempotent re-entry: a repeat pass
over the same ledger touches nothing new). For every remaining claim, runs the
deterministic citation gate and provenance audit first (no vote budget consumed, D8
ordering), then selects up to `MAX_VERIFY` claims (depth-keyed) for the adaptive 1-3
vote protocol under a shared `MAX_VOTE_CALLS = 3*MAX_VERIFY` budget. D11 item 3: there
is no post-verify provenance rescue -- data_provenance is decided solely by the
deterministic audit in step 2 below. Every network call (citation lookup, source-evidence
fetch) has a hard timeout and degrades on failure rather than raising or stalling
(D8 decision 6). Returns one full-claim payload per touched claim through
`dr_claims` -- `merge_ledger`'s transition guards are the enforcement backstop for
the advance-only ladders; this node's writes are advance-only by construction.
"""

from __future__ import annotations

import asyncio
import urllib.error
import urllib.request

from dr_core.models.derive import reopens_requirement
from dr_core.models.enums import CitationStatus, CoverageRelation, VerificationStatus
from dr_core.models.ledger import Claim, CoverageMapping, Requirement
from dr_core.verify.citation import CITE_RE, decide_citation_status, lookup_citations
from dr_core.verify.provenance import audit_provenance
from dr_core.verify.selection import max_verify_for_depth, select_verification_claim_ids
from dr_core.verify.votes import verify_claim

_VOTE_CONCURRENCY = 3
_EVIDENCE_TIMEOUT = 15.0
_EVIDENCE_MAX_CHARS = 6000
_EVIDENCE_READ_CAP = _EVIDENCE_MAX_CHARS * 4  # bound the read() itself, not just the slice


def _is_terminal(claim: Claim) -> bool:
    return claim.verification.complete or claim.verification.status == VerificationStatus.KILLED_ON_REFUTE


def _must_cover_sole_supporter_ids(dr_requirements: dict, dr_coverage: dict) -> set[str]:
    """D9: claim ids that are the SOLE direct supporter of an active
    must-cover requirement. Reuses ``derive.reopens_requirement`` verbatim --
    "the only direct support a kill would reopen" is exactly "the sole direct
    supporter" -- rather than reimplementing the same check."""
    requirements = {req_id: Requirement.model_validate(payload) for req_id, payload in dr_requirements.items()}
    mappings_by_req: dict[str, list[CoverageMapping]] = {}
    for payload in dr_coverage.values():
        mapping = CoverageMapping.model_validate(payload)
        mappings_by_req.setdefault(mapping.requirement_id, []).append(mapping)

    sole_ids: set[str] = set()
    for req_id, requirement in requirements.items():
        if not requirement.must_cover:
            continue
        mappings_for_req = mappings_by_req.get(req_id, [])
        direct_claim_ids = {m.claim_id for m in mappings_for_req if m.relation == CoverageRelation.DIRECT}
        for claim_id in direct_claim_ids:
            if reopens_requirement(claim_id, mappings_for_req):
                sole_ids.add(claim_id)
    return sole_ids


# D10 addendum: the S10 re-acceptance saw live 403s from primary .gov sources on the
# bare "deerflow-dr-verify/1.0" User-Agent -- a default-urllib-UA block, not a real
# access restriction. A browser-like UA/Accept pair clears those without changing the
# no-raise/timeout contract; fewer failed fetches also means fewer evidence-absent
# votes overall (D10's other hardening targets the votes that still land there).
_EVIDENCE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _fetch_evidence_sync(url: str) -> str | None:
    """Blocking excerpt fetch -- only ever invoked via `asyncio.to_thread`. Any
    failure (network error, timeout, non-decodable body) returns None rather than
    raising; the caller folds that into a "source_unreachable" risk reason, never
    a stall."""
    try:
        req = urllib.request.Request(url, headers=_EVIDENCE_HEADERS)
        with urllib.request.urlopen(req, timeout=_EVIDENCE_TIMEOUT) as resp:
            raw = resp.read(_EVIDENCE_READ_CAP)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return raw.decode("utf-8", errors="replace")[:_EVIDENCE_MAX_CHARS]


async def _fetch_evidence(url: str) -> str | None:
    try:
        return await asyncio.to_thread(_fetch_evidence_sync, url)
    except Exception:
        return None


def _new_usage_totals() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}


def _merge_usage(totals: dict[str, int], delta: dict[str, int]) -> None:
    for key in totals:
        totals[key] += delta.get(key, 0)


async def verify_node(state) -> dict:
    dr_claims_raw: dict = state.get("dr_claims") or {}
    dr_sources: dict = state.get("dr_sources") or {}
    dr_requirements: dict = state.get("dr_requirements") or {}
    dr_coverage: dict = state.get("dr_coverage") or {}
    dr_run: dict = state.get("dr_run") or {}

    claims: dict[str, Claim] = {claim_id: Claim.model_validate(payload) for claim_id, payload in dr_claims_raw.items()}
    # D11 run scoping: prior-run claims are never eligible or rendered this
    # run (gate/render baseline filter), so never spend votes on them either.
    baseline_claim_ids = set(dr_run.get("baseline_claim_ids") or [])
    active: dict[str, Claim] = {claim_id: claim for claim_id, claim in claims.items() if claim_id not in baseline_claim_ids and not _is_terminal(claim)}

    touched: dict[str, dict] = {}

    def _touch(claim: Claim) -> None:
        touched[claim.claim_id] = claim.model_dump(mode="json")

    # 1. Citation gate: deterministic, every active claim, no vote budget consumed.
    # D10: token resolution is lookup_citations' own job (COURTLISTENER_TOKEN, with
    # COURTLISTENER_API_TOKEN as a fallback spelling) -- this node no longer reads the
    # env var itself, which used to only check the fallback spelling and never the
    # primary one.
    for claim in active.values():
        if claim.citation_status != CitationStatus.UNRESOLVED or not CITE_RE.search(claim.text or ""):
            continue
        cites = await lookup_citations(claim.text)
        decision = decide_citation_status(cites)
        if decision is not None:
            claim.citation_status = decision
            _touch(claim)

    # 2. Provenance audit: deterministic, data_ref claims only, no vote budget consumed.
    for claim in active.values():
        decision = audit_provenance(claim)
        if decision is not None:
            claim.data_provenance = decision
            _touch(claim)

    # 3. Select up to MAX_VERIFY claims (depth-keyed) for the adaptive vote pass.
    depth = dr_run.get("depth") or "standard"
    max_verify = max_verify_for_depth(depth)
    selected_ids = select_verification_claim_ids(active, dr_sources, dr_requirements, dr_coverage, max_verify)
    max_vote_calls = 3 * max_verify
    sole_supporter_ids = _must_cover_sole_supporter_ids(dr_requirements, dr_coverage)

    votes_used = 0
    budget_lock = asyncio.Lock()

    async def _reserve(n: int) -> bool:
        nonlocal votes_used
        async with budget_lock:
            if votes_used + n > max_vote_calls:
                return False
            votes_used += n
            return True

    usage_totals = _new_usage_totals()
    evidence_cache: dict[str, str | None] = {}
    semaphore = asyncio.Semaphore(_VOTE_CONCURRENCY)

    async def _evidence_for(source: dict | None) -> str | None:
        if not source:
            return None
        url = source.get("url_or_id")
        if not url:
            return None
        if url not in evidence_cache:
            evidence_cache[url] = await _fetch_evidence(url)
        return evidence_cache[url]

    async def _verify_one(claim_id: str) -> None:
        claim = active[claim_id]
        if claim.citation_status == CitationStatus.NOT_FOUND:
            return  # citation gate already excluded it -- skip the vote pass entirely (D8)
        source = dr_sources.get(claim.source_id)
        async with semaphore:
            evidence = await _evidence_for(source)
            record, usage = await verify_claim(claim, source, evidence, reserve_votes=_reserve, sole_must_cover_supporter=claim_id in sole_supporter_ids)
        _merge_usage(usage_totals, usage)
        claim.verification = record
        _touch(claim)

    await asyncio.gather(*(_verify_one(claim_id) for claim_id in selected_ids))

    # D11 item 3 (DECISIONS.md): the post-verify SUPPORTED+UNAUDITED -> MATCHED
    # rescue (dr.js:2693) that used to run here is REMOVED. Text-only voters
    # never compare values, so the rescue laundered a vote outcome into a
    # provenance assertion the votes never made -- a D8 sub-clause reversal.
    # A data_ref claim whose deterministic audit (step 2, above) could not
    # affirmatively match or contradict its own asserted value now stays
    # UNAUDITED permanently, regardless of verification outcome.

    result: dict = {"dr_run": {"verify_mode": "adaptive_1_3", "verify_usage": usage_totals}}
    if touched:
        result["dr_claims"] = touched
    return result

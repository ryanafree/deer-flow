"""verify_node -- the D8 outer-graph verification node (research -> verify -> gate).

Owns the I/O and shared-budget bookkeeping; the actual decisions live in the pure/
independently-tested `dr_core.verify` modules (`citation`, `provenance`, `selection`,
`votes`). Skips claims already complete/terminal (idempotent re-entry: a repeat pass
over the same ledger touches nothing new). For every remaining claim, runs the
deterministic citation gate and provenance audit first (no vote budget consumed, D8
ordering), then selects up to `MAX_VERIFY` claims (depth-keyed) for the adaptive 1-3
vote protocol under a shared `MAX_VOTE_CALLS = 3*MAX_VERIFY` budget, then applies the
post-verify provenance rescue. Every network call (citation lookup, source-evidence
fetch) has a hard timeout and degrades on failure rather than raising or stalling
(D8 decision 6). Returns one full-claim payload per touched claim through
`dr_claims` -- `merge_ledger`'s transition guards are the enforcement backstop for
the advance-only ladders; this node's writes are advance-only by construction.
"""

from __future__ import annotations

import asyncio
import os
import urllib.error
import urllib.request

from dr_core.models.enums import CitationStatus, VerificationStatus
from dr_core.models.ledger import Claim
from dr_core.verify.citation import CITE_RE, decide_citation_status, lookup_citations
from dr_core.verify.provenance import audit_provenance, rescue_unaudited_matched
from dr_core.verify.selection import max_verify_for_depth, select_verification_claim_ids
from dr_core.verify.votes import verify_claim

_VOTE_CONCURRENCY = 3
_EVIDENCE_TIMEOUT = 15.0
_EVIDENCE_MAX_CHARS = 6000
_EVIDENCE_READ_CAP = _EVIDENCE_MAX_CHARS * 4  # bound the read() itself, not just the slice


def _is_terminal(claim: Claim) -> bool:
    return claim.verification.complete or claim.verification.status == VerificationStatus.KILLED_ON_REFUTE


def _fetch_evidence_sync(url: str) -> str | None:
    """Blocking excerpt fetch -- only ever invoked via `asyncio.to_thread`. Any
    failure (network error, timeout, non-decodable body) returns None rather than
    raising; the caller folds that into a "source_unreachable" risk reason, never
    a stall."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "deerflow-dr-verify/1.0"})
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
    active: dict[str, Claim] = {claim_id: claim for claim_id, claim in claims.items() if not _is_terminal(claim)}

    touched: dict[str, dict] = {}

    def _touch(claim: Claim) -> None:
        touched[claim.claim_id] = claim.model_dump(mode="json")

    # 1. Citation gate: deterministic, every active claim, no vote budget consumed.
    courtlistener_token = os.environ.get("COURTLISTENER_API_TOKEN")
    for claim in active.values():
        if claim.citation_status != CitationStatus.UNRESOLVED or not CITE_RE.search(claim.text or ""):
            continue
        cites = await lookup_citations(claim.text, token=courtlistener_token)
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
            record, usage = await verify_claim(claim, source, evidence, reserve_votes=_reserve)
        _merge_usage(usage_totals, usage)
        claim.verification = record
        _touch(claim)

    await asyncio.gather(*(_verify_one(claim_id) for claim_id in selected_ids))

    # 4. Post-verify provenance rescue: a data_ref claim that just landed SUPPORTED
    # this pass with provenance still UNAUDITED advances to MATCHED (dr.js:2693).
    for claim_id in selected_ids:
        claim = active.get(claim_id)
        if claim is None:
            continue
        rescued = rescue_unaudited_matched(claim)
        if rescued is not None:
            claim.data_provenance = rescued
            _touch(claim)

    result: dict = {"dr_run": {"verify_mode": "adaptive_1_3", "verify_usage": usage_totals}}
    if touched:
        result["dr_claims"] = touched
    return result

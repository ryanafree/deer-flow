"""Verification-claim selection (D8), ports dr.js:2483-2519 ``selectVerificationClaimIds``.

Picks which non-terminal claims enter the adaptive vote pass this node run, up to a
depth-keyed ``MAX_VERIFY`` cap: must-cover direct supporters first (degrades
gracefully while ``requirements``/``coverage`` are empty -- S8 has not landed the
requirement/coverage channels yet), then rank-fill by (importance desc, authority_tier
asc via the cited source, claim_id asc). Claims citing a ``source_system ==
"seeded-trap"`` source are ALWAYS included regardless of the cap (the injected-claims
rule, dr.js's ``includeInjected``), and complete/terminal claims are never
re-selected -- though the caller (``dr_core.graph.verify.verify_node``) already
excludes those before calling in, this module re-guards it defensively since it is a
public, independently testable seam.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dr_core.models.enums import VerificationStatus
from dr_core.models.ledger import Claim

MAX_VERIFY: dict[str, int] = {"quick": 5, "standard": 12, "full": 25}
_DEFAULT_DEPTH = "standard"
_WORST_AUTHORITY_TIER = 4  # sort last when a claim's source is missing/unranked


def max_verify_for_depth(depth: str | None) -> int:
    return MAX_VERIFY.get(depth or _DEFAULT_DEPTH, MAX_VERIFY[_DEFAULT_DEPTH])


def _is_terminal(claim: Claim) -> bool:
    return claim.verification.complete or claim.verification.status == VerificationStatus.KILLED_ON_REFUTE


def _authority_tier(claim: Claim, sources: Mapping[str, dict]) -> int:
    source = sources.get(claim.source_id) or {}
    tier = source.get("authority_tier")
    return tier if isinstance(tier, int) else _WORST_AUTHORITY_TIER


def _coverage_values(coverage: Mapping[str, dict] | Sequence[dict] | None) -> list[dict]:
    if not coverage:
        return []
    if isinstance(coverage, Mapping):
        return list(coverage.values())
    return list(coverage)


def select_verification_claim_ids(
    claims: Mapping[str, Claim],
    sources: Mapping[str, dict],
    requirements: Mapping[str, dict] | None,
    coverage: Mapping[str, dict] | Sequence[dict] | None,
    max_verify: int,
) -> list[str]:
    """``claims`` should already be filtered to non-terminal claims by the caller
    (the terminal check here is a defensive re-guard, not the primary filter).
    ``requirements``/``coverage`` are id-keyed payload dicts shaped like the
    ``dr_requirements``/``dr_coverage`` channels (``Requirement``/``CoverageMapping``
    ``model_dump`` payloads); both degrade to empty gracefully."""
    requirements = requirements or {}
    candidates = {claim_id: claim for claim_id, claim in claims.items() if not _is_terminal(claim)}

    selected: list[str] = []
    seen: set[str] = set()

    must_cover_ids = [req_id for req_id, req in requirements.items() if req.get("must_cover")]
    direct_by_req: dict[str, list[str]] = {}
    for mapping in _coverage_values(coverage):
        if mapping.get("relation") == "direct":
            direct_by_req.setdefault(mapping["requirement_id"], []).append(mapping["claim_id"])

    for req_id in must_cover_ids:
        if max_verify and len(selected) >= max_verify:
            break
        for claim_id in direct_by_req.get(req_id, []):
            if claim_id in candidates and claim_id not in seen:
                selected.append(claim_id)
                seen.add(claim_id)
                break

    if max_verify:
        ranked = sorted(candidates.values(), key=lambda c: (-(c.importance or 0), _authority_tier(c, sources), c.claim_id))
        for claim in ranked:
            if len(selected) >= max_verify:
                break
            if claim.claim_id in seen:
                continue
            selected.append(claim.claim_id)
            seen.add(claim.claim_id)

    for claim_id, claim in candidates.items():
        if claim_id in seen:
            continue
        source = sources.get(claim.source_id) or {}
        if source.get("source_system") == "seeded-trap":
            selected.append(claim_id)
            seen.add(claim_id)

    return selected

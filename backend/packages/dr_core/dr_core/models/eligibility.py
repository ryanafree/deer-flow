"""Shared claim-eligibility helpers (S3 contract, REVIEW_FINISH_PLAN_2026-07-06.md;
extended by D9 for real materiality).

Factors out the EXACT derivation `dr_core.graph.gate.eligibility_gate` uses for
publish-readiness (D6 ruling B: `derived_materiality(claim, mappings_for_claim,
requirements_by_id)` + `is_grounded(claim, materiality)` + `conflicts=()`) so the
render node can reconstruct the identical eligible set without re-deriving it ad
hoc. Pure functions only: no I/O, no LLM, no randomness.
`dr_core.graph.gate.eligibility_gate` imports from here too, so there is exactly
one copy of the derivation.

D9: `mappings_for_claim`/`requirements_by_id` default to empty so every caller
that predates coverage mappings (and every existing test) stands unchanged --
`derived_materiality` degrades identically to the pre-D9 behavior (cap at MEDIUM)
when given nothing. Gate and render both now pass the REAL mappings/requirements
so a claim directly satisfying a must-cover requirement derives HIGH materiality
identically wherever eligibility is checked (D6-B: one derivation, two callers).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from dr_core.models.derive import derived_materiality, is_grounded, publication_status
from dr_core.models.enums import PublicationStatus
from dr_core.models.ledger import Claim, CoverageMapping, Requirement


def derive_status(
    claim: Claim,
    mappings_for_claim: Sequence[CoverageMapping] = (),
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> PublicationStatus:
    """The gate's derivation (D6-B / D9): `is_grounded` uses its default
    `post_verify=False`; no conflict channel exists yet, so `conflicts=()`."""
    materiality = derived_materiality(claim, mappings_for_claim, requirements_by_id or {})
    grounded = is_grounded(claim, materiality)
    return publication_status(claim, materiality=materiality, grounded=grounded, conflicts=())


def is_eligible(
    claim: Claim,
    dr_sources: Mapping[str, dict],
    mappings_for_claim: Sequence[CoverageMapping] = (),
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> bool:
    """ELIGIBLE = publication_status(claim) != excluded AND the claim's cited
    source is present in dr_sources (gate.py's `route_after_gate` predicate)."""
    return derive_status(claim, mappings_for_claim, requirements_by_id) != PublicationStatus.EXCLUDED and claim.source_id in dr_sources


def ineligibility_reason(
    claim: Claim,
    dr_sources: Mapping[str, dict],
    mappings_for_claim: Sequence[CoverageMapping] = (),
    requirements_by_id: Mapping[str, Requirement] | None = None,
) -> str | None:
    """None when eligible; otherwise the same mechanical reason string gate.py's
    corrective message reports (m5: a claim that is both EXCLUDED and missing
    its source reports both reasons, never conflated to one)."""
    status = derive_status(claim, mappings_for_claim, requirements_by_id)
    source_ok = claim.source_id in dr_sources
    if status != PublicationStatus.EXCLUDED and source_ok:
        return None
    if source_ok:
        return status.value
    if status == PublicationStatus.EXCLUDED:
        return f"{status.value}+missing_source"
    return "missing_source"

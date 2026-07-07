"""Shared claim-eligibility helpers (S3 contract, REVIEW_FINISH_PLAN_2026-07-06.md).

Factors out the EXACT derivation `dr_core.graph.gate.eligibility_gate` uses for
publish-readiness (D6 ruling B: `derived_materiality(claim, [], {})` +
`is_grounded(claim, materiality)` + `conflicts=()` -- no coverage mappings or
conflict channel exist this sub-phase) so the render node can reconstruct the
identical eligible set without re-deriving it ad hoc. Pure functions only: no
I/O, no LLM, no randomness. `dr_core.graph.gate.eligibility_gate` imports from
here too, so there is exactly one copy of the derivation.
"""

from __future__ import annotations

from collections.abc import Mapping

from dr_core.models.derive import derived_materiality, is_grounded, publication_status
from dr_core.models.enums import PublicationStatus
from dr_core.models.ledger import Claim


def derive_status(claim: Claim) -> PublicationStatus:
    """The gate's derivation (D6-B), verbatim: no coverage mappings/requirements
    exist yet, so `derived_materiality` caps an unmapped claim at MEDIUM exactly
    as it does for a real run before coverage mappings land; `is_grounded` uses
    its default `post_verify=False`; no conflict channel exists this sub-phase."""
    materiality = derived_materiality(claim, [], {})
    grounded = is_grounded(claim, materiality)
    return publication_status(claim, materiality=materiality, grounded=grounded, conflicts=())


def is_eligible(claim: Claim, dr_sources: Mapping[str, dict]) -> bool:
    """ELIGIBLE = publication_status(claim) != excluded AND the claim's cited
    source is present in dr_sources (gate.py's `route_after_gate` predicate)."""
    return derive_status(claim) != PublicationStatus.EXCLUDED and claim.source_id in dr_sources


def ineligibility_reason(claim: Claim, dr_sources: Mapping[str, dict]) -> str | None:
    """None when eligible; otherwise the same mechanical reason string gate.py's
    corrective message reports (m5: a claim that is both EXCLUDED and missing
    its source reports both reasons, never conflated to one)."""
    status = derive_status(claim)
    source_ok = claim.source_id in dr_sources
    if status != PublicationStatus.EXCLUDED and source_ok:
        return None
    if source_ok:
        return status.value
    if status == PublicationStatus.EXCLUDED:
        return f"{status.value}+missing_source"
    return "missing_source"

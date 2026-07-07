"""Pydantic v2 domain models for the DeepResearch claim ledger.

Ported from the harness's runtime JS object shapes (state.js / dr.js) per the D2 ruling
(build-logs/fable-consult-02-RULING.md, section A). Plain data only: materiality,
grounded, requirement live-state, effective_relation, and PublicationStatus are PURE
FUNCTIONS in derive.py, never fields here ("DERIVED ONLY, never a stored field").
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from dr_core.models.enums import (
    CitationStatus,
    ConflictOutcome,
    CoverageRelation,
    DataProvenance,
    GateFlag,
    RequirementKind,
    RequirementState,
    SupportRelation,
    VerificationStatus,
)

if TYPE_CHECKING:
    from dr_core.models.enums import Materiality


class Source(BaseModel):
    id: str
    url_or_id: str
    source_system: str
    title: str | None = None
    authority_tier: int = Field(ge=1, le=4)
    publication_date: str | None = None
    retrieved_at: datetime


class SupportRecord(BaseModel):
    """Quote-path support only. Structured (data_ref) claims never populate this
    (trap e) — they ground via derive.structured_grounded on a separate path."""

    quote: str
    relation_extractor: SupportRelation
    relation_reviewer: SupportRelation | None = None  # None = reviewer unavailable
    inference_note: str | None = None
    qualifier: str | None = None

    def effective_relation(self, materiality: Materiality) -> str:
        """Ports state.js effectiveRelation EXACTLY. Deferred import: derive.py
        imports the ledger models, so this call is resolved at call time to avoid
        a module-level import cycle."""
        from dr_core.models.derive import effective_relation

        return effective_relation(self.relation_extractor, self.relation_reviewer, materiality)


class Vote(BaseModel):
    vote_id: str
    refuted: bool = False
    abstain: bool = False
    confidence: str = "low"
    reasoning: str = ""


class VerificationRecord(BaseModel):
    selected: bool = False
    risk_reasons: list[str] = Field(default_factory=list)
    mode: str | None = None  # "single_clear" | "three_vote" | "incomplete" | None
    votes: list[Vote] = Field(default_factory=list)
    complete: bool = False
    status: VerificationStatus = VerificationStatus.PENDING


class Claim(BaseModel):
    claim_id: str
    text: str
    angle_idx: int | None = None
    importance: int = Field(ge=1, le=5)
    source_id: str
    support: SupportRecord | None = None
    data_ref: dict | None = None
    data_provenance: DataProvenance = DataProvenance.UNAUDITED
    citation_status: CitationStatus = CitationStatus.UNRESOLVED
    gate_flags: list[GateFlag] = Field(default_factory=list)
    verification: VerificationRecord = Field(default_factory=VerificationRecord)
    refutation_notes: str | None = None


class Requirement(BaseModel):
    id: str
    kind: RequirementKind
    text: str
    entities: list[str] = Field(default_factory=list)
    window: dict | None = None  # {"from": str | None, "to": str | None}
    must_cover: bool = False
    attempts: int = 0
    terminal_state: RequirementState | None = None


class CoverageMapping(BaseModel):
    requirement_id: str
    claim_id: str
    relation: CoverageRelation
    elements_satisfied: list[str] = Field(default_factory=list)
    relationship_stated: bool | None = None


class Conflict(BaseModel):
    id: str
    claim_ids: list[str]
    description: str
    outcome: ConflictOutcome | None = None

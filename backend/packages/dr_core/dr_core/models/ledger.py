"""Pydantic v2 domain models for the DeepResearch claim ledger.

Ported from the harness's runtime JS object shapes (state.js / dr.js) per the D2 ruling
(build-logs/fable-consult-02-RULING.md, section A). Plain data only: materiality,
grounded, requirement live-state, effective_relation, and PublicationStatus are PURE
FUNCTIONS in derive.py, never fields here ("DERIVED ONLY, never a stored field").
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

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


class Snapshot(BaseModel):
    """Immutable, append-only record of a structured payload actually
    retrieved for a ``Source`` (D13 -- DECISIONS.md, ratifying Option 1 of
    ``build-logs/fable-options-memo-claim-grounding-snapshots-2026-07-18.md``).

    ``snapshot_id`` is deterministic -- ``compute_snapshot_id(source_id,
    tool_call_id, normalized_content)`` -- so the SAME sighting re-processed
    (e.g. a replayed graph step) merges idempotently; a DIFFERENT payload
    minted under an id already seen is a data-integrity error and RAISES in
    ``graph/state.py``'s merge rule (D13's "divergent content under the same
    ID raises"), never silently overwritten.

    ``data_ref`` is the structured connector tool's own payload fragment
    (WRDS/EDGAR/FRED ``tools.py``'s ``data_ref``, verbatim) -- the ground
    truth ``record_claim``'s model-supplied ``data_ref`` argument is checked
    against (``graph/claim_tool.py``), so a claim's structured grounding is
    verified against a value the model cannot alter after the fact, closing
    the memo's secondary boundary 1 (structured connector JSON retained as a
    snapshot; new ``record_claim`` calls must quote-match it).
    """

    snapshot_id: str
    source_id: str
    tool_call_id: str
    data_ref: dict | None = None
    retrieved_at: datetime


def compute_snapshot_id(source_id: str, tool_call_id: str, normalized_content: str) -> str:
    """Deterministic snapshot id (D13): ``hash(source_id, tool_call_id,
    normalized_content)``. ``normalized_content`` is whatever the snapshot's
    own normalization produces -- for a structured snapshot,
    ``graph/middleware.py`` passes ``json.dumps(data_ref, sort_keys=True,
    default=str)``."""
    return hashlib.sha256(f"{source_id}|{tool_call_id}|{normalized_content}".encode()).hexdigest()[:16]


class Source(BaseModel):
    id: str
    url_or_id: str
    source_system: str
    title: str | None = None
    authority_tier: int = Field(ge=1, le=4)
    publication_date: str | None = None
    retrieved_at: datetime
    snapshots: list[Snapshot] = Field(default_factory=list)


class SupportRecord(BaseModel):
    """Quote-path support only. Structured (data_ref) claims never populate this
    (trap e) — they ground via derive.structured_grounded on a separate path."""

    quote: str
    relation_extractor: SupportRelation
    relation_reviewer: SupportRelation | None = None  # None = reviewer unavailable
    reviewer_note: str | None = None
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
    target_requirement_ids: list[str] = Field(default_factory=list)
    support: SupportRecord | None = None
    data_ref: dict | None = None
    data_provenance: DataProvenance = DataProvenance.UNAUDITED
    citation_status: CitationStatus = CitationStatus.UNRESOLVED
    gate_flags: list[GateFlag] = Field(default_factory=list)
    verification: VerificationRecord = Field(default_factory=VerificationRecord)
    refutation_notes: str | None = None


#  SPEC_evidence_routing_2026-07-10.md, Interfaces: field NAMES and value sets are
#  FROZEN. Plain str (not StrEnum) because invalid/missing values must COERCE to
#  "any" (settled question 12 / test 5), which StrEnum's raise-on-invalid validation
#  does not support.
VALID_EVIDENCE_CLASSES = frozenset({"academic", "primary_data", "practitioner", "news", "any"})
VALID_FRESHNESS_VALUES = frozenset({"foundational", "frontier", "any"})


class Requirement(BaseModel):
    id: str
    kind: RequirementKind
    text: str
    entities: list[str] = Field(default_factory=list)
    window: dict | None = None  # {"from": str | None, "to": str | None}
    must_cover: bool = False
    attempts: int = 0
    terminal_state: RequirementState | None = None
    evidence_class: str = "any"
    freshness: str = "any"

    @field_validator("evidence_class", mode="before")
    @classmethod
    def _coerce_evidence_class(cls, v: object) -> str:
        return v if v in VALID_EVIDENCE_CLASSES else "any"

    @field_validator("freshness", mode="before")
    @classmethod
    def _coerce_freshness(cls, v: object) -> str:
        return v if v in VALID_FRESHNESS_VALUES else "any"


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

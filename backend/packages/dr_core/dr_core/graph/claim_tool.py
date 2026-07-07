"""record_claim -- the C1 claim seam (D6 ruling A).

A model-facing tool the ASSERTING MODEL calls to place a claim in the
``dr_claims`` channel so ``eligibility_gate`` has something to gate. The
handler is DETERMINISTIC (no LLM, no network): the model supplies content
fields only (text/source_id/quote/importance/gate_flags); verification,
citation_status, and data_provenance are minted here at their INITIAL model
values -- those ladders belong to later verify passes, never the asserting
model (D6 ruling A.4).

Four binding refinements (D6 ruling A):
1. rides ``DrLedgerMiddleware.tools`` -- zero FORK_DELTA (factory.py:876 /
   types.py:395 collect middleware ``.tools``).
2. deterministic id = ``sha256(norm(text) + source_id)[:16]`` -- re-asserting
   the same (text, source_id) pair is idempotent.
3. duplicate-divergent GUARD: an existing id with a DIFFERENT payload is
   confirmed back to the model and NOT written -- first assertion wins, so a
   model quirk (e.g. importance 3 then 4) never reaches ``merge_ledger`` as a
   same-field divergent write (which raises and kills the run).
4. rejections are actionable (name the bad source_id / gate_flag).

Per D7, a successful record also sets ``dr_run["deliverable"]=True`` on the
returned Command -- a recorded claim is definitionally a research
deliverable, so the turn is routed to ``eligibility_gate``.
"""

from __future__ import annotations

import hashlib
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import ValidationError

from dr_core.models import Claim, GateFlag, SupportRecord, SupportRelation, norm


def _claim_id(text: str, source_id: str) -> str:
    """Mirrors middleware.py's ``_source_id`` pattern: a short, deterministic id
    from a normalized content hash, so re-asserting the same (text, source_id)
    pair always resolves to the same claim."""
    return hashlib.sha256((norm(text) + source_id).encode("utf-8")).hexdigest()[:16]


def _source_id_for_url(url: str) -> str:
    """The C2 hook's deterministic source id for a url (middleware._source_id);
    duplicated one-liner rather than importing the middleware module (which
    pulls in the langgraph runtime) into this tool module."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _normalize_gate_flags(gate_flags: list[str] | None) -> list[str] | None:
    """Validates model-supplied flags against ``GateFlag``, normalizing
    hyphens to underscores like ``eval/score_run.py::_normalize_flag`` (fixture
    flags are hyphenated harness spelling; ``GateFlag`` values are underscored).
    Returns ``None`` on the first unrecognized flag so the caller rejects the
    whole call instead of silently dropping one."""
    if not gate_flags:
        return []
    normalized: list[str] = []
    for flag in gate_flags:
        try:
            normalized.append(GateFlag(flag.replace("-", "_")).value)
        except ValueError:
            return None
    return normalized


def _reject(tool_call_id: str, content: str) -> Command:
    """No dr_claims write -- only a confirming/rejecting ToolMessage."""
    return Command(update={"messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]})


@tool
def record_claim(
    text: str,
    source_id: str,
    quote: str,
    importance: int,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    gate_flags: list[str] | None = None,
) -> Command:
    """Record a factual claim asserted from a previously retrieved source.

    Call this once you have found a claim-worthy fact via web_search/web_fetch.
    Cite the source_id from that prior tool result and quote the exact
    supporting span. This call is content-only -- verification and citation
    status are decided by later verify passes, not by you.

    Args:
        text: The claim being asserted, in your own words.
        source_id: The exact url of a source already returned by
            web_search/web_fetch (its ledger id is also accepted).
        quote: The exact supporting span from that source.
        importance: 1 (minor) to 5 (central to the answer).
        gate_flags: Optional risk flags, e.g. "vendor-reported", "period-mismatch".
    """
    dr_sources = state.get("dr_sources") or {}
    if source_id not in dr_sources:
        # The model only ever SEES urls (search/fetch results never carry the
        # sha256-minted ledger ids), so resolve a url citation to its
        # deterministic id -- same hash the C2 hook minted it under. Caught by
        # the S6 live E2E run: every record_claim was rejected because the
        # model had no way to know a ledger id. Resolution is exact-match
        # deterministic; referential integrity is unchanged.
        url_resolved = _source_id_for_url(source_id)
        if url_resolved not in dr_sources:
            return _reject(tool_call_id, f"source {source_id!r} not found -- cite the exact url (or id) of a prior web_search/web_fetch result.")
        source_id = url_resolved

    normalized_flags = _normalize_gate_flags(gate_flags)
    if normalized_flags is None:
        known = ", ".join(f.value for f in GateFlag)
        return _reject(tool_call_id, f"unknown gate_flags in {gate_flags!r} -- use one of: {known}.")

    try:
        claim = Claim(
            claim_id=_claim_id(text, source_id),
            text=text,
            importance=importance,
            source_id=source_id,
            support=SupportRecord(quote=quote, relation_extractor=SupportRelation.SUPPORTS_DIRECTLY),
            gate_flags=normalized_flags,
        )
    except ValidationError as exc:
        return _reject(tool_call_id, f"invalid claim: {exc}")

    payload = claim.model_dump(mode="json")
    existing = (state.get("dr_claims") or {}).get(claim.claim_id)
    if existing is not None:
        if existing == payload:
            return _reject(tool_call_id, f"already recorded as {claim.claim_id}")
        # First assertion wins: confirm the existing id, write nothing. Never
        # raise or overwrite -- that guard belongs to merge_ledger, and a
        # divergent pair reaching it there would kill the run.
        return _reject(tool_call_id, f"a claim from this text and source is already recorded as {claim.claim_id} -- first assertion kept.")

    message = ToolMessage(content=f"recorded claim {claim.claim_id}", tool_call_id=tool_call_id)
    # D7: a recorded claim is definitionally a deliverable -- mark the turn
    # so route_after_research gates it.
    return Command(update={"dr_claims": {claim.claim_id: payload}, "dr_run": {"deliverable": True}, "messages": [message]})

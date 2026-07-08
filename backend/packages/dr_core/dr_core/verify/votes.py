"""Adaptive 1-3 vote protocol (D8), ports dr.js:2536-2648 exactly.

``risk_reasons`` and ``cast_vote`` are the small, independently testable primitives;
``verify_claim`` is the per-claim orchestration (vote 1, ``is_single_clear`` fast
path, else escalate to two more votes, aggregate) that ``dr_core.graph.verify``'s
node calls once per selected claim. The vote budget is NOT owned here -- it is
injected via ``reserve_votes`` (an async callable returning whether N more vote
calls may be cast) so this module stays testable without a real shared,
concurrency-safe counter, and so the node can enforce ``MAX_VOTE_CALLS =
3*MAX_VERIFY`` across every claim being verified this pass, not just one.

The voter model has NO tool access and is intentionally adversarial (D8: "default
toward refuted=true when the support is weak") -- it is a text-in/JSON-out check,
not a research agent. ``_get_vote_model`` is a module-level factory so tests can
monkeypatch it to a stub without touching ``deerflow.models``.

D10 recalibration: the S10 live acceptance found the ported dr.js bias assumed a
voter with TOOLS that could independently check a source; a text-only voter was
instead scoring "I could not fetch this" as "this is false", producing a 100%
kill rate. The system prompt now branches on evidence availability -- the
adversarial refute-on-weak-support bias survives ONLY when a fetched excerpt is
present; when it is absent, the voter is told explicitly that unavailability is
not falsity and judges claim-vs-quote consistency only (see DECISIONS.md D10).

D10 addendum: the prompt-only fix was insufficient -- a cheap voter (or-mid) kept
refuting plainly-true claims at high confidence in the evidence-absent regime
(an instruction-following gap, not a code bug). ``cast_vote`` now enforces a
DETERMINISTIC clamp in that regime only: the vote JSON schema gains a
``contradiction`` field the voter must fill with a verbatim claim-vs-quote
conflict when refuting; a parsed ``refuted=true`` with an empty/missing
``contradiction`` is coerced to ``abstain=True`` in code, before aggregation,
never by asking the model to reconsider. ``contradiction`` is not a ``Vote``
model field -- it is consumed at parse time and folded into ``reasoning`` for
auditability. Evidence-present votes are unaffected.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from dr_core.models.derive import aggregate_verification_status, is_single_clear
from dr_core.models.enums import DataProvenance, VerificationStatus
from dr_core.models.ledger import Claim, VerificationRecord, Vote

_BASE_INSTRUCTIONS = (
    "You are an adversarial fact-checking voter for a research-assurance pipeline. Your job is to "
    "find reasons a claim's support does NOT hold up, not to confirm it. refuted=true requires "
    "AFFIRMATIVE grounds: the evidence in front of you contradicts the claim, the quote plainly fails "
    "to support the claim as stated (a plain overstatement), or there are internal fabrication signals. "
    "Never mark refuted=true merely because the support is thin, indirect, or unconfirmed -- that is "
    "abstain territory, not refutation. Respond with strict JSON only, no prose, no markdown fences."
)

# D10: the refute-on-weak-support bias applies only when there is real evidence to weigh.
_EVIDENCE_PRESENT_INSTRUCTIONS = (
    "A fetched source excerpt is provided below. Weigh it adversarially: default toward refuted=true "
    "when the support is weak, indirect, contradicted, or the source is not authoritative -- do not "
    "give the claim the benefit of the doubt when real evidence is in front of you."
)

# D10: an unreachable source is NOT evidence of falsity. The voter judges only whether the
# claim text is consistent with the quote it was given for the claim's own support.
_EVIDENCE_ABSENT_INSTRUCTIONS = (
    "The source could not be retrieved; do not treat unavailability as falsity. Judge ONLY whether the "
    "claim's own supporting quote is consistent with the claim text: if the quote plainly does NOT "
    "support the claim (an overstatement or non sequitur), refute it. If the quote supports the claim "
    "but you have no way to independently verify the underlying source, abstain -- never refute solely "
    "because the source was unreachable."
)


def _vote_system_instructions(evidence_available: bool) -> str:
    regime = _EVIDENCE_PRESENT_INSTRUCTIONS if evidence_available else _EVIDENCE_ABSENT_INSTRUCTIONS
    return f"{_BASE_INSTRUCTIONS} {regime}"


_VOTE_JSON_SHAPE = '{"refuted": <bool>, "abstain": <bool>, "confidence": "<low|medium|high>", "reasoning": "<short string>"}'

# D10 addendum: evidence-absent regime only. `contradiction` is not a Vote model field
# (see module docstring) -- it exists purely so `cast_vote` can deterministically clamp
# an unsupported refute before it ever reaches aggregation.
_VOTE_JSON_SHAPE_EVIDENCE_ABSENT = '{"refuted": <bool>, "abstain": <bool>, "confidence": "<low|medium|high>", "reasoning": "<short string>", "contradiction": "<verbatim quote of the conflict, or empty string>"}'
_CONTRADICTION_INSTRUCTION = (
    '\n\nIf refuted=true, "contradiction" MUST quote, verbatim, the exact words from the supporting '
    "quote above that conflict with or plainly fail to support the claim text. If you are not "
    'refuting, leave "contradiction" as an empty string.'
)


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}


def _accumulate_usage(usage: dict[str, int], response) -> None:
    if response is None:
        return
    meta = getattr(response, "usage_metadata", None) or {}
    usage["input_tokens"] += int(meta.get("input_tokens") or 0)
    usage["output_tokens"] += int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    usage["cache_read_tokens"] += int(details.get("cache_read") or 0)
    usage["cache_creation_tokens"] += int(details.get("cache_creation") or 0)
    usage["calls"] += 1


def risk_reasons(claim: Claim, source: dict | None, *, sole_must_cover_supporter: bool = False) -> list[str]:
    """Deterministic per-claim risk signals (ports dr.js:2536-2544
    ``verificationRiskReasons``): seeded/injected source, any gate flag,
    authority_tier>=4 (or unranked), a structured claim whose provenance is
    not yet MATCHED, and (D9, activating the S8 requirement/coverage
    channels) whether this claim is the SOLE direct supporter of an active
    must-cover requirement -- losing it on refutation would uncover a
    mandatory requirement, so it forces full 3-vote scrutiny instead of the
    single-vote fast path. ``sole_must_cover_supporter`` is a plain bool the
    caller derives from state (``dr_core.graph.verify``, via
    ``dr_core.models.derive.reopens_requirement``) -- this function stays
    pure.

    D10: the trigger was originally >=3 ("tier-3-or-worse"), but the fork's
    only source right now is the C2 web-fetch hook, which mints every source
    at the default authority_tier=3 -- the old threshold fired on every
    single claim, making the single-clear fast path unreachable. Recalibrated
    to >=4 ("tier-4-or-worse") until S9's connector registry maps real domains
    to a real tier taxonomy."""
    reasons: list[str] = []
    if source and source.get("source_system") == "seeded-trap":
        reasons.append("injected")
    for flag in claim.gate_flags:
        value = flag.value if hasattr(flag, "value") else str(flag)
        reasons.append(f"gate:{value}")
    tier = (source or {}).get("authority_tier")
    if not isinstance(tier, int) or tier >= 4:
        reasons.append("tier-4-or-worse")
    if claim.data_ref and claim.data_provenance != DataProvenance.MATCHED:
        reasons.append("structured-unmatched")
    if sole_must_cover_supporter:
        reasons.append("sole-must-cover-supporter")
    return reasons


def _get_vote_model():
    """Module-level injectable factory -- tests monkeypatch this name directly
    rather than reaching into ``deerflow.models``."""
    from deerflow.models import create_chat_model

    return create_chat_model(os.environ.get("DR_VERIFY_MODEL", "or-sonnet"))


def _format_evidence(claim: Claim, source: dict | None, evidence_excerpt: str | None) -> str:
    lines = [f"Claim: {claim.text}"]
    if claim.support is not None:
        lines.append(f"Supporting quote: {claim.support.quote}")
    if claim.data_ref:
        lines.append(f"Structured data reference: {json.dumps(claim.data_ref, sort_keys=True, default=str)}")
    if source:
        lines.append(f"Source: {source.get('title') or '(untitled)'} <{source.get('url_or_id', '')}>")
    lines.append(f"Fetched source excerpt:\n{evidence_excerpt}" if evidence_excerpt else "Fetched source excerpt: (unavailable -- source could not be retrieved)")
    return "\n".join(lines)


def _build_vote_prompt(claim: Claim, source: dict | None, evidence_excerpt: str | None, risk: list[str]) -> str:
    risk_text = ", ".join(risk) if risk else "none"
    evidence_available = evidence_excerpt is not None
    shape = _VOTE_JSON_SHAPE if evidence_available else _VOTE_JSON_SHAPE_EVIDENCE_ABSENT
    contradiction_note = "" if evidence_available else _CONTRADICTION_INSTRUCTION
    return (
        f"{_format_evidence(claim, source, evidence_excerpt)}\n\n"
        f"Risk signals already flagged for this claim: {risk_text}.\n\n"
        "Decide whether the evidence actually supports the claim as stated. Respond with ONLY a JSON "
        f"object matching exactly this shape: {shape}{contradiction_note}"
    )


def _parse_vote_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
    except (TypeError, ValueError):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed, dict) or "refuted" not in parsed or "abstain" not in parsed:
        return None
    return parsed


async def cast_vote(claim: Claim, source: dict | None, evidence_excerpt: str | None, n: int, risk: list[str]) -> tuple[Vote, object | None]:
    """Cast vote ``n`` for ``claim``. Any model-call failure or unparseable
    output degrades to an ABSTAIN vote (fails toward not_verified, never toward
    supported or killed) -- this function never raises. Returns ``(vote,
    raw_response_or_None)`` so the caller can accumulate token usage without this
    module owning that bookkeeping."""
    vote_id = f"{claim.claim_id}:v{n}"
    try:
        model = _get_vote_model()
        response = await model.ainvoke(
            [
                SystemMessage(content=_vote_system_instructions(evidence_excerpt is not None)),
                HumanMessage(content=_build_vote_prompt(claim, source, evidence_excerpt, risk)),
            ]
        )
    except Exception as exc:
        return Vote(vote_id=vote_id, refuted=False, abstain=True, confidence="low", reasoning=f"vote call failed: {exc}"), None

    content = response.content if isinstance(response.content, str) else str(response.content)
    parsed = _parse_vote_json(content)
    if parsed is None:
        return Vote(vote_id=vote_id, refuted=False, abstain=True, confidence="low", reasoning="unparseable vote output"), response

    refuted = bool(parsed.get("refuted", False))
    abstain = bool(parsed.get("abstain", False))
    reasoning = str(parsed.get("reasoning", ""))

    # D10 addendum: in the evidence-absent regime, a refute with no quoted
    # contradiction is code-coerced to abstain BEFORE aggregation -- a deterministic
    # clamp, never a second prose judgment call. Evidence-present votes are untouched
    # (no `contradiction` field is requested or checked there).
    if evidence_excerpt is None and refuted:
        contradiction = str(parsed.get("contradiction") or "").strip()
        if not contradiction:
            refuted = False
            abstain = True
            reasoning = f"{reasoning} [D10 clamp: refuted=true with no quoted contradiction in the evidence-absent regime -> coerced to abstain]".strip()

    return (
        Vote(
            vote_id=vote_id,
            refuted=refuted,
            abstain=abstain,
            confidence=str(parsed.get("confidence", "low")),
            reasoning=reasoning[:2000],
        ),
        response,
    )


async def verify_claim(
    claim: Claim,
    source: dict | None,
    evidence_excerpt: str | None,
    *,
    reserve_votes: Callable[[int], Awaitable[bool]],
    sole_must_cover_supporter: bool = False,
) -> tuple[VerificationRecord, dict[str, int]]:
    """The adaptive 1-3 vote protocol for one claim (D8, ports dr.js:2536-2648):
    vote 1 always attempted, subject to budget; ``is_single_clear`` fast path ->
    SUPPORTED/single_clear/complete; else escalate to two more votes (also
    subject to budget) and aggregate via ``derive.aggregate_verification_status``
    -> three_vote/complete. Either budget check failing -> NOT_VERIFIED/
    incomplete/``complete=False`` (never SUPPORTED on a partial vote set).
    ``reserve_votes(n)`` reserves N vote calls against the caller's shared
    budget; a fetch failure for this claim's source should already be folded
    into ``evidence_excerpt is None`` (adds the ``source_unreachable`` risk
    reason) by the caller before calling this. ``sole_must_cover_supporter``
    (D9) is threaded straight into ``risk_reasons`` -- see its docstring.
    """
    risk = risk_reasons(claim, source, sole_must_cover_supporter=sole_must_cover_supporter)
    if source is not None and evidence_excerpt is None and source.get("url_or_id"):
        risk = [*risk, "source_unreachable"]

    usage = _empty_usage()

    if not await reserve_votes(1):
        return VerificationRecord(selected=True, risk_reasons=risk, mode="incomplete", votes=[], complete=False, status=VerificationStatus.NOT_VERIFIED), usage

    vote1, resp1 = await cast_vote(claim, source, evidence_excerpt, 1, risk)
    _accumulate_usage(usage, resp1)

    if is_single_clear(vote1, risk):
        return VerificationRecord(selected=True, risk_reasons=risk, mode="single_clear", votes=[vote1], complete=True, status=VerificationStatus.SUPPORTED), usage

    if not await reserve_votes(2):
        return VerificationRecord(selected=True, risk_reasons=risk, mode="incomplete", votes=[vote1], complete=False, status=VerificationStatus.NOT_VERIFIED), usage

    vote2, resp2 = await cast_vote(claim, source, evidence_excerpt, 2, risk)
    vote3, resp3 = await cast_vote(claim, source, evidence_excerpt, 3, risk)
    _accumulate_usage(usage, resp2)
    _accumulate_usage(usage, resp3)

    all_votes = [vote1, vote2, vote3]
    status = aggregate_verification_status(all_votes)
    return VerificationRecord(selected=True, risk_reasons=risk, mode="three_vote", votes=all_votes, complete=True, status=status), usage

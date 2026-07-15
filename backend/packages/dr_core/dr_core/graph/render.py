"""Render node -- real implementation (D6 ruling D; REVIEW_FINISH_PLAN_2026-07-06.md
Part II S3 / Part III contract sheet).

Deterministic: reconstructs the eligible claim set from `state["dr_claims"]` via the
shared eligibility helper (`dr_core.models.eligibility`, drawing the SAME derivation the
gate freezes into `dr_run["citation_ordinals"]`), generates the report body
(`dr_core.render.body`), writes the run folder (`dr_core.run.write_run`) and the HTML
render (`dr_core.render.render_report`), and writes `run_dir` back into `dr_run`. No LLM
calls, no network, no randomness -- everything here is a pure function of state plus
filesystem I/O for the run folder itself.

The load-bearing invariant (Part III): claims.jsonl must contain EXACTLY the eligible
claims, in `citation_ordinals` order, with `[n]` markers using those same numbers.
`citation_ordinals` is the gate's frozen map (1-based, eligible set only, compacted) --
this node reads it rather than recomputing eligibility order itself, so the two can never
drift out of sync.
"""

from __future__ import annotations

import os

from dr_core.accounting import build_accounting
from dr_core.models.eligibility import ineligibility_reason
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, Source
from dr_core.profiles import ProfileError, load_profile
from dr_core.render.body import generate_body
from dr_core.render.render_report import render as render_html
from dr_core.run.write_run import write_run

_RUNS_DIR_ENV = "DR_RUNS_DIR"
_DEFAULT_RUNS_DIR = os.path.expanduser("~/Documents/Projects/deerflow-dr/runs")


def _resolve_runs_dir(dr_run: dict) -> str:
    return dr_run.get("runs_dir") or os.environ.get(_RUNS_DIR_ENV) or _DEFAULT_RUNS_DIR


def render_node(state) -> dict:
    """Render the frozen, gated ledger into a run folder + HTML report.

    Reads `dr_claims`/`dr_sources` (the ledger) and `dr_run` (citation_ordinals,
    stop_reason, question, profile) from state; writes nothing back onto the
    ledger channels -- only `dr_run.run_dir`/`render_completed` are returned.
    """
    dr_claims = state.get("dr_claims") or {}
    dr_sources = state.get("dr_sources") or {}
    dr_requirements = state.get("dr_requirements") or {}
    dr_coverage = state.get("dr_coverage") or {}
    dr_run = dict(state.get("dr_run") or {})

    # D9: gate and render derive materiality/status identically (D6-B) -- both
    # get the SAME real mappings/requirements, never re-derived ad hoc.
    requirements_by_id: dict[str, Requirement] = {req_id: Requirement.model_validate(payload) for req_id, payload in dr_requirements.items()}
    coverage_mappings: list[CoverageMapping] = [CoverageMapping.model_validate(payload) for payload in dr_coverage.values()]
    mappings_by_claim: dict[str, list[CoverageMapping]] = {}
    for mapping in coverage_mappings:
        mappings_by_claim.setdefault(mapping.claim_id, []).append(mapping)

    citation_ordinals: dict[str, int] = dr_run.get("citation_ordinals") or {}

    claims_by_ordinal: dict[int, Claim] = {}
    ineligible: list[tuple[str, Claim, str]] = []
    all_claims: list[Claim] = []
    # D11 run scoping: prior-run claims (present at initialize's baseline
    # snapshot) never render this run -- same filter the gate applied when it
    # froze citation_ordinals, so gate and render still agree claim-for-claim.
    baseline_claim_ids = set(dr_run.get("baseline_claim_ids") or [])
    for claim_id, payload in dr_claims.items():
        if claim_id in baseline_claim_ids:
            continue
        claim = Claim.model_validate(payload)
        all_claims.append(claim)
        ordinal = citation_ordinals.get(claim_id)
        if ordinal is not None:
            claims_by_ordinal[ordinal] = claim
        else:
            # Defensive fallback only: ineligibility_reason mirrors the gate's own
            # per-claim derivation exactly, so a None here would mean gate and
            # render have drifted out of sync (should never happen -- D6-B ties
            # both to the same fixed formula).
            reason = ineligibility_reason(claim, dr_sources, mappings_by_claim.get(claim_id, ()), requirements_by_id) or "excluded"
            ineligible.append((claim_id, claim, reason))

    sources_by_id: dict[str, Source] = {source_id: Source.model_validate(payload) for source_id, payload in dr_sources.items()}

    body = generate_body(
        claims_by_ordinal,
        sources_by_id,
        dr_run,
        ineligible=ineligible,
        mappings_by_claim=mappings_by_claim,
        requirements_by_id=requirements_by_id,
    )
    eligible_claims = [claims_by_ordinal[n] for n in sorted(claims_by_ordinal)]

    runs_dir = _resolve_runs_dir(dr_run)
    os.makedirs(runs_dir, exist_ok=True)

    question = dr_run.get("question") or "Untitled research question"
    profile = dr_run.get("profile", "general")

    # Profile lookup is deterministic (a static YAML keyed by dr_run's own profile
    # string) so accounting stays a pure function of state, per this node's
    # contract. An unrecognized/invalid profile degrades to unpriced accounting
    # rather than failing the render.
    try:
        model_tiers = load_profile(profile).get("model_tiers")
    except (ValueError, ProfileError):
        model_tiers = None

    accounting = build_accounting(state.get("messages"), dr_run, profile_tiers=model_tiers)

    manifest_in = {
        "loop": {
            "conflicts_open": 0,
            "requirements_covered": dr_run.get("requirements_covered", 0),
            "requirements_must_cover": dr_run.get("requirements_must_cover", 0),
        },
        "verify_mode": dr_run.get("verify_mode", "off"),
        "depth": dr_run.get("depth", "quick"),
        "profile": profile,
        "accounting": accounting,
    }
    if dr_run.get("stop_reason"):
        manifest_in["stop_reason"] = dr_run["stop_reason"]

    run_dir = write_run(
        report_body=body,
        sources=list(sources_by_id.values()),
        claims=eligible_claims,
        requirements=list(requirements_by_id.values()),
        coverage=coverage_mappings,
        # D10: the full ledger (every claim, any eligibility state) for ledger.jsonl --
        # claims= above stays eligible-only/ordinal-ordered (Part III invariant, unchanged).
        ledger_claims=all_claims,
        profile=profile,
        question=question,
        runs_dir=runs_dir,
        manifest_in=manifest_in,
    )
    html_out = render_html(run_dir)
    with open(os.path.join(run_dir, "report.html"), "w") as f:
        f.write(html_out)

    return {"dr_run": {"run_dir": run_dir, "render_completed": True}}

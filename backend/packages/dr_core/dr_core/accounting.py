"""accounting.py — token / dollar accounting for one dr_core run (S9).

Pure functions only: no I/O, no state mutation, no network. Sums the four
LangChain ``usage_metadata`` metrics (input, output, cache_read, cache_write)
over a phase's messages, folds in the optional verify-phase usage dict the
verify node writes onto ``dr_run["verify_usage"]`` (S7; absent today, treated
as all-zero), and prices each phase against a static $/1M-token table keyed by
model id (D1: OpenRouter drives the cheap/mid gruntwork and verify tiers;
``claude-top`` runs on the Claude subscription CLI, so its marginal API cost is
$0 -- see DECISIONS.md D1).

D10 companion ruling: a third "plan" phase bucket is folded in the same
defensive way as "verify", off ``dr_run["plan_usage"]`` (the D9 plan_coverage
node's usage; absent today on runs predating it, treated as all-zero).
"""

from __future__ import annotations

METRICS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")

# D1: profile model_tiers use abstract tier names; this maps them to the concrete
# model id the pricing table is keyed by. A tier name not in this map is passed
# through as-is, so a profile that names a concrete model id directly still prices.
MODEL_TIER_IDS = {
    "or-cheap": "openai/gpt-oss-20b",
    "or-mid": "openai/gpt-4o-mini",
    "or-sonnet": "anthropic/claude-sonnet-5",
    "claude-top": "claude-top",
    # q.5 (2026-07-10): DR_VERIFY_MODEL default, subscription `claude -p` shim.
    # Reuses the claude-top pricing row (marginal API cost $0) -- or-sonnet
    # stays mapped above for the explicit OpenRouter fallback.
    "claude-verify": "claude-top",
}

# $/1M tokens. claude-top is the subscription `claude -p` shim: no marginal API
# cost, priced at 0 per D1's cost-bound rationale (kept in the table, not just
# hardcoded, so an unknown model is still distinguishable from a known-free one).
PRICING = {
    "openai/gpt-oss-20b": {"input": 0.03, "output": 0.14},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-top": {"input": 0.0, "output": 0.0},
}


def _empty_usage() -> dict[str, int]:
    return dict.fromkeys(METRICS, 0)


def _extract_usage(message) -> dict[str, int]:
    """Pull the four metrics off one message (AIMessage instance or a dict with a
    'usage_metadata' key). A message with no usage_metadata contributes all-zero;
    never raises."""
    if isinstance(message, dict):
        usage = message.get("usage_metadata")
    else:
        usage = getattr(message, "usage_metadata", None)
    usage = usage or {}
    input_details = usage.get("input_token_details") or {}
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read_tokens": input_details.get("cache_read") or 0,
        "cache_write_tokens": input_details.get("cache_creation") or 0,
    }


def sum_usage(messages) -> dict[str, int]:
    """Sum the four metrics over a list of messages. Missing/None input -> all-zero."""
    totals = _empty_usage()
    for message in messages or []:
        usage = _extract_usage(message)
        for key in METRICS:
            totals[key] += usage[key]
    return totals


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {key: a.get(key, 0) + b.get(key, 0) for key in METRICS}


def _resolve_model_id(tier_or_model: str | None) -> str | None:
    if not tier_or_model:
        return None
    return MODEL_TIER_IDS.get(tier_or_model, tier_or_model)


def price_phase(usage: dict[str, int], model: str | None) -> float | None:
    """Dollar cost for one phase's usage under `model`'s pricing row.

    `model` may be a raw model id or a MODEL_TIER_IDS tier name. Unknown/missing
    model -> None (never a crash; render as "unknown" upstream)."""
    resolved = _resolve_model_id(model)
    row = PRICING.get(resolved) if resolved else None
    if row is None:
        return None
    input_tokens = usage.get("input_tokens", 0) + usage.get("cache_write_tokens", 0)
    cache_read_tokens = usage.get("cache_read_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = (input_tokens + cache_read_tokens) / 1_000_000 * row["input"] + output_tokens / 1_000_000 * row["output"]
    return round(cost, 6)


def build_accounting(messages, dr_run: dict | None = None, *, profile_tiers: dict | None = None) -> dict:
    """Build the manifest['accounting'] block.

    `messages` is the research phase's message list (typically `state["messages"]`).
    `dr_run` is the dr_run channel dict; `dr_run["verify_usage"]` and
    `dr_run["plan_usage"]`, if present, are optional {"input_tokens":...,
    "output_tokens":..., ...} dicts the verify node (S7) and plan_coverage node (D9)
    write respectively -- absent today on older runs, folded in as all-zero when
    missing.
    `profile_tiers` is a profile's `model_tiers` dict ({"gruntwork":..., "verify":...,
    "top":...}); when given, prices the research phase against `gruntwork` and the
    verify phase against `verify`. The plan phase has no dedicated tier key yet, so it
    prices against `gruntwork` too (plan_coverage is a structured gruntwork-tier call,
    same model class as research). Omitted -> phases price as unknown (None).
    """
    dr_run = dr_run or {}
    profile_tiers = profile_tiers or {}

    research_usage = sum_usage(messages)
    verify_usage_raw = dr_run.get("verify_usage") or {}
    verify_usage = {key: verify_usage_raw.get(key, 0) for key in METRICS}
    plan_usage_raw = dr_run.get("plan_usage") or {}
    plan_usage = {key: plan_usage_raw.get(key, 0) for key in METRICS}

    phases = {"research": research_usage, "verify": verify_usage, "plan": plan_usage}

    totals = _empty_usage()
    for usage in phases.values():
        totals = _add(totals, usage)

    research_cost = price_phase(research_usage, profile_tiers.get("gruntwork"))
    verify_cost = price_phase(verify_usage, profile_tiers.get("verify"))
    plan_cost = price_phase(plan_usage, profile_tiers.get("gruntwork"))

    known_costs = [c for c in (research_cost, verify_cost, plan_cost) if c is not None]
    dollar_cost = round(sum(known_costs), 6) if known_costs else None

    return {
        "phases": phases,
        "totals": totals,
        "dollar_cost": dollar_cost,
    }

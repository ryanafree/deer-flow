"""Tests for dr_core.accounting (S9)."""

from dr_core.accounting import build_accounting, price_phase, sum_usage
from langchain_core.messages import AIMessage, HumanMessage


def _ai_message(input_tokens, output_tokens, cache_read=0, cache_creation=0):
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    if cache_read or cache_creation:
        usage["input_token_details"] = {"cache_read": cache_read, "cache_creation": cache_creation}
    return AIMessage(content="x", usage_metadata=usage)


class TestSumUsage:
    def test_sums_across_messages(self):
        messages = [_ai_message(10, 5), _ai_message(20, 8)]
        totals = sum_usage(messages)
        assert totals == {"input_tokens": 30, "output_tokens": 13, "cache_read_tokens": 0, "cache_write_tokens": 0}

    def test_includes_cache_metrics(self):
        messages = [_ai_message(10, 5, cache_read=100, cache_creation=50)]
        totals = sum_usage(messages)
        assert totals["cache_read_tokens"] == 100
        assert totals["cache_write_tokens"] == 50

    def test_message_without_usage_metadata_contributes_zero(self):
        messages = [HumanMessage(content="hi"), _ai_message(10, 5)]
        totals = sum_usage(messages)
        assert totals["input_tokens"] == 10
        assert totals["output_tokens"] == 5

    def test_empty_and_none_input(self):
        assert sum_usage([]) == {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert sum_usage(None) == {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}

    def test_dict_shaped_messages_supported(self):
        messages = [{"usage_metadata": {"input_tokens": 4, "output_tokens": 2}}]
        totals = sum_usage(messages)
        assert totals["input_tokens"] == 4
        assert totals["output_tokens"] == 2


class TestPricePhase:
    def test_known_model_prices(self):
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 0, "cache_write_tokens": 0}
        cost = price_phase(usage, "openai/gpt-oss-20b")
        assert cost == 0.03 + 0.14

    def test_tier_name_resolves_to_model_id(self):
        usage = {"input_tokens": 1_000_000, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert price_phase(usage, "or-cheap") == 0.03

    def test_claude_top_is_free(self):
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert price_phase(usage, "claude-top") == 0.0

    def test_claude_verify_resolves_to_claude_top_pricing(self):
        """q.5 (2026-07-10): claude-verify is the DR_VERIFY_MODEL default (subscription
        `claude -p` shim) and must price as free, same as claude-top, not fall through
        to None or the old or-sonnet OpenRouter row."""
        usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert price_phase(usage, "claude-verify") == 0.0

    def test_unknown_model_returns_none_not_crash(self):
        usage = {"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert price_phase(usage, "some-made-up-model") is None

    def test_missing_model_returns_none(self):
        usage = {"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_write_tokens": 0}
        assert price_phase(usage, None) is None

    def test_cache_tokens_included_in_cost(self):
        base = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
        with_cache = dict(base, cache_read_tokens=1_000_000)
        cost = price_phase(with_cache, "openai/gpt-oss-20b")
        assert cost == 0.03


class TestBuildAccounting:
    def test_shape_has_phases_totals_dollar_cost(self):
        messages = [_ai_message(10, 5)]
        result = build_accounting(messages, {})
        assert set(result.keys()) == {"phases", "totals", "dollar_cost"}
        assert set(result["phases"].keys()) == {"research", "verify", "plan"}

    def test_verify_usage_absent_defaults_to_zero(self):
        messages = [_ai_message(10, 5)]
        result = build_accounting(messages, {})
        assert result["phases"]["verify"] == {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}

    def test_plan_usage_absent_defaults_to_zero(self):
        messages = [_ai_message(10, 5)]
        result = build_accounting(messages, {})
        assert result["phases"]["plan"] == {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}

    def test_verify_usage_present_folds_in(self):
        messages = [_ai_message(10, 5)]
        dr_run = {"verify_usage": {"input_tokens": 100, "output_tokens": 20}}
        result = build_accounting(messages, dr_run)
        assert result["phases"]["verify"]["input_tokens"] == 100
        assert result["phases"]["verify"]["output_tokens"] == 20
        assert result["totals"]["input_tokens"] == 110

    def test_plan_usage_present_folds_in(self):
        messages = [_ai_message(10, 5)]
        dr_run = {"plan_usage": {"input_tokens": 50, "output_tokens": 15}}
        result = build_accounting(messages, dr_run)
        assert result["phases"]["plan"]["input_tokens"] == 50
        assert result["phases"]["plan"]["output_tokens"] == 15
        assert result["totals"]["input_tokens"] == 60

    def test_totals_sum_all_phases(self):
        messages = [_ai_message(10, 5)]
        dr_run = {"verify_usage": {"input_tokens": 1, "output_tokens": 1}, "plan_usage": {"input_tokens": 2, "output_tokens": 2}}
        result = build_accounting(messages, dr_run)
        assert result["totals"]["input_tokens"] == 13
        assert result["totals"]["output_tokens"] == 8

    def test_dollar_cost_none_when_no_profile_tiers_given(self):
        messages = [_ai_message(1_000_000, 0)]
        result = build_accounting(messages, {})
        assert result["dollar_cost"] is None

    def test_dollar_cost_computed_with_profile_tiers(self):
        messages = [_ai_message(1_000_000, 0)]
        profile_tiers = {"gruntwork": "or-cheap", "verify": "or-mid", "top": "claude-top"}
        result = build_accounting(messages, {}, profile_tiers=profile_tiers)
        assert result["dollar_cost"] == 0.03

    def test_plan_phase_prices_against_the_gruntwork_tier(self):
        messages = []
        dr_run = {"plan_usage": {"input_tokens": 1_000_000, "output_tokens": 0}}
        profile_tiers = {"gruntwork": "or-cheap", "verify": "or-mid", "top": "claude-top"}
        result = build_accounting(messages, dr_run, profile_tiers=profile_tiers)
        assert result["dollar_cost"] == 0.03

    def test_unknown_tier_name_in_profile_degrades_to_none_without_crashing(self):
        messages = [_ai_message(1_000_000, 0)]
        profile_tiers = {"gruntwork": "nonexistent-tier", "verify": "or-mid", "top": "claude-top"}
        result = build_accounting(messages, {}, profile_tiers=profile_tiers)
        # research phase unpriced (None), verify phase zero-usage -> 0.0; total is 0.0, not None,
        # since at least one phase (verify) had a known, priceable model.
        assert result["dollar_cost"] == 0.0

    def test_dr_run_none_does_not_crash(self):
        result = build_accounting([], None)
        assert result["phases"]["research"]["input_tokens"] == 0

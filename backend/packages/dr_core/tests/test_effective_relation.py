"""Translated from harness/evals/state_test.js's "effectiveRelation (P0-4
conservative precedence)" block (trap b: ~12 materiality-asymmetric tests,
load-bearing). Test names keep the original JS names (snake_cased).

state.js's null-reviewer and 'unavailable'-reviewer paths collapse into a single
None case here per ruling divergence 3 (reviewer "unavailable" is None); both
original JS test names are kept even though they now exercise identical inputs.
"""

from dr_core.models.derive import effective_relation, is_grounding_relation
from dr_core.models.enums import Materiality, SupportRelation


def test_eff_agree_direct():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, SupportRelation.SUPPORTS_DIRECTLY, Materiality.HIGH) == "supports_directly"


def test_eff_weaker_reviewer_wins():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, SupportRelation.SUPPORTS_INFERENTIALLY, Materiality.HIGH) == "supports_inferentially"


def test_eff_weaker_extractor_wins():
    assert effective_relation(SupportRelation.QUALIFIES, SupportRelation.SUPPORTS_DIRECTLY, Materiality.HIGH) == "qualifies"


def test_eff_reviewer_context_only_ungrounds():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, SupportRelation.CONTEXT_ONLY, Materiality.HIGH) == "context_only"


def test_eff_context_only_is_not_grounding():
    rel = effective_relation(SupportRelation.SUPPORTS_DIRECTLY, SupportRelation.CONTEXT_ONLY, Materiality.HIGH)
    assert not is_grounding_relation(rel)


def test_eff_unavailable_plus_high_to_unreviewed():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, None, Materiality.HIGH) == "unreviewed"


def test_eff_unreviewed_not_grounding():
    assert not is_grounding_relation("unreviewed")


def test_eff_unavailable_plus_medium_inherits_extractor():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, None, Materiality.MEDIUM) == "supports_directly"


def test_eff_null_reviewer_plus_high_to_unreviewed():
    # Same input as test_eff_unavailable_plus_high_to_unreviewed post ruling
    # divergence 3 (reviewer "unavailable" is None, not a separate sentinel).
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, None, Materiality.HIGH) == "unreviewed"


def test_eff_reviewer_contradicts_wins():
    assert effective_relation(SupportRelation.SUPPORTS_DIRECTLY, SupportRelation.CONTRADICTS, Materiality.HIGH) == "contradicts"


def test_eff_extractor_contradicts_passes_through():
    assert effective_relation(SupportRelation.CONTRADICTS, SupportRelation.SUPPORTS_DIRECTLY, Materiality.HIGH) == "contradicts"


def test_eff_stale_passes_through():
    assert effective_relation(SupportRelation.STALE, SupportRelation.SUPPORTS_DIRECTLY, Materiality.HIGH) == "stale"

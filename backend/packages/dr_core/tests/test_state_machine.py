"""Translated from harness/evals/state_test.js. Covers every block EXCEPT
effectiveRelation (test_effective_relation.py) and the claimCaveat/PublicationStatus/
verification-transition logic (test_publication_status.py, authored fresh — see that
file's docstring for why state_test.js has no equivalent there).

NOT ported here (flagged for orchestrator review, see derive.py's module docstring):
rankClaimForProtection / selectProtectedClaimIds / capPendingClaims (claim-selection
rate-limit hardening over the engine's live claim pool) and normalizeRequirements
(raw-model-output ingestion). Neither is named in the ruling's section A schema or the
worker contract's SOURCE list, and both operate over ad hoc engine dict shapes that
don't map onto the pydantic ledger models.
"""

from dr_core.models.derive import (
    ClaimEvidence,
    StopBase,
    WorkItem,
    classify_stop,
    close_deliverable_on_synth,
    conflict_blocks_convergence,
    counts_as_attempt,
    derived_materiality,
    evidence_state,
    is_success_stop,
    materiality_uncalibrated,
    normalize_conflict_outcome,
    reopens_requirement,
    structured_grounded,
    terminal_state,
)
from dr_core.models.enums import (
    ConflictOutcome,
    CoverageRelation,
    DataProvenance,
    GateFlag,
    Materiality,
    RequirementKind,
    RequirementState,
    StopReason,
)
from dr_core.models.ledger import Claim, CoverageMapping, Requirement


def make_claim(**overrides) -> Claim:
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


def make_requirement(kind: RequirementKind, **overrides) -> Requirement:
    defaults = dict(id="r1", kind=kind, text="requirement text")
    defaults.update(overrides)
    return Requirement(**defaults)


def make_mapping(**overrides) -> CoverageMapping:
    defaults = dict(requirement_id="r1", claim_id="c1", relation=CoverageRelation.DIRECT)
    defaults.update(overrides)
    return CoverageMapping(**defaults)


# ---- structuredGrounded (P0-4: structured claims use a separate path) ----------


def test_struct_matched_high_grounded():
    claim = make_claim(data_ref={"value": "1"}, data_provenance=DataProvenance.MATCHED)
    assert structured_grounded(claim, Materiality.HIGH) is True


def test_struct_unaudited_high_not_grounded():
    claim = make_claim(data_ref={"value": "1"}, data_provenance=DataProvenance.UNAUDITED)
    assert structured_grounded(claim, Materiality.HIGH) is False


def test_struct_mismatch_not_grounded():
    claim = make_claim(data_ref={"value": "1"}, data_provenance=DataProvenance.MISMATCH)
    assert structured_grounded(claim, Materiality.HIGH) is False


def test_struct_missing_vintage_bars_grounding():
    claim = make_claim(data_ref={"value": "1"}, data_provenance=DataProvenance.MATCHED, gate_flags=[GateFlag.MISSING_VINTAGE])
    assert structured_grounded(claim, Materiality.HIGH) is False


def test_struct_non_data_ref_claim_is_not_structured_grounded():
    claim = make_claim(data_ref=None)
    assert structured_grounded(claim, Materiality.HIGH) is False


def test_struct_medium_unaudited_is_provisional_not_mismatch():
    claim = make_claim(data_ref={"value": "1"}, data_provenance=DataProvenance.UNAUDITED)
    assert structured_grounded(claim, Materiality.MEDIUM) is True


# ---- derivedMateriality (P0-3) --------------------------------------------------


def _reqs_r1_must_cover_r2_optional() -> dict[str, Requirement]:
    return {
        "r1": make_requirement(RequirementKind.ENTITY, id="r1", must_cover=True),
        "r2": make_requirement(RequirementKind.ENTITY, id="r2", must_cover=False),
    }


def test_mat_direct_to_must_cover_equals_high():
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=2)
    mappings = [make_mapping(requirement_id="r1", relation=CoverageRelation.DIRECT)]
    assert derived_materiality(claim, mappings, reqs) == Materiality.HIGH


def test_mat_partial_to_must_cover_honors_extractor_low():
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=2)
    mappings = [make_mapping(requirement_id="r1", relation=CoverageRelation.PARTIAL)]
    assert derived_materiality(claim, mappings, reqs) == Materiality.LOW


def test_mat_maps_to_optional_req_honors_extractor_high():
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=5)
    mappings = [make_mapping(requirement_id="r2", relation=CoverageRelation.DIRECT)]
    assert derived_materiality(claim, mappings, reqs) == Materiality.HIGH


def test_mat_maps_to_nothing_capped_at_medium():
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=5)
    assert derived_materiality(claim, [], reqs) == Materiality.MEDIUM


def test_mat_unmapped_low_stays_low():
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=1)
    assert derived_materiality(claim, [], reqs) == Materiality.LOW


def test_mat_ignores_preset_materiality_no_compounding_across_passes():
    # dr_core's Claim has no materiality field at all (derived-only, never stored),
    # so this uses claim.importance directly with no place for a stale value to hide.
    reqs = _reqs_r1_must_cover_r2_optional()
    claim = make_claim(importance=2)
    assert derived_materiality(claim, [], reqs) == Materiality.LOW


# ---- materialityUncalibrated -----------------------------------------------------


def test_calib_100_percent_high_warns():
    assert materiality_uncalibrated([Materiality.HIGH, Materiality.HIGH]) is True


def test_calib_balanced_does_not_warn():
    assert materiality_uncalibrated([Materiality.HIGH, Materiality.LOW, Materiality.MEDIUM]) is False


def test_calib_empty_does_not_warn():
    assert materiality_uncalibrated([]) is False


# ---- evidenceState: comparison (the false-substitute guard) ---------------------

_CMP = Requirement(id="r1", kind=RequirementKind.COMPARISON, text="beta vs IV", entities=["beta", "implied volatility"])
_BY_ID = {
    "c1": ClaimEvidence(source_id="s1", grounded=True),
    "c2": ClaimEvidence(source_id="s2", grounded=True),
}


def test_cmp_direct_with_relationship_stated_covered():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, relationship_stated=True)]
    assert evidence_state(_CMP, mappings, _BY_ID) == RequirementState.COVERED


def test_cmp_direct_relationship_stated_false_not_covered_partial():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, relationship_stated=False)]
    assert evidence_state(_CMP, mappings, _BY_ID) == RequirementState.PARTIAL


def test_cmp_fallback_operand_match_lenient_substring_covered():
    mappings = [
        make_mapping(
            claim_id="c1",
            relation=CoverageRelation.DIRECT,
            elements_satisfied=["equity beta", "implied volatility (IV)", "correlation"],
        )
    ]
    assert evidence_state(_CMP, mappings, _BY_ID) == RequirementState.COVERED


def test_cmp_fallback_low_beta_claim_mentioning_only_beta_not_covered_partial():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, elements_satisfied=["beta", "low-beta anomaly"])]
    assert evidence_state(_CMP, mappings, _BY_ID) == RequirementState.PARTIAL


def test_cmp_separate_a_and_b_claims_partial():
    mappings = [
        make_mapping(claim_id="c1", relation=CoverageRelation.PARTIAL, elements_satisfied=["beta"]),
        make_mapping(claim_id="c2", relation=CoverageRelation.PARTIAL, elements_satisfied=["implied volatility"]),
    ]
    assert evidence_state(_CMP, mappings, _BY_ID) == RequirementState.PARTIAL


def test_cmp_no_evidence_uncovered():
    assert evidence_state(_CMP, [], _BY_ID) == RequirementState.UNCOVERED


# ---- evidenceState: entity / metric / subtopic / deliverable -------------------


def test_entity_one_direct_covered():
    req = Requirement(id="r1", kind=RequirementKind.ENTITY, text="Spitznagel", entities=["Spitznagel"])
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, elements_satisfied=["Universa"])]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.COVERED


def test_entity_only_partial_partial():
    req = make_requirement(RequirementKind.ENTITY)
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.PARTIAL)]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.PARTIAL


def test_metric_direct_with_elements_covered():
    req = make_requirement(RequirementKind.METRIC)
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, elements_satisfied=["QLIKE", "0.42", "out-of-sample"])]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.COVERED


def test_metric_direct_without_elements_partial():
    req = make_requirement(RequirementKind.METRIC)
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT, elements_satisfied=[])]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.PARTIAL


def test_subtopic_2_claims_2_sources_covered():
    req = make_requirement(RequirementKind.SUBTOPIC)
    mappings = [
        make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT),
        make_mapping(claim_id="c2", relation=CoverageRelation.DIRECT),
    ]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.COVERED


def test_subtopic_2_claims_1_source_partial_thin():
    req = make_requirement(RequirementKind.SUBTOPIC)
    by_id = {"c1": ClaimEvidence(source_id="s1", grounded=True), "c1b": ClaimEvidence(source_id="s1", grounded=True)}
    mappings = [
        make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT),
        make_mapping(claim_id="c1b", relation=CoverageRelation.DIRECT),
    ]
    assert evidence_state(req, mappings, by_id) == RequirementState.PARTIAL


def test_deliverable_always_deferred():
    req = make_requirement(RequirementKind.DELIVERABLE)
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT)]
    assert evidence_state(req, mappings, _BY_ID) == RequirementState.DEFERRED


# ---- evidenceState: date_window -------------------------------------------------

_DW = Requirement(id="r1", kind=RequirementKind.DATE_WINDOW, text="frontier", window={"from": "2022-01-01", "to": "2026-12-31"})


def test_date_in_window_direct_covered():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT)]
    by_id = {"c1": ClaimEvidence(publication_date="2024-05-01", grounded=True)}
    assert evidence_state(_DW, mappings, by_id) == RequirementState.COVERED


def test_date_stale_direct_partial():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT)]
    by_id = {"c1": ClaimEvidence(publication_date="2019-01-01", grounded=True)}
    assert evidence_state(_DW, mappings, by_id) == RequirementState.PARTIAL


def test_date_no_parsed_window_plus_direct_covered_freshness_unverifiable_do_not_hang():
    req = make_requirement(RequirementKind.DATE_WINDOW)
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT)]
    by_id = {"c1": ClaimEvidence(publication_date="2024-01-01", grounded=True)}
    assert evidence_state(req, mappings, by_id) == RequirementState.COVERED


def test_date_no_parsed_window_plus_no_direct_uncovered():
    req = make_requirement(RequirementKind.DATE_WINDOW)
    assert evidence_state(req, [], {}) == RequirementState.UNCOVERED


# ---- evidenceState defensively drops ungrounded claims --------------------------


def test_grounded_filter_ungrounded_direct_ignored_uncovered():
    req = make_requirement(RequirementKind.ENTITY)
    mappings = [make_mapping(claim_id="cx", relation=CoverageRelation.DIRECT)]
    by_id = {"cx": ClaimEvidence(grounded=False)}
    assert evidence_state(req, mappings, by_id) == RequirementState.UNCOVERED


# ---- terminalState ----------------------------------------------------------------


def test_term_covered_stays_covered():
    req = make_requirement(RequirementKind.ENTITY)
    assert terminal_state(req, RequirementState.COVERED, run_ended=True, scheduled_at_least_once=True) == RequirementState.COVERED


def test_term_never_scheduled_not_attempted():
    req = make_requirement(RequirementKind.ENTITY)
    assert terminal_state(req, RequirementState.UNCOVERED, scheduled_at_least_once=False) == RequirementState.NOT_ATTEMPTED


def test_term_budget_ended_blocked_budget():
    req = make_requirement(RequirementKind.ENTITY)
    assert terminal_state(req, RequirementState.UNCOVERED, scheduled_at_least_once=True, budget_ended=True) == RequirementState.BLOCKED_BUDGET


def test_term_run_ended_uncovered_search_exhausted():
    req = make_requirement(RequirementKind.ENTITY)
    assert terminal_state(req, RequirementState.UNCOVERED, scheduled_at_least_once=True, run_ended=True) == RequirementState.SEARCH_EXHAUSTED


def test_term_run_ended_partial_partial():
    req = make_requirement(RequirementKind.ENTITY)
    assert terminal_state(req, RequirementState.PARTIAL, scheduled_at_least_once=True, run_ended=True) == RequirementState.PARTIAL


def test_term_deliverable_deferred():
    req = make_requirement(RequirementKind.DELIVERABLE)
    assert terminal_state(req, RequirementState.DEFERRED, scheduled_at_least_once=True, run_ended=True) == RequirementState.DEFERRED


# ---- closeDeliverablesOnSynth (ported per-requirement; caller assigns the result) --


def test_deliverable_close_successful_report_covers_deferred_deliverable():
    req = Requirement(id="r1", kind=RequirementKind.DELIVERABLE, text="a ranked table", must_cover=True, terminal_state=RequirementState.DEFERRED)
    assert close_deliverable_on_synth(req, True) == RequirementState.COVERED


def test_deliverable_close_non_deliverable_remains_unchanged():
    req = Requirement(id="r2", kind=RequirementKind.METRIC, text="a metric", must_cover=True, terminal_state=RequirementState.PARTIAL)
    assert close_deliverable_on_synth(req, True) == RequirementState.PARTIAL


def test_deliverable_close_failed_synth_leaves_deliverable_deferred():
    req = Requirement(id="r1", kind=RequirementKind.DELIVERABLE, text="a ranked table", must_cover=True, terminal_state=RequirementState.DEFERRED)
    assert close_deliverable_on_synth(req, False) == RequirementState.DEFERRED


# ---- reopensRequirement -----------------------------------------------------------


def test_reopen_only_direct_killed_reopen():
    mappings = [make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT)]
    assert reopens_requirement("c1", mappings) is True


def test_reopen_one_of_two_directs_killed_stays_covered():
    mappings = [
        make_mapping(claim_id="c1", relation=CoverageRelation.DIRECT),
        make_mapping(claim_id="c2", relation=CoverageRelation.DIRECT),
    ]
    assert reopens_requirement("c1", mappings) is False


def test_reopen_a_partial_killed_no_reopen():
    mappings = [
        make_mapping(claim_id="c1", relation=CoverageRelation.PARTIAL),
        make_mapping(claim_id="c2", relation=CoverageRelation.DIRECT),
    ]
    assert reopens_requirement("c1", mappings) is False


# ---- countsAsAttempt ---------------------------------------------------------------


def test_attempt_scheduled_plus_executed_counts():
    assert counts_as_attempt(WorkItem(scheduled=True, executed=True)) is True


def test_attempt_scheduled_only_does_not_count():
    assert counts_as_attempt(WorkItem(scheduled=True, executed=False)) is False


def test_attempt_pass_elapsing_does_not_count():
    assert counts_as_attempt(WorkItem()) is False


# ---- conflict outcomes --------------------------------------------------------------


def test_conflict_known_outcome_preserved():
    assert normalize_conflict_outcome("resolved_refutes") == ConflictOutcome.RESOLVED_REFUTES


def test_conflict_unknown_unresolved_persistent():
    assert normalize_conflict_outcome("garbage") == ConflictOutcome.UNRESOLVED_PERSISTENT


def test_conflict_persistent_blocks_convergence():
    assert conflict_blocks_convergence(ConflictOutcome.UNRESOLVED_PERSISTENT) is True


def test_conflict_resolved_does_not_block():
    assert conflict_blocks_convergence(ConflictOutcome.RESOLVED_SUPPORTS) is False


def test_conflict_null_blocks():
    assert conflict_blocks_convergence(None) is True


# ---- classifyStop (requirement-aware) -----------------------------------------------


def test_stop_converged_plus_no_open_req_equals_converged():
    assert classify_stop(StopBase(stop=True, reason="converged"), open_must_cover=0).reason == StopReason.CONVERGED


def test_stop_base_converged_but_open_must_cover_equals_completed_with_open_requirements():
    assert classify_stop(StopBase(stop=True, reason="converged"), open_must_cover=2).reason == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS


def test_stop_budget_plus_open_must_cover_equals_budget_exhausted():
    assert classify_stop(StopBase(stop=True, reason="budget"), open_must_cover=1).reason == StopReason.BUDGET_EXHAUSTED


def test_stop_budget_plus_zero_open_must_cover_is_still_budget_exhausted():
    assert classify_stop(StopBase(stop=True, reason="budget"), open_must_cover=0).reason == StopReason.BUDGET_EXHAUSTED


def test_stop_iter_cap_plus_open_must_cover_equals_iteration_cap():
    assert classify_stop(StopBase(stop=True, reason="iter-cap"), open_must_cover=1).reason == StopReason.ITERATION_CAP


def test_stop_diminishing_plus_open_must_cover_equals_completed_with_open_requirements():
    assert classify_stop(StopBase(stop=True, reason="diminishing"), open_must_cover=1).reason == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS


def test_stop_any_stop_plus_no_open_must_cover_equals_converged():
    assert classify_stop(StopBase(stop=True, reason="dead-end"), open_must_cover=0).reason == StopReason.CONVERGED


def test_stop_base_continue_stays_continue():
    assert classify_stop(StopBase(stop=False, reason="continue"), open_must_cover=3).reason == StopReason.CONTINUE


def test_stop_diminishing_plus_open_req_plus_min_attempts_not_met_continue():
    result = classify_stop(StopBase(stop=True, reason="diminishing"), open_must_cover=2, min_attempts_met=False)
    assert result.stop is False


def test_stop_dead_end_plus_open_req_plus_min_attempts_not_met_continue():
    result = classify_stop(StopBase(stop=True, reason="dead-end"), open_must_cover=2, min_attempts_met=False)
    assert result.reason == StopReason.CONTINUE


def test_stop_diminishing_plus_open_req_plus_min_attempts_met_completed_with_open_requirements():
    result = classify_stop(StopBase(stop=True, reason="diminishing"), open_must_cover=2, min_attempts_met=True)
    assert result.reason == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS


def test_stop_iter_cap_binds_even_when_min_attempts_not_met():
    result = classify_stop(StopBase(stop=True, reason="iter-cap"), open_must_cover=2, min_attempts_met=False)
    assert result.reason == StopReason.ITERATION_CAP


def test_stop_budget_binds_even_when_min_attempts_not_met():
    result = classify_stop(StopBase(stop=True, reason="budget"), open_must_cover=2, min_attempts_met=False)
    assert result.reason == StopReason.BUDGET_EXHAUSTED


def test_stop_only_converged_is_success():
    assert is_success_stop(StopReason.CONVERGED) is True
    assert is_success_stop(StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS) is False
    assert is_success_stop(StopReason.BUDGET_EXHAUSTED) is False

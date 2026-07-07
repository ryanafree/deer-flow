"""Tests for the PublicationStatus derivation, the claimCaveat table, and the
verification-status transition rules (REFUTATIONS_REQUIRED=2, trap c no-backward).

Unlike test_effective_relation.py / test_state_machine.py, these are NOT translated
from harness/evals/state_test.js — that suite only exercises state.js, and this logic
(aggregateVerificationStatus, claimCaveat, the eligibility precedence) lives in dr.js
instead (dr.js:106, :2557-2564, :2807-2816), which state_test.js does not cover. They
are authored fresh against the D2 ruling (fable-consult-02-RULING.md, section A) per
the worker contract's explicit requirement to port "verification transitions
(REFUTATIONS_REQUIRED=2) ... claimCaveat (the fixed GateFlag->string table — port
VERBATIM)".
"""

import pytest
from dr_core.models.derive import (
    aggregate_verification_status,
    assert_verification_transition_allowed,
    claim_caveat,
    publication_status,
)
from dr_core.models.enums import (
    CitationStatus,
    ConflictOutcome,
    DataProvenance,
    GateFlag,
    Materiality,
    PublicationStatus,
    VerificationStatus,
)
from dr_core.models.ledger import Claim, Conflict, Vote


def make_claim(**overrides) -> Claim:
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


def vote(refuted=False, abstain=False) -> Vote:
    return Vote(vote_id="v", refuted=refuted, abstain=abstain, confidence="high", reasoning="")


# ---- claimCaveat (dr.js:2807-2816, VERBATIM table, first-match order) --------------


@pytest.mark.parametrize(
    ("flag", "expected"),
    [
        (GateFlag.VENDOR_REPORTED, "vendor-reported and not independently reproduced"),
        (GateFlag.UNREPRODUCED_PAPER_CLAIM, "preprint benchmark without released code"),
        (GateFlag.OVERSTATED_EVIDENCE, "the source type does not support the claim strength"),
        (GateFlag.PERIOD_MISMATCH, "the source period does not match the claimed period"),
        (GateFlag.MISSING_VINTAGE, "the required source vintage was unavailable"),
        (GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE, "the number lacks a primary-source trace"),
    ],
)
def test_claim_caveat_table_verbatim(flag, expected):
    assert claim_caveat(make_claim(gate_flags=[flag])) == expected


def test_claim_caveat_no_flags_is_none():
    assert claim_caveat(make_claim(gate_flags=[])) is None


def test_claim_caveat_first_match_wins_priority_order():
    # vendor_reported precedes overstated_evidence in the JS if-chain.
    claim = make_claim(gate_flags=[GateFlag.OVERSTATED_EVIDENCE, GateFlag.VENDOR_REPORTED])
    assert claim_caveat(claim) == "vendor-reported and not independently reproduced"


# ---- aggregateVerificationStatus (dr.js:2557-2564, REFUTATIONS_REQUIRED=2) ---------


def test_aggregate_single_clean_vote_supported():
    assert aggregate_verification_status([vote()]) == VerificationStatus.SUPPORTED


def test_aggregate_two_refutes_of_three_killed_on_refute():
    votes = [vote(refuted=True), vote(refuted=True), vote()]
    assert aggregate_verification_status(votes) == VerificationStatus.KILLED_ON_REFUTE


def test_aggregate_one_refute_of_three_not_killed():
    votes = [vote(refuted=True), vote(), vote()]
    assert aggregate_verification_status(votes) == VerificationStatus.SUPPORTED


def test_aggregate_all_abstain_not_verified():
    votes = [vote(abstain=True), vote(abstain=True), vote(abstain=True)]
    assert aggregate_verification_status(votes) == VerificationStatus.NOT_VERIFIED


def test_aggregate_empty_votes_supported_vacuously():
    # No refutes (0 >= 2 is false) and the all-abstain check is vacuously false for an
    # empty list, mirroring dr.js's `valid.length && abstains === valid.length`.
    assert aggregate_verification_status([]) == VerificationStatus.SUPPORTED


# ---- verification transition guard (trap c: killed_on_refute is terminal) ---------


def test_transition_killed_on_refute_to_anything_raises():
    with pytest.raises(ValueError):
        assert_verification_transition_allowed(VerificationStatus.KILLED_ON_REFUTE, VerificationStatus.PENDING)


def test_transition_killed_on_refute_to_itself_is_a_noop_not_an_error():
    assert_verification_transition_allowed(VerificationStatus.KILLED_ON_REFUTE, VerificationStatus.KILLED_ON_REFUTE)


def test_transition_supported_to_pending_reselect_is_legal():
    assert_verification_transition_allowed(VerificationStatus.SUPPORTED, VerificationStatus.PENDING)


def test_transition_not_verified_to_pending_reselect_is_legal():
    assert_verification_transition_allowed(VerificationStatus.NOT_VERIFIED, VerificationStatus.PENDING)


def test_transition_pending_to_supported_is_legal():
    assert_verification_transition_allowed(VerificationStatus.PENDING, VerificationStatus.SUPPORTED)


# ---- publication_status: FIXED PRECEDENCE (ruling section A) ----------------------


def _supported_claim(**overrides) -> Claim:
    defaults = dict(verification={"status": VerificationStatus.SUPPORTED, "complete": True})
    defaults.update(overrides)
    return make_claim(**defaults)


def test_publication_step1_killed_on_refute_excludes():
    claim = _supported_claim(verification={"status": VerificationStatus.KILLED_ON_REFUTE, "complete": True})
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True)
    assert result == PublicationStatus.EXCLUDED


def test_publication_step1_citation_not_found_excludes():
    claim = _supported_claim(citation_status=CitationStatus.NOT_FOUND)
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True)
    assert result == PublicationStatus.EXCLUDED


def test_publication_step1_data_provenance_mismatch_excludes():
    claim = _supported_claim(data_provenance=DataProvenance.MISMATCH)
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True)
    assert result == PublicationStatus.EXCLUDED


def test_publication_step2_gate_flag_contests_even_if_supported_and_grounded():
    claim = _supported_claim(gate_flags=[GateFlag.PERIOD_MISMATCH])
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True)
    assert result == PublicationStatus.CONTESTED


def test_publication_step2_unresolved_persistent_conflict_contests():
    claim = _supported_claim()
    conflicts = [Conflict(id="k1", claim_ids=["c1"], description="d", outcome=ConflictOutcome.UNRESOLVED_PERSISTENT)]
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True, conflicts=conflicts)
    assert result == PublicationStatus.CONTESTED


def test_publication_step2_conflict_with_none_outcome_contests():
    claim = _supported_claim()
    conflicts = [Conflict(id="k1", claim_ids=["c1"], description="d", outcome=None)]
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True, conflicts=conflicts)
    assert result == PublicationStatus.CONTESTED


def test_publication_step2_resolved_conflict_does_not_contest():
    claim = _supported_claim()
    conflicts = [Conflict(id="k1", claim_ids=["c1"], description="d", outcome=ConflictOutcome.RESOLVED_SUPPORTS)]
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True, conflicts=conflicts)
    assert result == PublicationStatus.SUPPORTED


def test_publication_step3_supported_and_grounded_supported():
    claim = _supported_claim()
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=True)
    assert result == PublicationStatus.SUPPORTED


def test_publication_step3_supported_but_not_grounded_falls_through_to_not_verified():
    # context_only/unreviewed effective relations never ground; a claim can be
    # verification-status SUPPORTED yet still not reach PublicationStatus.SUPPORTED.
    claim = _supported_claim()
    result = publication_status(claim, materiality=Materiality.HIGH, grounded=False)
    assert result == PublicationStatus.NOT_VERIFIED


def test_publication_step4_default_not_verified():
    claim = make_claim()  # pending verification, no flags, no conflicts
    result = publication_status(claim, materiality=Materiality.MEDIUM, grounded=False)
    assert result == PublicationStatus.NOT_VERIFIED

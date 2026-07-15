"""Tests for dr_core.verify.provenance (D8 provenance audit, amended by D11
item 3: value/unit/period comparison + post-verify rescue removal). Pure, no
I/O, except the last class which drives verify_node end-to-end (still
hermetic -- vote model and network calls monkeypatched) to pin the removed
rescue's absence."""

import json
from types import SimpleNamespace

from dr_core.graph.state import merge_ledger
from dr_core.graph.verify import verify_node
from dr_core.models.enums import DataProvenance, GateFlag, VerificationStatus
from dr_core.models.ledger import Claim
from dr_core.verify import votes as votes_mod
from dr_core.verify.provenance import audit_provenance


def _claim(**overrides) -> Claim:
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


class TestAuditProvenanceNoDataRef:
    def test_none_data_ref_is_not_audited(self):
        assert audit_provenance(_claim(data_ref=None)) is None


class TestAuditProvenanceMismatch:
    def test_missing_vintage_flag_forces_mismatch(self):
        claim = _claim(data_ref={"value": "1", "period": "2023-09-30"}, gate_flags=[GateFlag.MISSING_VINTAGE])
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_period_mismatch_flag_forces_mismatch(self):
        claim = _claim(data_ref={"value": "1", "period": "2023-09-30"}, gate_flags=[GateFlag.PERIOD_MISMATCH])
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_numeric_without_primary_trace_flag_forces_mismatch(self):
        claim = _claim(data_ref={"value": "1", "period": "2023-09-30"}, gate_flags=[GateFlag.NUMERIC_WITHOUT_PRIMARY_TRACE])
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_claimed_period_disagreeing_with_data_ref_period_is_mismatch(self):
        """The stale-vintage trap: a claim asserting FY2024 backed by a data_ref
        whose actual period is FY2023. The claimed_period/period disagreement is
        checked BEFORE the D11 item 3 value comparison, so it still fires even
        though the claim's own $383.285bn figure numerically matches the record."""
        claim = _claim(
            text="Apple's FY2024 net sales were $383.285 billion.",
            data_ref={"value": "383285000000", "period": "2023-09-30", "claimed_period": "2024-09-30", "source_class": "primary_filing"},
        )
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_unrelated_gate_flag_does_not_force_mismatch(self):
        """VENDOR_REPORTED is not one of the three blocking flags, and the
        claim's own asserted figure matches the record, so this still reaches
        MATCHED (D11 item 3: value/unit/period match is now also required, in
        addition to the untouched source_class/period gate)."""
        claim = _claim(
            text="The reported figure was $1.",
            data_ref={"value": "1", "period": "2023-09-30", "source_class": "primary_filing"},
            gate_flags=[GateFlag.VENDOR_REPORTED],
        )
        assert audit_provenance(claim) == DataProvenance.MATCHED


class TestAuditProvenanceValueMatch:
    """D11 item 3: MATCHED now additionally requires the claim's own asserted
    figure to affirmatively match the data_ref's value/unit within tolerance
    after scale normalization."""

    def test_exact_value_match_is_matched(self):
        claim = _claim(
            text="Apple's reported revenue was $10,000,000.",
            data_ref={"value": "10000000", "period": "2024-09-30", "source_class": "primary_filing"},
        )
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_scale_artifact_same_value_is_matched_not_mismatch(self):
        """The scale-artifact guard (D11 item 3's binding amendment): a claim
        written as "$1.5 million" against a raw data_ref value of "1500000"
        must normalize to the same quantity, not read as a contradiction."""
        claim = _claim(
            text="Quarterly revenue was $1.5 million.",
            data_ref={"value": "1500000", "period": "2024-03-31", "source_class": "primary_filing"},
        )
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_genuine_value_contradiction_is_mismatch(self):
        claim = _claim(
            text="Quarterly revenue was $500 million.",
            data_ref={"value": "300000000", "period": "2024-03-31", "source_class": "primary_filing"},
        )
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_unparseable_claim_value_stays_unaudited(self):
        """The binding guard: a claim value that cannot be parsed never
        produces MISMATCH, even though the record has a comparable value."""
        claim = _claim(
            text="Quarterly revenue grew substantially year over year.",
            data_ref={"value": "1500000", "period": "2024-03-31", "source_class": "primary_filing"},
        )
        assert audit_provenance(claim) is None

    def test_no_value_in_structured_record_stays_unaudited(self):
        """S9-C: WRDS's data_ref conventionally carries only period/source_class
        (no "value" key) -- with no comparable structured record value, D11
        item 3 means this can no longer reach MATCHED (pre-D11 behavior; see
        DECISIONS.md D11 item 3's accepted cost)."""
        claim = _claim(text="AAPL closed at $150 on that date.", data_ref={"period": "2023", "source_class": "primary_database"})
        assert audit_provenance(claim) is None

    def test_percent_claim_against_raw_record_is_inconclusive_not_matched(self):
        """Different normalized units (percent vs. a raw count) are
        inconclusive -- never an affirmative match, even if the bare numbers
        happen to coincide."""
        claim = _claim(text="Growth was 5%.", data_ref={"value": "5", "period": "2024-01-01", "source_class": "official_stat"})
        assert audit_provenance(claim) is None


class TestAuditProvenanceMatched:
    def test_primary_source_with_period_and_no_flags_matches(self):
        claim = _claim(text="Net income was $1.", data_ref={"value": "1", "period": "2024-09-30", "source_class": "primary_filing"})
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_official_stat_source_class_matches(self):
        claim = _claim(text="The rate was $1.", data_ref={"value": "1", "period": "2024-01-01", "source_class": "official_stat"})
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_primary_database_source_class_matches(self):
        """S9-C: WRDS's CRSP/Compustat data_ref source_class, now with a
        comparable value present so the D11 item 3 value check can affirm the
        match (see TestAuditProvenanceValueMatch for the no-value case)."""
        claim = _claim(text="The closing price was $150.", data_ref={"value": "150", "period": "2023", "source_class": "primary_database"})
        assert audit_provenance(claim) == DataProvenance.MATCHED


class TestAuditProvenanceInconclusive:
    def test_non_primary_source_class_stays_unaudited(self):
        claim = _claim(text="The figure was $1.", data_ref={"value": "1", "period": "2024-09-30", "source_class": "press"})
        assert audit_provenance(claim) is None

    def test_primary_without_period_stays_unaudited(self):
        claim = _claim(text="The figure was $1.", data_ref={"value": "1", "source_class": "primary_filing"})
        assert audit_provenance(claim) is None

    def test_already_terminal_provenance_is_a_no_op(self):
        claim = _claim(data_ref={"value": "1", "period": "2024-09-30"}, data_provenance=DataProvenance.MATCHED)
        assert audit_provenance(claim) is None
        claim2 = _claim(data_ref={"value": "1", "period": "2024-09-30"}, gate_flags=[GateFlag.PERIOD_MISMATCH], data_provenance=DataProvenance.MISMATCH)
        assert audit_provenance(claim2) is None


class TestAuditNeverMutatesGateFlags:
    def test_gate_flags_untouched_by_audit(self):
        claim = _claim(data_ref={"value": "1", "period": "2024-09-30"}, gate_flags=[GateFlag.VENDOR_REPORTED])
        audit_provenance(claim)
        assert claim.gate_flags == [GateFlag.VENDOR_REPORTED]


class TestPostVerifyRescueRemoved:
    """D11 item 3 (D8 sub-clause reversal): the post-verify SUPPORTED+UNAUDITED
    -> MATCHED rescue is gone. Drives verify_node end-to-end (hermetic: vote
    model and evidence fetch monkeypatched, no network) on a data_ref claim
    whose deterministic audit is inconclusive (no comparable record value), and
    asserts that even after the claim lands SUPPORTED via the adaptive vote
    pass, data_provenance stays UNAUDITED."""

    async def test_supported_data_ref_claim_stays_unaudited(self, monkeypatch):
        def _clean_vote_response() -> SimpleNamespace:
            content = json.dumps({"refuted": False, "abstain": False, "confidence": "high", "reasoning": "well supported"})
            return SimpleNamespace(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cache_read": 0, "cache_creation": 0}})

        class _FakeCleanModel:
            async def ainvoke(self, messages):
                return _clean_vote_response()

        async def _no_evidence(url):
            return None

        monkeypatch.setattr("dr_core.graph.verify._fetch_evidence", _no_evidence)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeCleanModel())

        claim = Claim(
            claim_id="c-rescue",
            text="The metric was reported for the period, exact figure not restated here.",
            importance=4,
            source_id="s1",
            data_ref={"period": "2024-01-01", "source_class": "primary_filing"},  # no "value" -> audit stays inconclusive
        )
        dr_claims = {claim.claim_id: claim.model_dump(mode="json")}
        dr_sources = {"s1": {"id": "s1", "url_or_id": "https://example.com/s1", "source_system": "edgar", "authority_tier": 1, "retrieved_at": "2026-07-11T00:00:00+00:00"}}
        state = {"dr_claims": dr_claims, "dr_sources": dr_sources, "dr_run": {"depth": "quick"}}

        result = await verify_node(state)
        merged = merge_ledger(dr_claims, result.get("dr_claims"))
        rescued = Claim.model_validate(merged["c-rescue"])

        assert rescued.verification.status == VerificationStatus.SUPPORTED
        assert rescued.verification.complete is True
        assert rescued.data_provenance == DataProvenance.UNAUDITED

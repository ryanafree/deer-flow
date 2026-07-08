"""Tests for dr_core.verify.provenance (D8 provenance audit). Pure, no I/O."""

from dr_core.models.enums import DataProvenance, GateFlag, VerificationStatus
from dr_core.models.ledger import Claim, VerificationRecord
from dr_core.verify.provenance import audit_provenance, rescue_unaudited_matched


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
        whose actual period is FY2023."""
        claim = _claim(data_ref={"value": "383285000000", "period": "2023-09-30", "claimed_period": "2024-09-30", "source_class": "primary_filing"})
        assert audit_provenance(claim) == DataProvenance.MISMATCH

    def test_unrelated_gate_flag_does_not_force_mismatch(self):
        claim = _claim(data_ref={"value": "1", "period": "2023-09-30", "source_class": "primary_filing"}, gate_flags=[GateFlag.VENDOR_REPORTED])
        assert audit_provenance(claim) == DataProvenance.MATCHED


class TestAuditProvenanceMatched:
    def test_primary_source_with_period_and_no_flags_matches(self):
        claim = _claim(data_ref={"value": "1", "period": "2024-09-30", "source_class": "primary_filing"})
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_official_stat_source_class_matches(self):
        claim = _claim(data_ref={"value": "1", "period": "2024-01-01", "source_class": "official_stat"})
        assert audit_provenance(claim) == DataProvenance.MATCHED


class TestAuditProvenanceInconclusive:
    def test_non_primary_source_class_stays_unaudited(self):
        claim = _claim(data_ref={"value": "1", "period": "2024-09-30", "source_class": "press"})
        assert audit_provenance(claim) is None

    def test_primary_without_period_stays_unaudited(self):
        claim = _claim(data_ref={"value": "1", "source_class": "primary_filing"})
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


class TestRescueUnauditedMatched:
    def test_supported_data_ref_claim_still_unaudited_is_rescued(self):
        claim = _claim(
            data_ref={"value": "1", "period": "2024-09-30"},
            data_provenance=DataProvenance.UNAUDITED,
            verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        )
        assert rescue_unaudited_matched(claim) == DataProvenance.MATCHED

    def test_not_supported_is_not_rescued(self):
        claim = _claim(
            data_ref={"value": "1", "period": "2024-09-30"},
            data_provenance=DataProvenance.UNAUDITED,
            verification=VerificationRecord(status=VerificationStatus.NOT_VERIFIED, complete=True),
        )
        assert rescue_unaudited_matched(claim) is None

    def test_already_matched_is_not_re_rescued(self):
        claim = _claim(
            data_ref={"value": "1", "period": "2024-09-30"},
            data_provenance=DataProvenance.MATCHED,
            verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        )
        assert rescue_unaudited_matched(claim) is None

    def test_no_data_ref_is_not_rescued(self):
        claim = _claim(data_ref=None, verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True))
        assert rescue_unaudited_matched(claim) is None

"""record_claim (C1 claim seam) tests — D6 ruling A.

HERMETIC: no live model, no network. Calls the tool's underlying function
directly (``record_claim.func``) with a synthetic ``InjectedState``-shaped
dict + a tool_call_id, mirroring the direct-invocation style already used for
DrLedgerMiddleware's before_model hook in test_dr_source_hook.py. Where the
duplicate-divergent guard matters, the returned Command's dr_claims update is
folded through the REAL merge_ledger to prove a divergent re-assert never
reaches (and never raises inside) the reducer.
"""

from dr_core.graph.claim_tool import record_claim
from dr_core.graph.state import merge_ledger
from dr_core.models import Claim, CitationStatus, DataProvenance, VerificationStatus

SOURCE_ID = "src1"


def _state(dr_claims: dict | None = None) -> dict:
    return {"dr_sources": {SOURCE_ID: {"id": SOURCE_ID, "url_or_id": "https://example.com"}}, "dr_claims": dr_claims or {}}


def _record(**overrides) -> dict:
    args = dict(text="The sky is blue.", source_id=SOURCE_ID, quote="the sky appears blue", importance=3, tool_call_id="tc1", state=_state())
    args.update(overrides)
    return record_claim.func(**args)


class TestUrlCitationResolution:
    """The model only sees urls in search/fetch results (never the sha256
    ledger ids), so record_claim resolves an exact-url citation to the C2
    hook's deterministic id. Pinned by the S6 live E2E finding."""

    URL = "https://mw.example.gov/minimum-wage"

    def _url_state(self):
        import hashlib

        ledger_id = hashlib.sha256(self.URL.encode("utf-8")).hexdigest()[:16]
        return ledger_id, {"dr_sources": {ledger_id: {"id": ledger_id, "url_or_id": self.URL}}, "dr_claims": {}}

    def test_url_citation_resolves_to_ledger_id(self):
        ledger_id, state = self._url_state()
        out = _record(source_id=self.URL, state=state)
        ((_, payload),) = out.update["dr_claims"].items()
        assert payload["source_id"] == ledger_id

    def test_url_and_id_citations_yield_same_claim_id(self):
        ledger_id, state = self._url_state()
        by_url = _record(source_id=self.URL, state=state)
        by_id = _record(source_id=ledger_id, state=state)
        assert list(by_url.update["dr_claims"]) == list(by_id.update["dr_claims"])

    def test_unknown_url_still_rejected(self):
        _, state = self._url_state()
        out = _record(source_id="https://not-retrieved.example.com/", state=state)
        assert "dr_claims" not in out.update
        assert "not found" in out.update["messages"][0].content


class TestValidClaim:
    def test_records_claim_under_deterministic_id(self):
        out = _record()
        assert "dr_claims" in out.update
        ((claim_id, payload),) = out.update["dr_claims"].items()
        assert payload["text"] == "The sky is blue."
        assert payload["source_id"] == SOURCE_ID
        message = out.update["messages"][0]
        assert message.tool_call_id == "tc1"
        assert claim_id in message.content

    def test_dump_reloads_via_claim_model(self):
        out = _record()
        ((claim_id, payload),) = out.update["dr_claims"].items()
        claim = Claim(**payload)
        assert claim.claim_id == claim_id
        assert claim.support.quote == "the sky appears blue"

    def test_folding_through_merge_ledger_from_empty_works(self):
        out = _record()
        merged = merge_ledger(None, out.update["dr_claims"])
        assert merged == out.update["dr_claims"]

    def test_initial_verification_citation_provenance_at_exact_defaults(self):
        out = _record()
        ((_, payload),) = out.update["dr_claims"].items()
        assert payload["verification"]["status"] == VerificationStatus.PENDING.value
        assert payload["verification"]["complete"] is False
        assert payload["verification"]["selected"] is False
        assert payload["verification"]["votes"] == []
        assert payload["citation_status"] == CitationStatus.UNRESOLVED.value
        assert payload["data_provenance"] == DataProvenance.UNAUDITED.value

    def test_success_command_sets_deliverable_marker(self):
        """D7: a recorded claim is definitionally a deliverable, so the
        success Command must set dr_run["deliverable"]=True for
        route_after_research to gate the turn."""
        out = _record()
        assert out.update["dr_run"] == {"deliverable": True}


class TestUnknownSource:
    def test_rejects_with_no_dr_claims_key(self):
        out = _record(source_id="not-a-real-source", state=_state())
        assert "dr_claims" not in out.update
        message = out.update["messages"][0]
        assert "not-a-real-source" in message.content
        assert message.tool_call_id == "tc1"


class TestDeterministicId:
    def test_same_text_and_source_yields_same_id(self):
        out_1 = _record(tool_call_id="tc1")
        out_2 = _record(tool_call_id="tc2", state=_state(out_1.update["dr_claims"]))
        # Second call is a duplicate re-assert (identical payload) -- confirms,
        # writes nothing -- but the id it references must match the first.
        assert "dr_claims" not in out_2.update
        ((claim_id_1, _),) = out_1.update["dr_claims"].items()
        assert claim_id_1 in out_2.update["messages"][0].content

    def test_different_source_id_yields_different_id(self):
        other_source = "src2"
        state = _state()
        state["dr_sources"][other_source] = {"id": other_source, "url_or_id": "https://example.org"}

        out_1 = _record(state=state)
        out_2 = _record(source_id=other_source, state=state)

        ((claim_id_1, _),) = out_1.update["dr_claims"].items()
        ((claim_id_2, _),) = out_2.update["dr_claims"].items()
        assert claim_id_1 != claim_id_2


class TestDuplicateIdentical:
    def test_reassert_is_noop_confirm_and_does_not_raise_through_merge_ledger(self):
        first = _record(tool_call_id="tc1")
        dr_claims = merge_ledger(None, first.update["dr_claims"])

        second = _record(tool_call_id="tc2", state=_state(dr_claims))
        assert "dr_claims" not in second.update
        assert "already recorded" in second.update["messages"][0].content

        # Nothing new to fold, but prove folding an absent update is inert.
        merged = merge_ledger(dr_claims, second.update.get("dr_claims"))
        assert merged == dr_claims


class TestDuplicateDivergent:
    def test_same_id_different_importance_keeps_first_and_writes_nothing(self):
        first = _record(importance=3, tool_call_id="tc1")
        dr_claims = merge_ledger(None, first.update["dr_claims"])
        ((claim_id, first_payload),) = dr_claims.items()
        assert first_payload["importance"] == 3

        second = _record(importance=4, tool_call_id="tc2", state=_state(dr_claims))

        # Guard fires: no write in the returned update at all.
        assert "dr_claims" not in second.update
        assert claim_id in second.update["messages"][0].content

        # merge_ledger is never handed the divergent pair -- dr_claims still
        # holds only the FIRST assertion (importance=3).
        assert dr_claims[claim_id]["importance"] == 3


class TestDataRef:
    """S9-C: record_claim accepts an optional data_ref, passed through verbatim
    to the Claim so the provenance audit can see it -- the one content field
    the model doesn't author, only relays from a structured tool's payload."""

    def test_data_ref_lands_on_the_claim_unmodified(self):
        data_ref = {"period": "2023", "source_class": "primary_database"}
        out = _record(data_ref=data_ref)
        ((_, payload),) = out.update["dr_claims"].items()
        assert payload["data_ref"] == data_ref

    def test_omitted_data_ref_defaults_to_none(self):
        out = _record()
        ((_, payload),) = out.update["dr_claims"].items()
        assert payload["data_ref"] is None

    def test_data_ref_reloads_via_claim_model(self):
        data_ref = {"period": "2023-09-30", "claimed_period": "2024-09-30", "source_class": "primary_filing"}
        out = _record(data_ref=data_ref)
        ((_, payload),) = out.update["dr_claims"].items()
        claim = Claim(**payload)
        assert claim.data_ref == data_ref


class TestGateFlags:
    def test_valid_hyphenated_flag_is_normalized_and_lands(self):
        out = _record(gate_flags=["vendor-reported"])
        ((_, payload),) = out.update["dr_claims"].items()
        assert payload["gate_flags"] == ["vendor_reported"]

    def test_unknown_flag_is_rejected_with_no_write(self):
        out = _record(gate_flags=["not-a-real-flag"])
        assert "dr_claims" not in out.update
        assert "not-a-real-flag" in out.update["messages"][0].content

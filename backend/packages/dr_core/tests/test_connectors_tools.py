"""Tests for dr_core.connectors.tools (S9-C batch 1: the WRDS structured tool).

HERMETIC: no live network/database. wrds_client's connection + query functions
are monkeypatched at their WRDS_QUERY_BOUNDARY (get_connection /
fetch_crsp_price / fetch_compustat_fundamentals) -- everything else in this
file drives the REAL tool function, the REAL DrLedgerMiddleware source hook,
the REAL record_claim tool, the REAL provenance audit, and the REAL
DrProfileToolMiddleware + connector registry + profile YAMLs. This is the
acceptance-item demonstration HANDOFF_CONNECTORS_BC.md calls for when live
WRDS auth is unavailable: "the same demonstrated end-to-end with a mocked
WRDS response through the REAL tool + middleware path."
"""

from __future__ import annotations

import json

from dr_core.connectors import tools as tools_mod
from dr_core.connectors import wrds_client
from dr_core.connectors.registry import load_connectors
from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS
from dr_core.graph.claim_tool import record_claim
from dr_core.graph.middleware import SOURCE_TOOL_NAMES, DrLedgerMiddleware
from dr_core.graph.profile_middleware import DrProfileToolMiddleware
from dr_core.graph.state import merge_ledger
from dr_core.models.enums import DataProvenance
from dr_core.models.ledger import Claim
from dr_core.verify.provenance import audit_provenance
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, ToolMessage


class TestToolRegistration:
    def test_wrds_query_is_registered_to_the_wrds_connector(self):
        assert TOOL_TO_CONNECTORS["wrds_query"] == frozenset({"wrds"})

    def test_tool_name_is_distinct_from_community_hardcoded_names(self):
        # web_search/web_fetch are hardcoded by community search/fetch tools
        # (Option A finding, BUILD_LEDGER 2026-07-08) -- a Stage-C tool must
        # never collide with them.
        assert tools_mod.WRDS_TOOL_NAME not in {"web_search", "web_fetch"}

    def test_wrds_query_is_a_source_bearing_tool(self):
        assert "wrds_query" in SOURCE_TOOL_NAMES

    def test_connector_tools_middleware_contributes_wrds_query(self):
        assert tools_mod.DrConnectorToolsMiddleware.tools == [tools_mod.wrds_query]


def _mock_wrds(monkeypatch, crsp=None, compustat=None, connect_ok=True):
    if connect_ok:
        monkeypatch.setattr(wrds_client, "get_connection", lambda: object())
    else:

        def _raise():
            raise wrds_client.WrdsUnavailable("no credentials")

        monkeypatch.setattr(wrds_client, "get_connection", _raise)
    monkeypatch.setattr(wrds_client, "fetch_crsp_price", lambda ticker, period, conn=None: crsp)
    monkeypatch.setattr(wrds_client, "fetch_compustat_fundamentals", lambda ticker, period, conn=None: compustat)


class TestWrdsQueryTool:
    def test_success_payload_shape(self, monkeypatch):
        crsp = {"date": "2023-12-29", "prc": 192.53}
        compustat = {"fyear": 2023, "revt": 383285000000.0}
        _mock_wrds(monkeypatch, crsp=crsp, compustat=compustat)

        out = json.loads(tools_mod.wrds_query.func("AAPL", "2023"))

        assert out["ok"] is True
        assert out["source_system"] == "wrds"
        assert out["source_class"] == "primary_database"
        assert out["url_or_id"] == "wrds://crsp-compustat/AAPL/2023"
        assert out["data_ref"] == {"period": "2023", "source_class": "primary_database"}
        assert out["crsp"] == crsp
        assert out["compustat"] == compustat

    def test_data_ref_period_prefers_compustat_fyear_over_crsp_date(self, monkeypatch):
        _mock_wrds(monkeypatch, crsp={"date": "2023-09-28"}, compustat={"fyear": 2023})
        out = json.loads(tools_mod.wrds_query.func("AAPL", "2023"))
        assert out["data_ref"]["period"] == "2023"

    def test_data_ref_falls_back_to_crsp_date_when_no_compustat_row(self, monkeypatch):
        _mock_wrds(monkeypatch, crsp={"date": "2023-09-28"}, compustat=None)
        out = json.loads(tools_mod.wrds_query.func("AAPL", "2023"))
        assert out["data_ref"]["period"] == "2023-09-28"

    def test_both_none_yields_ok_false(self, monkeypatch):
        _mock_wrds(monkeypatch, crsp=None, compustat=None)
        out = json.loads(tools_mod.wrds_query.func("ZZZZ", "2023"))
        assert out["ok"] is False
        assert "no CRSP price or Compustat" in out["error"]

    def test_connection_unavailable_yields_ok_false_never_raises(self, monkeypatch):
        _mock_wrds(monkeypatch, connect_ok=False)
        out = json.loads(tools_mod.wrds_query.func("AAPL", "2023"))
        assert out["ok"] is False
        assert "WRDS unavailable" in out["error"]

    def test_query_error_yields_ok_false_never_raises(self, monkeypatch):
        monkeypatch.setattr(wrds_client, "get_connection", lambda: object())

        def _raise(*a, **k):
            raise wrds_client.WrdsQueryError("server exploded")

        monkeypatch.setattr(wrds_client, "fetch_crsp_price", _raise)
        monkeypatch.setattr(wrds_client, "fetch_compustat_fundamentals", lambda *a, **k: None)
        out = json.loads(tools_mod.wrds_query.func("AAPL", "2023"))
        assert out["ok"] is False
        assert "server exploded" in out["error"]

    def test_bad_period_yields_ok_false_never_raises(self, monkeypatch):
        monkeypatch.setattr(wrds_client, "get_connection", lambda: object())
        out = json.loads(tools_mod.wrds_query.func("AAPL", "not-a-period"))
        assert out["ok"] is False


class TestEndToEndRealToolPlusMiddlewarePath:
    """The acceptance demonstration: real wrds_query -> real DrLedgerMiddleware
    source hook -> real record_claim -> real merge_ledger -> real
    audit_provenance, all live code paths. Only wrds_client's DB boundary is
    mocked (the auth-blocked-case acceptance shape)."""

    def test_data_ref_claim_from_wrds_reaches_matched_provenance_live(self, monkeypatch):
        crsp = {"date": "2023-12-29", "prc": 192.53}
        compustat = {"fyear": 2023, "revt": 383285000000.0}
        _mock_wrds(monkeypatch, crsp=crsp, compustat=compustat)

        # 1. Real tool call.
        tool_content = tools_mod.wrds_query.func("AAPL", "2023")
        payload = json.loads(tool_content)
        assert payload["ok"] is True

        # 2. Real C2 source-extraction hook turns the ToolMessage into a Source.
        tool_call_id = "call_wrds_1"
        messages = [
            AIMessage(content="", tool_calls=[{"name": "wrds_query", "args": {"ticker": "AAPL", "period": "2023"}, "id": tool_call_id, "type": "tool_call"}]),
            ToolMessage(content=tool_content, tool_call_id=tool_call_id, name="wrds_query"),
        ]
        hook_out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        assert hook_out is not None
        dr_sources = merge_ledger(None, hook_out["dr_sources"])
        ((source_id, source_record),) = dr_sources.items()
        assert source_record["source_system"] == "wrds"
        assert source_record["authority_tier"] == 1

        # 3. Real record_claim tool, passing the tool's data_ref VERBATIM.
        state = {"dr_sources": dr_sources, "dr_claims": {}}
        record_out = record_claim.func(
            text="Apple's FY2023 revenue was $383.285 billion per Compustat.",
            source_id=payload["url_or_id"],
            quote="revt 383285000000.0",
            importance=5,
            tool_call_id="tc1",
            state=state,
            data_ref=payload["data_ref"],
        )
        assert "dr_claims" in record_out.update
        dr_claims = merge_ledger(None, record_out.update["dr_claims"])
        ((claim_id, claim_payload),) = dr_claims.items()
        claim = Claim(**claim_payload)
        assert claim.data_ref == {"period": "2023", "source_class": "primary_database"}

        # 4. Real provenance audit derives MATCHED -- not a fixture Claim, the
        # data_ref flowed from the real tool's output through the real
        # record_claim path.
        assert audit_provenance(claim) == DataProvenance.MATCHED

    def test_mismatched_claimed_period_still_derives_mismatch_live(self, monkeypatch):
        """Same real path, but the model (mis-)asserts a different claimed_period
        than the tool actually returned -- the audit must still catch it."""
        _mock_wrds(monkeypatch, crsp=None, compustat={"fyear": 2023})
        tool_content = tools_mod.wrds_query.func("AAPL", "2023")
        payload = json.loads(tool_content)

        tool_call_id = "call_wrds_2"
        messages = [
            AIMessage(content="", tool_calls=[{"name": "wrds_query", "args": {"ticker": "AAPL", "period": "2023"}, "id": tool_call_id, "type": "tool_call"}]),
            ToolMessage(content=tool_content, tool_call_id=tool_call_id, name="wrds_query"),
        ]
        hook_out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        dr_sources = merge_ledger(None, hook_out["dr_sources"])

        divergent_data_ref = dict(payload["data_ref"])
        divergent_data_ref["claimed_period"] = "2024"  # model mislabels the fiscal year
        record_out = record_claim.func(
            text="Apple's FY2024 revenue was $383.285 billion.",
            source_id=payload["url_or_id"],
            quote="revt 383285000000.0",
            importance=5,
            tool_call_id="tc1",
            state={"dr_sources": dr_sources, "dr_claims": {}},
            data_ref=divergent_data_ref,
        )
        dr_claims = merge_ledger(None, record_out.update["dr_claims"])
        ((_, claim_payload),) = dr_claims.items()
        claim = Claim(**claim_payload)
        assert audit_provenance(claim) == DataProvenance.MISMATCH


class TestProfileEnforcementWithTheRealTool:
    """Deferred-from-Stage-B live check (HANDOFF_CONNECTORS_BC.md): a
    legal-profile run cannot call the WRDS tool. Now cheap -- the real tool
    exists, so this exercises the REAL registry, REAL legal.yaml/financial.yaml,
    and the REAL wrds_query tool object (not a synthetic stand-in name)."""

    def _middleware(self) -> DrProfileToolMiddleware:
        return DrProfileToolMiddleware(connectors=load_connectors())

    def test_financial_profile_keeps_the_real_wrds_tool_bound(self):
        mw = self._middleware()
        request = ModelRequest(model=None, messages=[], tools=[tools_mod.wrds_query], state={"dr_run": {"profile": "financial"}})
        filtered = mw._filter_tools(request)
        assert [t.name for t in filtered.tools] == ["wrds_query"]

    def test_legal_profile_drops_the_real_wrds_tool_from_the_bound_schema(self):
        mw = self._middleware()
        request = ModelRequest(model=None, messages=[], tools=[tools_mod.wrds_query], state={"dr_run": {"profile": "legal"}})
        filtered = mw._filter_tools(request)
        assert filtered.tools == []

    def test_legal_profile_blocks_a_direct_invocation_attempt_at_execution_time(self):
        from types import SimpleNamespace

        mw = self._middleware()
        request = SimpleNamespace(tool_call={"name": "wrds_query", "id": "tc1"}, state={"dr_run": {"profile": "legal"}})

        def _handler(_req):  # pragma: no cover -- must never be reached
            raise AssertionError("wrds_query must be blocked before the real tool handler runs under the legal profile")

        result = mw.wrap_tool_call(request, _handler)
        assert result.status == "error"
        assert "not in the 'legal' profile's connector allowlist" in result.content

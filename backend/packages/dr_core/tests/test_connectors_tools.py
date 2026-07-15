"""Tests for dr_core.connectors.tools (S9-C: WRDS from batch 1;
EDGAR/FRED/CourtListener from batch 2).

HERMETIC: no live network/database. Each client module's own query boundary
is monkeypatched (wrds_client's get_connection/fetch_crsp_price/
fetch_compustat_fundamentals; edgar_client's resolve_cik/fetch_company_concept;
fred_client's fetch_series_observations; courtlistener_client's
search_opinions) -- everything else in this file drives the REAL tool
function, the REAL DrLedgerMiddleware source hook, the REAL record_claim
tool, the REAL provenance audit, and the REAL DrProfileToolMiddleware +
connector registry + profile YAMLs. This is the acceptance-item
demonstration HANDOFF_CONNECTORS_BC.md calls for when live auth is
unavailable: "the same demonstrated end-to-end with a mocked ... response
through the REAL tool + middleware path."
"""

from __future__ import annotations

import json

from dr_core.connectors import courtlistener_client, edgar_client, fred_client, wrds_client
from dr_core.connectors import tools as tools_mod
from dr_core.connectors.binding import get_default_registry
from dr_core.connectors.registry import load_connectors
from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS
from dr_core.graph.claim_tool import record_claim
from dr_core.graph.middleware import DrLedgerMiddleware
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
        # D11 P0: source-bearing-ness is now resolved through the
        # connectors.binding registry, not a static name set.
        assert get_default_registry().resolve("wrds_query") is not None

    def test_connector_tools_middleware_contributes_wrds_query(self):
        # Batch 2 (EDGAR/FRED/CourtListener) extends this list -- see
        # TestBatch2ToolRegistration::test_connector_tools_middleware_contributes_all_four_tools
        # for the full-membership assertion; this one just pins wrds_query's
        # continued presence.
        assert tools_mod.wrds_query in tools_mod.DrConnectorToolsMiddleware.tools


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

    def test_data_ref_claim_from_wrds_stays_unaudited_no_value_in_record_live(self, monkeypatch):
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

        # 4. Real provenance audit -- not a fixture Claim, the data_ref flowed
        # from the real tool's output through the real record_claim path.
        # D11 item 3 (DECISIONS.md): MATCHED now also requires an affirmative
        # value/unit match against the data_ref's own "value", but WRDS's
        # data_ref (see the assertion above) conventionally carries only
        # period/source_class, no "value" -- there is nothing to compare, so
        # this stays UNAUDITED. Accepted cost of D11 item 3, not a regression:
        # a data_ref claim with no comparable structured record value never
        # reaches MATCHED (see test_verify_provenance.py's
        # TestAuditProvenanceValueMatch::test_no_value_in_structured_record_stays_unaudited).
        assert audit_provenance(claim) is None

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


# ---------------------------------------------------------------------------
# Batch 2: EDGAR, FRED, CourtListener
# ---------------------------------------------------------------------------


class TestBatch2ToolRegistration:
    def test_all_three_tools_registered_to_their_connectors(self):
        assert TOOL_TO_CONNECTORS["edgar_company_facts"] == frozenset({"edgar"})
        assert TOOL_TO_CONNECTORS["fred_series"] == frozenset({"fred"})
        assert TOOL_TO_CONNECTORS["courtlistener_search"] == frozenset({"courtlistener"})

    def test_tool_names_are_distinct_from_community_hardcoded_names(self):
        assert {tools_mod.EDGAR_TOOL_NAME, tools_mod.FRED_TOOL_NAME, tools_mod.COURTLISTENER_TOOL_NAME}.isdisjoint({"web_search", "web_fetch"})

    def test_all_three_are_source_bearing_tools(self):
        registry = get_default_registry()
        for name in ("edgar_company_facts", "fred_series", "courtlistener_search"):
            assert registry.resolve(name) is not None

    def test_connector_tools_middleware_contributes_all_five_tools(self):
        names = {t.name for t in tools_mod.DrConnectorToolsMiddleware.tools}
        assert names == {"wrds_query", "edgar_company_facts", "fred_series", "courtlistener_search", "academic_search"}


def _mock_edgar(monkeypatch, cik10="0000320193", company="Apple Inc.", fact=None, resolve_ok=True):
    if resolve_ok:
        monkeypatch.setattr(edgar_client, "resolve_cik", lambda ticker_or_cik, transport=None: (cik10, company))
    else:

        def _raise(*a, **k):
            raise edgar_client.EdgarUnavailable("no UA")

        monkeypatch.setattr(edgar_client, "resolve_cik", _raise)
    monkeypatch.setattr(edgar_client, "fetch_company_concept", lambda cik10, concept, period, taxonomy="us-gaap", unit="USD", transport=None: fact)


class TestEdgarCompanyFactsTool:
    def test_success_payload_shape(self, monkeypatch):
        fact = {
            "cik10": "0000320193",
            "taxonomy": "us-gaap",
            "concept": "Revenues",
            "label": "Revenues",
            "entity_name": "Apple Inc.",
            "unit": "USD",
            "value": 383285000000,
            "end": "2023-09-30",
            "start": "2022-09-25",
            "fy": 2023,
            "fp": "FY",
            "form": "10-K",
            "accn": "0000320193-23-000106",
            "filed": "2023-11-03",
            "url": "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json",
        }
        _mock_edgar(monkeypatch, fact=fact)

        out = json.loads(tools_mod.edgar_company_facts.func("AAPL", "Revenues", "2023"))

        assert out["ok"] is True
        assert out["source_system"] == "edgar"
        assert out["source_class"] == "primary_filing"
        assert out["url_or_id"] == "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json#end=2023-09-30&accn=0000320193-23-000106"
        assert out["data_ref"] == {"period": "2023-09-30", "source_class": "primary_filing"}
        assert out["value"] == 383285000000

    def test_no_matching_fact_yields_ok_false(self, monkeypatch):
        _mock_edgar(monkeypatch, fact=None)
        out = json.loads(tools_mod.edgar_company_facts.func("AAPL", "Revenues", "2019"))
        assert out["ok"] is False

    def test_unavailable_yields_ok_false_never_raises(self, monkeypatch):
        _mock_edgar(monkeypatch, resolve_ok=False)
        out = json.loads(tools_mod.edgar_company_facts.func("AAPL", "Revenues", "2023"))
        assert out["ok"] is False
        assert "EDGAR unavailable" in out["error"]

    def test_query_error_yields_ok_false_never_raises(self, monkeypatch):
        monkeypatch.setattr(edgar_client, "resolve_cik", lambda ticker_or_cik, transport=None: ("0000320193", "Apple Inc."))

        def _raise(*a, **k):
            raise edgar_client.EdgarQueryError("server exploded")

        monkeypatch.setattr(edgar_client, "fetch_company_concept", _raise)
        out = json.loads(tools_mod.edgar_company_facts.func("AAPL", "Revenues", "2023"))
        assert out["ok"] is False
        assert "server exploded" in out["error"]


def _mock_fred(monkeypatch, result=None, unavailable=False):
    if unavailable:

        def _raise(*a, **k):
            raise fred_client.FredUnavailable("no key")

        monkeypatch.setattr(fred_client, "fetch_series_observations", _raise)
    else:
        monkeypatch.setattr(fred_client, "fetch_series_observations", lambda series_id, period, transport=None: result)


class TestFredSeriesTool:
    def test_success_payload_shape(self, monkeypatch):
        result = {"series_id": "GDP", "start": "2023-01-01", "end": "2023-12-31", "observations": [{"date": "2023-01-01", "value": "26000.0"}, {"date": "2023-10-01", "value": "27957.2"}]}
        _mock_fred(monkeypatch, result=result)

        out = json.loads(tools_mod.fred_series.func("GDP", "2023"))

        assert out["ok"] is True
        assert out["source_system"] == "fred"
        assert out["source_class"] == "official_stat"
        assert out["url_or_id"] == "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&observation_start=2023-01-01&observation_end=2023-12-31&file_type=json"
        assert "api_key" not in out["url_or_id"]
        assert out["data_ref"] == {"period": "2023-10-01", "source_class": "official_stat"}
        assert out["latest_observation"] == {"date": "2023-10-01", "value": "27957.2"}

    def test_no_observations_yields_ok_false(self, monkeypatch):
        _mock_fred(monkeypatch, result=None)
        out = json.loads(tools_mod.fred_series.func("BOGUS", "2023"))
        assert out["ok"] is False

    def test_unavailable_yields_ok_false_never_raises(self, monkeypatch):
        _mock_fred(monkeypatch, unavailable=True)
        out = json.loads(tools_mod.fred_series.func("GDP", "2023"))
        assert out["ok"] is False
        assert "FRED unavailable" in out["error"]


def _mock_courtlistener(monkeypatch, results=None, unavailable=False):
    if unavailable:

        def _raise(*a, **k):
            raise courtlistener_client.CourtListenerUnavailable("no token")

        monkeypatch.setattr(courtlistener_client, "search_opinions", _raise)
    else:
        monkeypatch.setattr(courtlistener_client, "search_opinions", lambda query, court=None, filed_after=None, filed_before=None, transport=None: results)


class TestCourtListenerSearchTool:
    def test_success_payload_shape_has_no_top_level_data_ref(self, monkeypatch):
        results = [
            {
                "case_name": "Harlow v. Fitzgerald",
                "court": "scotus",
                "date_filed": "1982-06-24",
                "citation": ["457 U.S. 800"],
                "cluster_id": 111,
                "absolute_url": "https://www.courtlistener.com/opinion/111/harlow-v-fitzgerald/",
                "snippet": "...",
            }
        ]
        _mock_courtlistener(monkeypatch, results=results)

        out = json.loads(tools_mod.courtlistener_search.func("qualified immunity"))

        assert out["ok"] is True
        assert out["source_system"] == "courtlistener"
        assert "data_ref" not in out
        assert out["count"] == 1
        assert out["results"][0]["url_or_id"] == "https://www.courtlistener.com/opinion/111/harlow-v-fitzgerald/"
        assert out["results"][0]["title"] == "Harlow v. Fitzgerald"

    def test_no_results_yields_ok_false(self, monkeypatch):
        _mock_courtlistener(monkeypatch, results=None)
        out = json.loads(tools_mod.courtlistener_search.func("nonexistent case xyz"))
        assert out["ok"] is False

    def test_unavailable_yields_ok_false_never_raises(self, monkeypatch):
        _mock_courtlistener(monkeypatch, unavailable=True)
        out = json.loads(tools_mod.courtlistener_search.func("x"))
        assert out["ok"] is False
        assert "CourtListener unavailable" in out["error"]


class TestEdgarFredEndToEndRealToolPlusMiddlewarePath:
    """Same acceptance shape as WRDS's TestEndToEndRealToolPlusMiddlewarePath:
    real tool -> real C2 source hook -> real record_claim -> real
    audit_provenance, live code paths, only the client's query boundary
    mocked."""

    def test_edgar_data_ref_claim_stays_unaudited_no_value_in_record_live(self, monkeypatch):
        fact = {
            "cik10": "0000320193",
            "taxonomy": "us-gaap",
            "concept": "Revenues",
            "entity_name": "Apple Inc.",
            "unit": "USD",
            "value": 383285000000,
            "end": "2023-09-30",
            "form": "10-K",
            "accn": "0000320193-23-000106",
            "filed": "2023-11-03",
            "url": "https://data.sec.gov/api/xbrl/companyconcept/CIK0000320193/us-gaap/Revenues.json",
        }
        _mock_edgar(monkeypatch, fact=fact)

        tool_content = tools_mod.edgar_company_facts.func("AAPL", "Revenues", "2023")
        payload = json.loads(tool_content)
        assert payload["ok"] is True

        tool_call_id = "call_edgar_1"
        messages = [
            AIMessage(content="", tool_calls=[{"name": "edgar_company_facts", "args": {"ticker_or_cik": "AAPL", "concept": "Revenues", "period": "2023"}, "id": tool_call_id, "type": "tool_call"}]),
            ToolMessage(content=tool_content, tool_call_id=tool_call_id, name="edgar_company_facts"),
        ]
        hook_out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        dr_sources = merge_ledger(None, hook_out["dr_sources"])
        ((_, source_record),) = dr_sources.items()
        assert source_record["source_system"] == "edgar"
        assert source_record["authority_tier"] == 1

        record_out = record_claim.func(
            text="Apple's FY2023 revenue was $383.285 billion per SEC EDGAR.",
            source_id=payload["url_or_id"],
            quote="Revenues 383285000000",
            importance=5,
            tool_call_id="tc1",
            state={"dr_sources": dr_sources, "dr_claims": {}},
            data_ref=payload["data_ref"],
        )
        dr_claims = merge_ledger(None, record_out.update["dr_claims"])
        ((_, claim_payload),) = dr_claims.items()
        claim = Claim(**claim_payload)
        # D11 item 3: EDGAR's data_ref (period/source_class only, no "value")
        # has no comparable structured record value, so this stays UNAUDITED --
        # accepted cost of D11 item 3, not a regression (see
        # test_verify_provenance.py's TestAuditProvenanceValueMatch).
        assert audit_provenance(claim) is None

    def test_fred_data_ref_claim_stays_unaudited_no_value_in_record_live(self, monkeypatch):
        result = {"series_id": "GDP", "start": "2023-01-01", "end": "2023-12-31", "observations": [{"date": "2023-10-01", "value": "27957.2"}]}
        _mock_fred(monkeypatch, result=result)

        tool_content = tools_mod.fred_series.func("GDP", "2023")
        payload = json.loads(tool_content)
        assert payload["ok"] is True

        tool_call_id = "call_fred_1"
        messages = [
            AIMessage(content="", tool_calls=[{"name": "fred_series", "args": {"series_id": "GDP", "period": "2023"}, "id": tool_call_id, "type": "tool_call"}]),
            ToolMessage(content=tool_content, tool_call_id=tool_call_id, name="fred_series"),
        ]
        hook_out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        dr_sources = merge_ledger(None, hook_out["dr_sources"])
        ((_, source_record),) = dr_sources.items()
        assert source_record["source_system"] == "fred"
        assert source_record["authority_tier"] == 1

        record_out = record_claim.func(
            text="US GDP was $27.957 trillion in Q4 2023 per FRED.",
            source_id=payload["url_or_id"],
            quote="27957.2",
            importance=5,
            tool_call_id="tc1",
            state={"dr_sources": dr_sources, "dr_claims": {}},
            data_ref=payload["data_ref"],
        )
        dr_claims = merge_ledger(None, record_out.update["dr_claims"])
        ((_, claim_payload),) = dr_claims.items()
        claim = Claim(**claim_payload)
        # D11 item 3: FRED's data_ref (period/source_class only, no "value")
        # has no comparable structured record value, so this stays UNAUDITED --
        # accepted cost of D11 item 3, not a regression (see
        # test_verify_provenance.py's TestAuditProvenanceValueMatch).
        assert audit_provenance(claim) is None


class TestCourtListenerSourceHookMintsOnePerResult:
    def test_two_results_mint_two_distinct_sources(self, monkeypatch):
        results = [
            {"case_name": "Case A", "court": "scotus", "date_filed": "1980-01-01", "citation": [], "cluster_id": 1, "absolute_url": "https://www.courtlistener.com/opinion/1/case-a/", "snippet": "a"},
            {"case_name": "Case B", "court": "ca9", "date_filed": "1990-01-01", "citation": [], "cluster_id": 2, "absolute_url": "https://www.courtlistener.com/opinion/2/case-b/", "snippet": "b"},
        ]
        _mock_courtlistener(monkeypatch, results=results)
        tool_content = tools_mod.courtlistener_search.func("immunity")

        tool_call_id = "call_cl_1"
        messages = [
            AIMessage(content="", tool_calls=[{"name": "courtlistener_search", "args": {"query": "immunity"}, "id": tool_call_id, "type": "tool_call"}]),
            ToolMessage(content=tool_content, tool_call_id=tool_call_id, name="courtlistener_search"),
        ]
        hook_out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        dr_sources = merge_ledger(None, hook_out["dr_sources"])

        assert len(dr_sources) == 2
        systems = {s["source_system"] for s in dr_sources.values()}
        tiers = {s["authority_tier"] for s in dr_sources.values()}
        urls = {s["url_or_id"] for s in dr_sources.values()}
        assert systems == {"courtlistener"}
        assert tiers == {1}
        assert urls == {"https://www.courtlistener.com/opinion/1/case-a/", "https://www.courtlistener.com/opinion/2/case-b/"}


class TestBatch2ProfileEnforcementWithTheRealTools:
    """Mirrors TestProfileEnforcementWithTheRealTool for the three batch-2
    tools: edgar/fred are financial-only, courtlistener is legal-only."""

    def _middleware(self) -> DrProfileToolMiddleware:
        return DrProfileToolMiddleware(connectors=load_connectors())

    def test_financial_profile_keeps_edgar_and_fred_bound_drops_courtlistener(self):
        mw = self._middleware()
        request = ModelRequest(model=None, messages=[], tools=[tools_mod.edgar_company_facts, tools_mod.fred_series, tools_mod.courtlistener_search], state={"dr_run": {"profile": "financial"}})
        filtered = mw._filter_tools(request)
        assert {t.name for t in filtered.tools} == {"edgar_company_facts", "fred_series"}

    def test_legal_profile_keeps_courtlistener_bound_drops_edgar_and_fred(self):
        mw = self._middleware()
        request = ModelRequest(model=None, messages=[], tools=[tools_mod.edgar_company_facts, tools_mod.fred_series, tools_mod.courtlistener_search], state={"dr_run": {"profile": "legal"}})
        filtered = mw._filter_tools(request)
        assert {t.name for t in filtered.tools} == {"courtlistener_search"}

    def test_legal_profile_blocks_a_direct_edgar_invocation_at_execution_time(self):
        from types import SimpleNamespace

        mw = self._middleware()
        request = SimpleNamespace(tool_call={"name": "edgar_company_facts", "id": "tc1"}, state={"dr_run": {"profile": "legal"}})

        def _handler(_req):  # pragma: no cover -- must never be reached
            raise AssertionError("edgar_company_facts must be blocked before the real tool handler runs under the legal profile")

        result = mw.wrap_tool_call(request, _handler)
        assert result.status == "error"
        assert "not in the 'legal' profile's connector allowlist" in result.content

    def test_financial_profile_blocks_a_direct_courtlistener_invocation_at_execution_time(self):
        from types import SimpleNamespace

        mw = self._middleware()
        request = SimpleNamespace(tool_call={"name": "courtlistener_search", "id": "tc1"}, state={"dr_run": {"profile": "financial"}})

        def _handler(_req):  # pragma: no cover -- must never be reached
            raise AssertionError("courtlistener_search must be blocked before the real tool handler runs under the financial profile")

        result = mw.wrap_tool_call(request, _handler)
        assert result.status == "error"
        assert "not in the 'financial' profile's connector allowlist" in result.content

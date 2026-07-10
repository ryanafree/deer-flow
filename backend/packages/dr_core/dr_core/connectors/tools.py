"""tools.py — Stage C: typed LangChain tools minted from the connector
registry (structured connectors: WRDS from batch 1; EDGAR/FRED/CourtListener
from batch 2 of HANDOFF_CONNECTORS_BC.md's "C — structured tools").

Binding contract (tool_map.py's docstring): every tool this module mints gets
a name DISTINCT from web_search/web_fetch (community tools hardcode those)
and calls ``register_tool(tool_name, connector_name)`` at creation time
(module import), before any graph is built.

Tools are wired into the bound tool set the same zero-FORK_DELTA way
``DrLedgerMiddleware`` contributes ``record_claim``: an ``AgentMiddleware``
subclass with a ``tools`` class attribute, collected by the harness's agent
factory (see claim_tool.py's docstring). ``DrConnectorToolsMiddleware`` here
is that seam for Stage C.

Two source-minting shapes (see graph/middleware.py):
  - single-fact lookups (wrds_query, edgar_company_facts, fred_series) mint
    ONE Source per tool call from a top-level ``url_or_id`` -- the C2 hook's
    ``_source_from_structured_tool`` parser, keyed via
    ``_STRUCTURED_TOOL_SOURCE_SYSTEMS``.
  - search tools (courtlistener_search) mint ONE Source PER RESULT from a
    ``results: [{url_or_id, title, ...}, ...]`` list -- the C2 hook's
    ``_sources_from_structured_search_tool`` parser, keyed via
    ``_STRUCTURED_SEARCH_TOOL_SOURCE_SYSTEMS``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from dr_core.connectors import courtlistener_client, edgar_client, fred_client, wrds_client
from dr_core.connectors.tool_map import register_tool

WRDS_TOOL_NAME = "wrds_query"
WRDS_SOURCE_CLASS = "primary_database"

EDGAR_TOOL_NAME = "edgar_company_facts"
EDGAR_SOURCE_CLASS = "primary_filing"  # reuses fetch/structured.py's fetch_edgar convention

FRED_TOOL_NAME = "fred_series"
FRED_SOURCE_CLASS = "official_stat"  # reuses fetch/structured.py's fetch_fred convention

COURTLISTENER_TOOL_NAME = "courtlistener_search"


def _fail(error: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": error, **extra})


def _wrds_url_or_id(ticker: str, period: str) -> str:
    """Deterministic citation handle for a WRDS lookup -- not a real URL (WRDS
    is a database, not HTTP), but the same "the model already has this string
    from its own tool call" shape web_fetch's url provides, so record_claim's
    citation-resolution path (middleware._source_id over url_or_id) works
    unchanged for a non-HTTP structured source."""
    return f"wrds://crsp-compustat/{ticker.upper()}/{period}"


def _wrds_fail(error: str, **extra: Any) -> str:
    return _fail(error, **extra)


@tool
def wrds_query(ticker: str, period: str) -> str:
    """Look up CRSP security price and Compustat standardized annual
    fundamentals for a US-listed ticker in a given period, from WRDS
    (institutional academic data: security prices, standardized financial
    statements).

    Args:
        ticker: exchange ticker symbol, e.g. "AAPL".
        period: "YYYY" (a fiscal/calendar year), "YYYY-MM", or "YYYY-MM-DD".
            CRSP resolves the nearest trading-day price within the period;
            Compustat resolves the standardized annual fundamentals for that
            fiscal year.

    Returns JSON. On success (`ok: true`) the payload carries a `url_or_id`
    field -- cite this EXACT string as record_claim's `source_id` argument --
    and a `data_ref` object; pass that `data_ref` dict VERBATIM as
    record_claim's `data_ref` argument when asserting a claim grounded in this
    data, so the provenance audit can verify the claim against the retrieved
    period. On failure (`ok: false`) do not cite this result as a source.
    """
    try:
        conn = wrds_client.get_connection()
    except wrds_client.WrdsUnavailable as exc:
        return _wrds_fail(f"WRDS unavailable: {exc}", ticker=ticker, period=period)

    try:
        crsp = wrds_client.fetch_crsp_price(ticker, period, conn=conn)
    except wrds_client.WrdsQueryError as exc:
        return _wrds_fail(str(exc), ticker=ticker, period=period)
    except ValueError as exc:
        return _wrds_fail(str(exc), ticker=ticker, period=period)

    try:
        compustat = wrds_client.fetch_compustat_fundamentals(ticker, period, conn=conn)
    except wrds_client.WrdsQueryError as exc:
        return _wrds_fail(str(exc), ticker=ticker, period=period)
    except ValueError as exc:
        return _wrds_fail(str(exc), ticker=ticker, period=period)

    if crsp is None and compustat is None:
        return _wrds_fail("no CRSP price or Compustat fundamentals found for that ticker/period", ticker=ticker, period=period)

    # The RETRIEVED data point's own period -- Compustat's fyear (an int) wins
    # when present (fundamentals are the more period-precise of the two);
    # otherwise CRSP's own trade date. This is data_ref["period"], never the
    # claim's asserted period (that distinction is provenance.py's mismatch
    # check, decided at record_claim time, not here).
    resolved_period = str((compustat or {}).get("fyear")) if compustat else str((crsp or {}).get("date"))
    url_or_id = _wrds_url_or_id(ticker, period)
    payload = {
        "ok": True,
        "source_system": "wrds",
        "source_class": WRDS_SOURCE_CLASS,
        "url_or_id": url_or_id,
        "title": f"WRDS CRSP/Compustat {ticker.upper()} {period}",
        "ticker": ticker.upper(),
        "period": period,
        "crsp": crsp,
        "compustat": compustat,
        "data_ref": {"period": resolved_period, "source_class": WRDS_SOURCE_CLASS},
    }
    return json.dumps(payload, default=str)


@tool
def edgar_company_facts(ticker_or_cik: str, concept: str, period: str) -> str:
    """Look up a single SEC EDGAR XBRL fact (annual, 10-K-sourced) for a US
    public company -- e.g. Revenues, Assets, NetIncomeLoss -- for a given
    fiscal year, from SEC's data.sec.gov companyconcept API (primary-filing
    data, not a crawled page).

    Args:
        ticker_or_cik: exchange ticker symbol (e.g. "AAPL") or a numeric SEC
            CIK.
        concept: the exact us-gaap XBRL tag name, e.g. "Revenues", "Assets",
            "NetIncomeLoss" (case-sensitive, SEC's own spelling).
        period: "YYYY" -- the fiscal year whose 10-K-reported annual figure
            you want. Disambiguated by the fact's own reporting-period END
            date, never the filing's fiscal-year field (a single 10-K
            reports multiple prior fiscal years, all tagged with the
            filing's own fy).

    Returns JSON. On success (`ok: true`) the payload carries a `url_or_id`
    field -- cite this EXACT string as record_claim's `source_id` argument --
    and a `data_ref` object; pass that `data_ref` dict VERBATIM as
    record_claim's `data_ref` argument. On failure (`ok: false`) do not cite
    this result as a source.
    """
    try:
        cik10, company_name = edgar_client.resolve_cik(ticker_or_cik)
    except edgar_client.EdgarUnavailable as exc:
        return _fail(f"EDGAR unavailable: {exc}", ticker_or_cik=ticker_or_cik, concept=concept, period=period)
    except edgar_client.EdgarQueryError as exc:
        return _fail(str(exc), ticker_or_cik=ticker_or_cik, concept=concept, period=period)

    try:
        fact = edgar_client.fetch_company_concept(cik10, concept, period)
    except edgar_client.EdgarUnavailable as exc:
        return _fail(f"EDGAR unavailable: {exc}", ticker_or_cik=ticker_or_cik, concept=concept, period=period)
    except (edgar_client.EdgarQueryError, ValueError) as exc:
        return _fail(str(exc), ticker_or_cik=ticker_or_cik, concept=concept, period=period)

    if fact is None:
        return _fail("no matching annual (10-K) fact for that ticker/concept/period", ticker_or_cik=ticker_or_cik, concept=concept, period=period)

    url_or_id = f"{fact['url']}#end={fact['end']}&accn={fact['accn']}"
    payload = {
        "ok": True,
        "source_system": "edgar",
        "source_class": EDGAR_SOURCE_CLASS,
        "url_or_id": url_or_id,
        "title": f"SEC EDGAR XBRL {fact.get('entity_name') or ticker_or_cik.upper()} {concept} FY ending {fact['end']}",
        "cik": cik10,
        "company": company_name or fact.get("entity_name"),
        "concept": concept,
        "value": fact["value"],
        "unit": fact["unit"],
        "form": fact["form"],
        "filed": fact["filed"],
        "data_ref": {"period": fact["end"], "source_class": EDGAR_SOURCE_CLASS},
    }
    return json.dumps(payload, default=str)


@tool
def fred_series(series_id: str, period: str) -> str:
    """Look up FRED (Federal Reserve Economic Data) observations for a
    series in a given period (e.g. GDP, UNRATE, CPIAUCSL) -- official U.S.
    government/Federal Reserve economic statistics.

    Args:
        series_id: the exact FRED series id, e.g. "GDP", "UNRATE".
        period: "YYYY" (calendar year), "YYYY-MM" (month), or "YYYY-MM-DD"
            (a single day) -- the observation window.

    Returns JSON. On success (`ok: true`) the payload carries a `url_or_id`
    field -- cite this EXACT string as record_claim's `source_id` argument --
    and a `data_ref` object (period = the LATEST observation's own date
    within the requested window); pass that `data_ref` dict VERBATIM as
    record_claim's `data_ref` argument. On failure (`ok: false`) do not cite
    this result as a source.
    """
    try:
        result = fred_client.fetch_series_observations(series_id, period)
    except fred_client.FredUnavailable as exc:
        return _fail(f"FRED unavailable: {exc}", series_id=series_id, period=period)
    except (fred_client.FredQueryError, ValueError) as exc:
        return _fail(str(exc), series_id=series_id, period=period)

    if result is None:
        return _fail("no observations returned for that series/period", series_id=series_id, period=period)

    observations = result["observations"]
    latest = observations[-1]
    url_or_id = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&observation_start={result['start']}&observation_end={result['end']}&file_type=json"
    payload = {
        "ok": True,
        "source_system": "fred",
        "source_class": FRED_SOURCE_CLASS,
        "url_or_id": url_or_id,
        "title": f"FRED {series_id} {period}",
        "series_id": series_id,
        "observations": observations,
        "latest_observation": latest,
        "data_ref": {"period": latest["date"], "source_class": FRED_SOURCE_CLASS},
    }
    return json.dumps(payload, default=str)


@tool
def courtlistener_search(query: str, court: str | None = None, filed_after: str | None = None, filed_before: str | None = None) -> str:
    """Search published US court opinions on CourtListener (case-law
    discovery, not citation verification -- the deterministic Eyecite
    citation gate is a separate, always-on check). Returns multiple
    candidate opinions; EACH result carries its own `url_or_id` -- cite the
    SPECIFIC opinion's `url_or_id` as record_claim's `source_id` when
    asserting a claim about that case, not this tool call as a whole.

    Args:
        query: free-text search (case name, legal issue, statute, etc).
        court: optional CourtListener court id filter, e.g. "scotus", "ca9".
        filed_after: optional ISO date (YYYY-MM-DD), inclusive lower bound
            on the opinion's filing date.
        filed_before: optional ISO date (YYYY-MM-DD), inclusive upper bound
            on the opinion's filing date.

    Returns JSON with a `results` list. No `data_ref` on this payload --
    court opinions are not vintage-dated numeric data points, so the
    structured-data provenance audit's period/source_class convention does
    not apply here; use gate_flags on the resulting claim (and the separate
    citation gate) for legal-source risk signals instead.
    """
    try:
        results = courtlistener_client.search_opinions(query, court=court, filed_after=filed_after, filed_before=filed_before)
    except courtlistener_client.CourtListenerUnavailable as exc:
        return _fail(f"CourtListener unavailable: {exc}", query=query)
    except courtlistener_client.CourtListenerQueryError as exc:
        return _fail(str(exc), query=query)

    if results is None:
        return _fail("no opinions found for that query", query=query)

    formatted = []
    for r in results:
        url_or_id = r.get("absolute_url") or f"courtlistener://search/{r.get('cluster_id')}"
        formatted.append(
            {
                "url_or_id": url_or_id,
                "title": r.get("case_name") or url_or_id,
                "court": r.get("court"),
                "date_filed": r.get("date_filed"),
                "citation": r.get("citation"),
                "snippet": r.get("snippet"),
            }
        )
    payload = {
        "ok": True,
        "source_system": "courtlistener",
        "query": query,
        "count": len(formatted),
        "results": formatted,
    }
    return json.dumps(payload, default=str)


class DrConnectorToolsMiddleware(AgentMiddleware):
    """Contributes Stage-C typed structured-connector tools (WRDS, EDGAR,
    FRED, CourtListener) to the bound tool set. No hooks -- the `tools`
    class attribute alone is the entire extension point (mirrors
    DrLedgerMiddleware.tools = [record_claim])."""

    tools = [wrds_query, edgar_company_facts, fred_series, courtlistener_search]


# Registered at import time, per tool_map.py's binding contract -- before any
# graph is built.
register_tool(WRDS_TOOL_NAME, "wrds")
register_tool(EDGAR_TOOL_NAME, "edgar")
register_tool(FRED_TOOL_NAME, "fred")
register_tool(COURTLISTENER_TOOL_NAME, "courtlistener")

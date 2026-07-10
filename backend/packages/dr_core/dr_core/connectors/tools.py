"""tools.py — Stage C: typed LangChain tools minted from the connector
registry (structured connectors first, WRDS anchor; batch 1 of
HANDOFF_CONNECTORS_BC.md's "C — structured tools").

Binding contract (tool_map.py's docstring): every tool this module mints gets
a name DISTINCT from web_search/web_fetch (community tools hardcode those)
and calls ``register_tool(tool_name, connector_name)`` at creation time
(module import), before any graph is built.

Tools are wired into the bound tool set the same zero-FORK_DELTA way
``DrLedgerMiddleware`` contributes ``record_claim``: an ``AgentMiddleware``
subclass with a ``tools`` class attribute, collected by the harness's agent
factory (see claim_tool.py's docstring). ``DrConnectorToolsMiddleware`` here
is that seam for Stage C.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import tool

from dr_core.connectors import wrds_client
from dr_core.connectors.tool_map import register_tool

WRDS_TOOL_NAME = "wrds_query"
WRDS_SOURCE_CLASS = "primary_database"


def _wrds_url_or_id(ticker: str, period: str) -> str:
    """Deterministic citation handle for a WRDS lookup -- not a real URL (WRDS
    is a database, not HTTP), but the same "the model already has this string
    from its own tool call" shape web_fetch's url provides, so record_claim's
    citation-resolution path (middleware._source_id over url_or_id) works
    unchanged for a non-HTTP structured source."""
    return f"wrds://crsp-compustat/{ticker.upper()}/{period}"


def _wrds_fail(error: str, **extra: Any) -> str:
    return json.dumps({"ok": False, "error": error, **extra})


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


class DrConnectorToolsMiddleware(AgentMiddleware):
    """Contributes Stage-C typed structured-connector tools (WRDS first) to
    the bound tool set. No hooks -- the `tools` class attribute alone is the
    entire extension point (mirrors DrLedgerMiddleware.tools = [record_claim])."""

    tools = [wrds_query]


# Registered at import time, per tool_map.py's binding contract -- before any
# graph is built.
register_tool(WRDS_TOOL_NAME, "wrds")

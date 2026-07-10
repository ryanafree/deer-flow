"""edgar_client.py — direct data.sec.gov XBRL companyconcept fetch (S9-C batch 2).

Endpoint and disambiguation rules mirror the already-ported
~/Documents/Projects/DeepResearch/harness/data_fetch.py::fetch_edgar (see
dr_core/fetch/structured.py's own port) as closely as the companyconcept (vs.
companyfacts) shape allows: the SEC ``fy`` field on an XBRL fact is the
FILING's fiscal year, not the data point's own reporting period, so period
disambiguation uses the fact's ``end`` date, never ``fy`` (a single 10-K
reports multiple prior fiscal years, all tagged with the filing's own fy).

Credentials: SEC_EDGAR_USER_AGENT (a descriptive UA string SEC requires;
connectors.yaml's edgar row and probe.py's probe_auth already send it for
reachability checks). Never printed or logged.

Mockable at the EDGAR_QUERY_BOUNDARY (``resolve_cik`` / ``fetch_company_concept``),
mirroring wrds_client.py's ``get_connection`` / ``fetch_crsp_price`` /
``fetch_compustat_fundamentals`` boundary -- each accepts an injectable
``transport`` callable so tests never hit the network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYCONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik10}/{taxonomy}/{concept}.json"
CONNECT_TIMEOUT_S = 30

_PERIOD_YEAR_RE = re.compile(r"^(\d{4})")

Transport = Callable[[str, dict[str, str]], Any]


class EdgarUnavailable(RuntimeError):
    """Raised when SEC_EDGAR_USER_AGENT is missing, or the HTTP request
    itself fails (network/DNS/timeout) -- distinct from a query returning no
    matching fact, which is a normal ``None`` result, not an error."""


class EdgarQueryError(RuntimeError):
    """Raised when an established request completes but the server rejects
    it (4xx/5xx) or returns unparseable JSON."""


def _user_agent() -> str:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    if not ua:
        raise EdgarUnavailable("SEC_EDGAR_USER_AGENT not set in the environment")
    return ua


def _default_transport(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise EdgarQueryError(f"EDGAR request failed: {exc.code} {url}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise EdgarQueryError(f"EDGAR request failed: {exc}") from exc


def _day_ordinal(s: str | None) -> int | None:
    if not s:
        return None
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d).toordinal()
    except Exception:
        return None


def resolve_cik(ticker_or_cik: str, transport: Transport | None = None) -> tuple[str, str | None]:
    """Ticker -> zero-padded 10-digit CIK + company name, via SEC's
    company_tickers.json (the same lookup fetch/structured.py's fetch_edgar
    uses). A purely numeric ``ticker_or_cik`` is treated as an already-resolved
    CIK (no network lookup, no company name available)."""
    cleaned = ticker_or_cik.strip()
    if cleaned.isdigit():
        return f"{int(cleaned):010d}", None
    transport = transport or _default_transport
    ua = _user_agent()
    headers = {"User-Agent": ua, "Accept": "application/json"}
    tickers = transport(COMPANY_TICKERS_URL, headers)
    target = cleaned.upper()
    for row in (tickers or {}).values():
        if isinstance(row, dict) and str(row.get("ticker", "")).upper() == target:
            return f"{int(row['cik_str']):010d}", row.get("title")
    raise EdgarQueryError(f"ticker {ticker_or_cik!r} not found in SEC company_tickers")


def fetch_company_concept(cik10: str, concept: str, period: str, *, taxonomy: str = "us-gaap", unit: str = "USD", transport: Transport | None = None) -> dict[str, Any] | None:
    """Companyconcept lookup for one XBRL tag, resolved to the single annual
    (10-K, full-year-span) fact whose ``end`` date falls in ``period``'s
    year -- disambiguating by end date, never by the filing-level ``fy``
    field (see module docstring). Returns ``None`` on no matching fact (not
    an error, mirrors wrds_client's ``fetch_crsp_price``/``fetch_compustat_fundamentals``
    contract)."""
    transport = transport or _default_transport
    ua = _user_agent()
    headers = {"User-Agent": ua, "Accept": "application/json"}
    url = COMPANYCONCEPT_URL.format(cik10=cik10, taxonomy=taxonomy, concept=concept)
    data = transport(url, headers)

    m = _PERIOD_YEAR_RE.match(period.strip())
    if not m:
        raise ValueError(f"period {period!r} is not a YYYY-leading period string")
    year = int(m.group(1))

    entries = (data.get("units") or {}).get(unit) or []
    matches: list[dict[str, Any]] = []
    for e in entries:
        form = e.get("form", "")
        if not form.startswith("10-K"):
            continue
        if e.get("fp") and e.get("fp") != "FY":
            continue
        end = e.get("end")
        if not end or not str(end).startswith(str(year)):
            continue
        start = e.get("start")
        start_ord, end_ord = _day_ordinal(start), _day_ordinal(end)
        if start_ord is not None and end_ord is not None and (end_ord - start_ord) < 300:
            continue  # quarterly/partial span, not the annual figure
        matches.append(e)
    if not matches:
        return None
    matches.sort(key=lambda e: e.get("filed") or "", reverse=True)
    best = matches[0]
    return {
        "cik10": cik10,
        "taxonomy": taxonomy,
        "concept": concept,
        "label": data.get("label"),
        "entity_name": data.get("entityName"),
        "unit": unit,
        "value": best.get("val"),
        "end": best.get("end"),
        "start": best.get("start"),
        "fy": best.get("fy"),
        "fp": best.get("fp"),
        "form": best.get("form"),
        "accn": best.get("accn"),
        "filed": best.get("filed"),
        "url": url,
    }

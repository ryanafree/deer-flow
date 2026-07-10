"""fred_client.py — direct FRED (Federal Reserve Economic Data) API fetch (S9-C batch 2).

Same request shape as the already-ported ~/Documents/Projects/DeepResearch/harness/
data_fetch.py::fetch_fred / dr_core/fetch/structured.py's port: FRED
series/observations, ``file_type=json``. The api_key is used only to build the
actual outbound request URL inside ``_default_transport`` -- every value this
module RETURNS (the observations, and the ``url_or_id`` the tool layer builds
from ``start``/``end``) is scrubbed of the key, matching
fetch/structured.py's ``query`` provenance-scrubbing convention.

Credentials: FRED_API_KEY. Never printed or logged.

Mockable at the FRED_QUERY_BOUNDARY (``fetch_series_observations``), same
injectable-``transport`` shape as edgar_client.py / wrds_client.py.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date
from typing import Any

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
CONNECT_TIMEOUT_S = 30

_PERIOD_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")

Transport = Callable[[str, dict[str, str]], Any]


class FredUnavailable(RuntimeError):
    """Missing FRED_API_KEY, or the HTTP request itself failed."""


class FredQueryError(RuntimeError):
    """An established request completed but FRED rejected it or returned
    unparseable JSON."""


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise FredUnavailable("FRED_API_KEY not set in the environment")
    return key


def _default_transport(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise FredQueryError(f"FRED request failed: {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise FredQueryError(f"FRED request failed: {exc}") from exc


def _period_to_date_range(period: str) -> tuple[str, str]:
    """ "YYYY" -> full calendar year; "YYYY-MM" -> that month; "YYYY-MM-DD" ->
    that single day. Raises ValueError on an unparseable period. (Mirrors
    wrds_client._period_to_date_range's year/month cases; FRED observations
    are daily-or-coarser so no +/-window is needed for a single day.)"""
    m = _PERIOD_RE.match(period.strip())
    if not m:
        raise ValueError(f"period {period!r} is not YYYY, YYYY-MM, or YYYY-MM-DD")
    year, month, day = m.groups()
    year_i = int(year)
    if day:
        return period, period
    if month:
        month_i = int(month)
        start = date(year_i, month_i, 1)
        end_year, end_month = (year_i, month_i + 1) if month_i < 12 else (year_i + 1, 1)
        end = date(end_year, end_month, 1).fromordinal(date(end_year, end_month, 1).toordinal() - 1)
        return start.isoformat(), end.isoformat()
    return date(year_i, 1, 1).isoformat(), date(year_i, 12, 31).isoformat()


def fetch_series_observations(series_id: str, period: str, *, transport: Transport | None = None) -> dict[str, Any] | None:
    """Observations for ``series_id`` within ``period``'s date window.
    Returns ``None`` when the series has no observations in that window (not
    an error)."""
    transport = transport or _default_transport
    key = _api_key()
    start, end = _period_to_date_range(period)
    url = f"{FRED_OBSERVATIONS_URL}?series_id={urllib.parse.quote(series_id)}&api_key={key}&file_type=json&observation_start={start}&observation_end={end}"
    data = transport(url, {})
    obs = [{"date": o["date"], "value": o["value"]} for o in (data.get("observations") or []) if o.get("value") not in (".", None, "")]
    if not obs:
        return None
    return {"series_id": series_id, "start": start, "end": end, "observations": obs}

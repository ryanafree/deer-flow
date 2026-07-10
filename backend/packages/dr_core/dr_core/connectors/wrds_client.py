"""wrds_client.py — direct psycopg2 connection to WRDS Postgres (S9-C, batch 1).

RULING (HANDOFF_CONNECTORS_BC.md open question 2, orchestrator seat, this
session): the `wrds` PyPI package was tried first for DRIF consistency, but
`uv add wrds` downgrades the workspace's `pandas` from 3.0.2 to 2.2.3 -- a
transitive pin conflict with markitdown's xlsx/Excel-conversion extra
(`packages/harness` file-upload document conversion), not a cosmetic one.
That is exactly the "drags conflicting dependencies into the backend uv env"
condition the ruling named as the fallback trigger, so this module talks to
WRDS directly over `psycopg2` instead -- confirmed additive-only in
`uv.lock` (43 added lines, no version churn elsewhere). No pandas import
anywhere in this module; rows come back as plain dicts.

Endpoint mirrors the registry row (`connectors.yaml`'s `wrds` entry) and the
same host:port `probe.py`'s SQL branch already TCP-probes:
wrds-pgdata.wharton.upenn.edu:9737/wrds.

Credentials: WRDS_USERNAME / WRDS_PASSWORD env vars, mirrored into
backend/.env per HANDOFF_CONNECTORS_BC.md's assumptions. Never printed,
never logged, never included in a returned record.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

import psycopg2
import psycopg2.extras

WRDS_HOST = "wrds-pgdata.wharton.upenn.edu"
WRDS_PORT = 9737
WRDS_DBNAME = "wrds"
CONNECT_TIMEOUT_S = 15

_PERIOD_YEAR_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


class WrdsUnavailable(RuntimeError):
    """Raised when WRDS credentials are missing or the connection attempt
    itself fails (network/auth). Distinct from a query returning no rows,
    which is a normal ``None`` result, not an error."""


class WrdsQueryError(RuntimeError):
    """Raised when a query against an established connection fails (bad SQL,
    server-side error) -- as opposed to WrdsUnavailable, which is a
    connection-establishment failure."""


def _credentials() -> tuple[str, str]:
    user = os.environ.get("WRDS_USERNAME", "").strip()
    password = os.environ.get("WRDS_PASSWORD", "").strip()
    if not user or not password:
        raise WrdsUnavailable("WRDS_USERNAME/WRDS_PASSWORD not set in the environment")
    return user, password


_CONN: Any = None


def get_connection():
    """Lazy, process-cached WRDS connection. Raises WrdsUnavailable (never a
    bare psycopg2 exception) on missing credentials or a failed connect."""
    global _CONN
    if _CONN is not None and not _CONN.closed:
        return _CONN
    user, password = _credentials()
    try:
        _CONN = psycopg2.connect(
            host=WRDS_HOST,
            port=WRDS_PORT,
            dbname=WRDS_DBNAME,
            user=user,
            password=password,
            sslmode="require",
            connect_timeout=CONNECT_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 -- normalize every failure mode to WrdsUnavailable
        raise WrdsUnavailable(f"WRDS connection failed: {exc}") from exc
    return _CONN


def reset_connection() -> None:
    """Test/ops hook: drop the cached connection so the next get_connection()
    call reconnects (or re-raises WrdsUnavailable) fresh."""
    global _CONN
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:  # noqa: BLE001 -- best-effort close, never raise on cleanup
            pass
    _CONN = None


def _period_to_date_range(period: str) -> tuple[str, str]:
    """ "YYYY" -> full calendar year; "YYYY-MM" -> that month; "YYYY-MM-DD" ->
    a +/-5 day window around that date (catches the nearest trading day either
    side of a non-trading date). Raises ValueError on an unparseable period."""
    m = _PERIOD_YEAR_RE.match(period.strip())
    if not m:
        raise ValueError(f"period {period!r} is not YYYY, YYYY-MM, or YYYY-MM-DD")
    year, month, day = m.groups()
    year_i = int(year)
    if day:
        d = date(year_i, int(month), int(day))
        start = d.fromordinal(d.toordinal() - 5)
        end = d.fromordinal(d.toordinal() + 5)
        return start.isoformat(), end.isoformat()
    if month:
        month_i = int(month)
        start = date(year_i, month_i, 1)
        end_year, end_month = (year_i, month_i + 1) if month_i < 12 else (year_i + 1, 1)
        end = date(end_year, end_month, 1).fromordinal(date(end_year, end_month, 1).toordinal() - 1)
        return start.isoformat(), end.isoformat()
    return date(year_i, 1, 1).isoformat(), date(year_i, 12, 31).isoformat()


def _period_to_fyear(period: str) -> int:
    m = _PERIOD_YEAR_RE.match(period.strip())
    if not m:
        raise ValueError(f"period {period!r} is not YYYY, YYYY-MM, or YYYY-MM-DD")
    return int(m.group(1))


def fetch_crsp_price(ticker: str, period: str, conn=None) -> dict[str, Any] | None:
    """Latest CRSP daily-file (crsp.dsf) price for `ticker` within `period`.
    Resolves ticker -> permno via crsp.stocknames' name-validity window
    (a.date BETWEEN b.namedt AND b.nameenddt), the standard CRSP ticker-
    resolution join. Returns None on no match (not an error)."""
    start, end = _period_to_date_range(period)
    conn = conn or get_connection()
    sql = """
        SELECT a.permno, a.date, a.prc, a.ret, a.vol, a.shrout, b.ticker, b.comnam
        FROM crsp.dsf a
        JOIN crsp.stocknames b
          ON a.permno = b.permno
         AND a.date BETWEEN b.namedt AND b.nameenddt
        WHERE b.ticker = %(ticker)s
          AND a.date BETWEEN %(start)s AND %(end)s
        ORDER BY a.date DESC
        LIMIT 1
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"ticker": ticker.upper(), "start": start, "end": end})
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 -- normalize to WrdsQueryError
        raise WrdsQueryError(f"CRSP query failed: {exc}") from exc
    return dict(row) if row else None


def fetch_compustat_fundamentals(ticker: str, period: str, conn=None) -> dict[str, Any] | None:
    """Standardized annual Compustat fundamentals (comp.funda) for `ticker` in
    fiscal year `period` (the leading 4 digits). Filtered to the standard
    INDL/STD/D/C combination (industrial format, standardized, domestic
    population, consolidated) -- the conventional "one row per company-year"
    Compustat annual slice. Returns None on no match (not an error)."""
    fyear = _period_to_fyear(period)
    conn = conn or get_connection()
    sql = """
        SELECT gvkey, tic, conm, datadate, fyear, at, revt, ni, sale
        FROM comp.funda
        WHERE tic = %(ticker)s
          AND fyear = %(fyear)s
          AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
        ORDER BY datadate DESC
        LIMIT 1
    """
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"ticker": ticker.upper(), "fyear": fyear})
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 -- normalize to WrdsQueryError
        raise WrdsQueryError(f"Compustat query failed: {exc}") from exc
    return dict(row) if row else None

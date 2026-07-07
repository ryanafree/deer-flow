"""dr_core.fetch.structured — typed-query structured connectors (EDGAR, FRED, CourtListener).

Ported from ~/Documents/Projects/DeepResearch/harness/data_fetch.py (Phase 1 port; see
build-logs/task-port-data-fetch.md and build-logs/phase1-port-inventory.md Target D).
Behavior preserved verbatim: same URLs, same fiscal-year disambiguation-by-end-date logic
for EDGAR (the SEC `fy` field is the FILING's fiscal year, not the data point's period),
same provenance scrubbing (the returned `query` string never contains an api_key).

Two ways to use this module:
- Importable typed functions (`fetch_edgar`, `fetch_fred`, `fetch_courtlistener`) return
  plain JSON-serializable dicts on success (`ok: True`) or failure (`ok: False, error: ...`)
  — no process exit, no printing. This is the shape a DeerFlow tool would call directly
  (wiring into the tool registry is Phase 3, out of scope here).
- `main()` is an argparse CLI parity wrapper matching the original script: one JSON object
  printed to stdout, always exits 0 (the caller reads `ok`, not the shell exit code).

Credentials are read from ~/.claude/.env via `env()` and never printed or included in any
returned record.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Any

try:
    import certifi

    _CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _CTX = ssl.create_default_context()

ENV_PATH = os.path.expanduser("~/.claude/.env")


def env(key: str) -> str:
    """Read a secret from ~/.claude/.env. Never prints or logs the value."""
    if not os.path.isfile(ENV_PATH):
        return ""
    for line in open(ENV_PATH):
        s = line.strip()
        if s.startswith(key + "="):
            v = s.split("=", 1)[1]
            v = re.split(r"\s#", v)[0].strip().strip('"').strip("'")
            return v
    return ""


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=30, context=_CTX) as r:
        return json.load(r)


def _post_form(url: str, fields: dict[str, str], headers: dict[str, str] | None = None) -> Any:
    body = urllib.parse.urlencode(fields).encode()
    h = {"Content-Type": "application/x-www-form-urlencoded"}
    h.update(headers or {})
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
        return json.load(r)


def _fail(msg: str, **extra: Any) -> dict[str, Any]:
    """Build an {"ok": False, "error": ...} record. Importable functions return it
    directly; the CLI wrapper prints it and exits 0 (see _print_and_exit)."""
    out: dict[str, Any] = {"ok": False, "error": msg}
    out.update(extra)
    return out


def _day_ordinal(s: str) -> int | None:
    try:
        y, m, d = map(int, s.split("-"))
        return date(y, m, d).toordinal()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# EDGAR
# ---------------------------------------------------------------------------
def fetch_edgar(
    ticker: str,
    concept: str | None = None,
    concept_keyword: str = "revenue",
    fy: int | None = None,
    unit: str = "USD",
    max_results: int = 8,
) -> dict[str, Any]:
    """SEC EDGAR XBRL companyfacts lookup: annual (10-K) facts for a ticker/concept/fy."""
    ua = env("SEC_EDGAR_USER_AGENT")
    if not ua:
        return _fail("SEC_EDGAR_USER_AGENT not set in ~/.claude/.env")
    hdr = {"User-Agent": ua, "Accept": "application/json"}

    # 1) ticker -> CIK
    try:
        tickers = _get_json("https://www.sec.gov/files/company_tickers.json", hdr)
    except urllib.error.HTTPError as e:
        return _fail(f"company_tickers fetch failed: {e.code}")
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"company_tickers fetch failed: {e}")
    cik = None
    name = None
    t = ticker.upper()
    for row in tickers.values():
        if row.get("ticker", "").upper() == t:
            cik = int(row["cik_str"])
            name = row.get("title")
            break
    if cik is None:
        return _fail(f"ticker {ticker} not found in SEC company_tickers")
    cik10 = f"{cik:010d}"

    # 2) companyfacts
    try:
        facts = _get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json", hdr)
    except urllib.error.HTTPError as e:
        return _fail(f"companyfacts fetch failed: {e.code}", cik=cik10)
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"companyfacts fetch failed: {e}", cik=cik10)

    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    kw = (concept_keyword or "").lower()
    matches: list[dict[str, Any]] = []
    for c, body in gaap.items():
        label = body.get("label") or ""
        if concept:
            if c != concept:
                continue
        elif kw and kw not in c.lower() and kw not in label.lower():
            continue
        for u, entries in (body.get("units") or {}).items():
            if unit and u != unit:
                continue
            for e in entries:
                # annual figure from a 10-K: full-year span and fiscal-period FY
                form = e.get("form", "")
                if not form.startswith("10-K"):
                    continue
                if e.get("fp") and e.get("fp") != "FY":
                    continue
                start, end = e.get("start"), e.get("end")
                if start and end:
                    start_ord, end_ord = _day_ordinal(start), _day_ordinal(end)
                    days = end_ord - start_ord if start_ord is not None and end_ord is not None else None
                    if days is not None and days < 300:  # skip quarterly/partial
                        continue
                # IMPORTANT: the SEC `fy` field is the FILING's fiscal year, not the data
                # point's period (a 10-K carries 3 years of income-statement data, all
                # tagged with the filing's fy). The data point's actual period is the END
                # date, so disambiguate fiscal year by end-date year. This is the period gate.
                end_year = int(end[:4]) if end else None
                if fy is not None and end_year != fy:
                    continue
                matches.append({
                    "concept": c, "label": label, "value": e.get("val"),
                    "unit": u, "fy": e.get("fy"), "fp": e.get("fp"), "form": form,
                    "start": start, "end": end, "accn": e.get("accn"), "filed": e.get("filed"),
                })
    # Dedupe identical period figures reported across multiple filings (keep newest filed).
    seen: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for m in sorted(matches, key=lambda x: x.get("filed") or "", reverse=True):
        k = (m["concept"], m["end"], m["value"])
        if k not in seen:
            seen[k] = m
    matches = list(seen.values())
    # Most relevant first: exact-keyword concepts, then larger magnitude (top-line first), then recency
    matches.sort(key=lambda m: (
        0 if kw and (kw == m["concept"].lower() or kw in (m["label"] or "").lower()) else 1,
        -(abs(m["value"]) if isinstance(m["value"], (int, float)) else 0),
        -(m["fy"] or 0),
    ))
    matches = matches[:max_results]
    if not matches:
        return _fail("no matching annual (10-K) facts for that ticker/concept/fy",
                      cik=cik10, company=name, ticker=t)
    query = (f"EDGAR XBRL companyfacts CIK{cik10} ({name}); "
             f"concept~='{concept or concept_keyword}'"
             + (f" fy={fy}" if fy is not None else " (latest annual)")
             + f" unit={unit}")
    return {
        "ok": True, "source_system": "edgar", "source_class": "primary_filing",
        "cik": cik10, "company": name, "ticker": t, "query": query,
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def fetch_fred(series: str, year: int | None = None, latest: bool = False) -> dict[str, Any]:
    """FRED series/observations lookup. `query` is scrubbed of the api_key."""
    key = env("FRED_API_KEY")
    if not key:
        return _fail("FRED_API_KEY not set in ~/.claude/.env")
    base = "https://api.stlouisfed.org/fred/series/observations"
    url = f"{base}?series_id={series}&api_key={key}&file_type=json"
    if year is not None:
        url += f"&observation_start={year}-01-01&observation_end={year}-12-31"
    try:
        data = _get_json(url)
    except urllib.error.HTTPError as e:
        return _fail(f"FRED fetch failed: {e.code}", series=series)
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"FRED fetch failed: {e}", series=series)
    obs = [{"date": o["date"], "value": o["value"]} for o in data.get("observations", [])
           if o.get("value") not in (".", None, "")]
    if not obs:
        return _fail("no observations returned", series=series)
    if latest and year is None:
        obs = obs[-1:]
    # scrubbed provenance (no api_key)
    query = (f"FRED series/observations series_id={series} file_type=json"
             + (f" observation window {year}" if year is not None else " (latest)"))
    return {
        "ok": True, "source_system": "fred", "source_class": "official_stat",
        "series_id": series, "query": query, "observations": obs,
    }


# ---------------------------------------------------------------------------
# CourtListener — Eyecite citation gate (the Legal Tier 2 deterministic check)
# ---------------------------------------------------------------------------
def fetch_courtlistener(text: str | None = None, textfile: str | None = None) -> dict[str, Any]:
    """Eyecite citation-lookup over submitted text via CourtListener's API."""
    tok = env("COURTLISTENER_TOKEN")
    if not tok:
        return _fail("COURTLISTENER_TOKEN not set in ~/.claude/.env")
    if textfile:
        with open(textfile) as f:
            text = f.read()
    if not text:
        return _fail("no --text or --textfile supplied")
    hdr = {"Authorization": f"Token {tok}"}
    url = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"
    try:
        res = _post_form(url, {"text": text[:64000]}, hdr)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return _fail("rate-limited (429): CourtListener daily/throttle cap reached", rate_limited=True)
        return _fail(f"citation-lookup failed: {e.code}")
    except (urllib.error.URLError, OSError) as e:
        return _fail(f"citation-lookup failed: {e}")
    # Normalize each returned citation to {citation, status, case_name, url, clusters, indices}.
    cites = []
    for c in (res if isinstance(res, list) else []):
        clusters = c.get("clusters") or []
        cites.append({
            "citation": c.get("citation"),
            "normalized": c.get("normalized_citations") or [],
            "status": c.get("status"),
            "error_message": c.get("error_message") or "",
            "clusters_count": len(clusters),
            "case_name": (clusters[0].get("case_name") if clusters else None),
            "absolute_url": (("https://www.courtlistener.com" + clusters[0]["absolute_url"]) if clusters and clusters[0].get("absolute_url") else None),
            "start_index": c.get("start_index"),
            "end_index": c.get("end_index"),
        })
    return {
        "ok": True, "source_system": "courtlistener",
        "query": "citation-lookup (Eyecite) over submitted text",
        "cites": cites,
    }


# ---------------------------------------------------------------------------
# CLI parity wrapper
# ---------------------------------------------------------------------------
def _print_and_exit(record: dict[str, Any]) -> None:
    print(json.dumps(record, indent=2))
    sys.exit(0)  # exit 0: the caller reads ok=false, not a shell error


def _cli_edgar(args: argparse.Namespace) -> None:
    _print_and_exit(fetch_edgar(
        ticker=args.ticker, concept=args.concept, concept_keyword=args.concept_keyword,
        fy=args.fy, unit=args.unit, max_results=args.max,
    ))


def _cli_fred(args: argparse.Namespace) -> None:
    _print_and_exit(fetch_fred(series=args.series, year=args.year, latest=args.latest))


def _cli_courtlistener(args: argparse.Namespace) -> None:
    _print_and_exit(fetch_courtlistener(text=args.text, textfile=args.textfile))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("edgar")
    pe.add_argument("--ticker", required=True)
    pe.add_argument("--concept", default=None, help="exact us-gaap concept name")
    pe.add_argument("--concept-keyword", default="revenue", help="substring match on concept/label")
    pe.add_argument("--fy", type=int, default=None, help="fiscal year; omit for latest annual")
    pe.add_argument("--unit", default="USD")
    pe.add_argument("--max", type=int, default=8)
    pe.set_defaults(func=_cli_edgar)

    pf = sub.add_parser("fred")
    pf.add_argument("--series", required=True)
    pf.add_argument("--year", type=int, default=None)
    pf.add_argument("--latest", action="store_true")
    pf.set_defaults(func=_cli_fred)

    pc = sub.add_parser("courtlistener")
    pc.add_argument("--text", default=None, help="text to run Eyecite citation-lookup over")
    pc.add_argument("--textfile", default=None, help="file with text to check")
    pc.set_defaults(func=_cli_courtlistener)

    return ap


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

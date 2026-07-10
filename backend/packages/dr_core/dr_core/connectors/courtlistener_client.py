"""courtlistener_client.py — CourtListener v4 opinion search (S9-C batch 2).

Distinct from ``dr_core/verify/citation.py``'s citation-lookup client (the D8
Eyecite gate, a different CourtListener endpoint, POST citation-lookup over
submitted text) -- this is the Legal profile's DISCOVERY tool: free-text
search over published court opinions (``/search/?type=o``), not a
citation-resolution check. Same auth convention as citation.py
(``Authorization: Token <COURTLISTENER_TOKEN>``), REST v4 substrate endpoint
(connectors.yaml's ``courtlistener_rest`` row).

Credentials: COURTLISTENER_TOKEN (COURTLISTENER_API_TOKEN accepted as a
fallback spelling, matching citation.py's convention). Never printed or
logged.

Mockable at the COURTLISTENER_QUERY_BOUNDARY (``search_opinions``), same
injectable-``transport`` shape as edgar_client.py / fred_client.py /
wrds_client.py.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
CONNECT_TIMEOUT_S = 30

Transport = Callable[[str, dict[str, str]], Any]


class CourtListenerUnavailable(RuntimeError):
    """Missing COURTLISTENER_TOKEN, or the HTTP request itself failed."""


class CourtListenerQueryError(RuntimeError):
    """An established request completed but CourtListener rejected it or
    returned unparseable JSON."""


def _token() -> str:
    tok = (os.environ.get("COURTLISTENER_TOKEN") or os.environ.get("COURTLISTENER_API_TOKEN") or "").strip()
    if not tok:
        raise CourtListenerUnavailable("COURTLISTENER_TOKEN not set in the environment")
    return tok


def _default_transport(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise CourtListenerQueryError("rate-limited (429): CourtListener daily/throttle cap reached") from exc
        raise CourtListenerQueryError(f"CourtListener request failed: {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CourtListenerQueryError(f"CourtListener request failed: {exc}") from exc


def search_opinions(
    query: str,
    *,
    court: str | None = None,
    filed_after: str | None = None,
    filed_before: str | None = None,
    max_results: int = 10,
    transport: Transport | None = None,
) -> list[dict[str, Any]] | None:
    """Free-text opinion search (``type=o``). Returns ``None`` on zero
    results (not an error)."""
    transport = transport or _default_transport
    token = _token()
    params: dict[str, str] = {"q": query, "type": "o"}
    if court:
        params["court"] = court
    if filed_after:
        params["filed_after"] = filed_after
    if filed_before:
        params["filed_before"] = filed_before
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Token {token}"}
    data = transport(url, headers)
    raw_results = (data or {}).get("results") or []

    results: list[dict[str, Any]] = []
    for r in raw_results[:max_results]:
        if not isinstance(r, dict):
            continue
        absolute_url = r.get("absolute_url")
        if absolute_url and str(absolute_url).startswith("/"):
            absolute_url = "https://www.courtlistener.com" + absolute_url
        results.append(
            {
                "case_name": r.get("caseName") or r.get("case_name"),
                "court": r.get("court") or r.get("court_id"),
                "date_filed": r.get("dateFiled") or r.get("date_filed"),
                "citation": r.get("citation"),
                "cluster_id": r.get("cluster_id"),
                "snippet": r.get("snippet") or r.get("text"),
                "absolute_url": absolute_url,
            }
        )
    return results or None

"""semantic_scholar_client.py — direct api.semanticscholar.org paper search
(2026-07-11, MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery").

Why this exists: the semantic_scholar MCP server (semantic-scholar-mcp,
launched via extensions_config.json) is a third-party package we cannot
patch. Live benchmark runs (build-logs/bench-b1-run1.log:620) showed seven
concurrent `search_papers` calls immediately hitting HTTP 429, retried
through what the review calls "a closed client" -- a client whose retry/
backoff behavior is opaque and unfixable from this repo. This module is the
"open" replacement referenced by the review: a small, auditable, hermetically
testable client that (a) serializes every outbound request through a single
process-wide lock so at most one Semantic Scholar request is ever in flight
from our own code, and (b) honors a 429 response's `Retry-After` header with
a bounded number of retries instead of hammering the endpoint.

Same injectable-``transport`` shape as courtlistener_client.py / edgar_client.py
/ fred_client.py / wrds_client.py -- tests never hit the network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from threading import Lock
from typing import Any

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
CONNECT_TIMEOUT_S = 30
FIELDS = "paperId,title,abstract,year,authors,venue,citationCount,externalIds"

# Concurrency cap: at most one Semantic Scholar request in flight at a time
# from this process, held for the duration of the request AND any
# retry-after backoff -- this is what actually prevents the seven-way
# concurrent 429 storm the review observed, not just a per-request setting.
_REQUEST_LOCK = Lock()

MAX_RETRIES = 3
DEFAULT_RETRY_AFTER_S = 2.0
MAX_RETRY_AFTER_S = 30.0

Transport = Callable[[str, dict[str, str]], Any]
Sleeper = Callable[[float], None]


class SemanticScholarUnavailable(RuntimeError):
    """The HTTP request itself failed (network/DNS/timeout)."""


class SemanticScholarQueryError(RuntimeError):
    """An established request completed but Semantic Scholar rejected it,
    returned unparseable JSON, or 429'd through every retry."""


class SemanticScholarRateLimited(RuntimeError):
    """Internal signal raised by a transport on HTTP 429; carries the
    bounded retry-after delay in seconds. Never escapes ``search_papers`` --
    the retry loop catches it."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate-limited, retry after {retry_after}s")
        self.retry_after = retry_after


def _parse_retry_after(value: str | None) -> float:
    if not value:
        return DEFAULT_RETRY_AFTER_S
    try:
        seconds = float(value)
    except ValueError:
        return DEFAULT_RETRY_AFTER_S
    return max(0.0, min(seconds, MAX_RETRY_AFTER_S))


def _default_transport(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            retry_after_hdr = exc.headers.get("Retry-After") if exc.headers else None
            raise SemanticScholarRateLimited(_parse_retry_after(retry_after_hdr)) from exc
        raise SemanticScholarQueryError(f"Semantic Scholar request failed: {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SemanticScholarUnavailable(f"Semantic Scholar request failed: {exc}") from exc


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _normalize(paper: dict[str, Any]) -> dict[str, Any]:
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    paper_id = paper.get("paperId")
    url_or_id = f"https://doi.org/{doi}" if doi else (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)
    authors = [a.get("name") for a in (paper.get("authors") or []) if isinstance(a, dict) and a.get("name")]
    return {
        "url_or_id": url_or_id,
        "title": paper.get("title"),
        "year": paper.get("year"),
        "authors": authors,
        "venue": paper.get("venue"),
        "snippet": paper.get("abstract"),
        "citation_count": paper.get("citationCount"),
    }


def search_papers(
    query: str,
    *,
    limit: int = 5,
    transport: Transport | None = None,
    sleep: Sleeper | None = None,
) -> list[dict[str, Any]] | None:
    """Search Semantic Scholar for papers matching ``query``. Serializes
    against every other in-process caller (module-level lock) and retries a
    429 up to ``MAX_RETRIES`` times honoring ``Retry-After`` (capped at
    ``MAX_RETRY_AFTER_S``). Returns ``None`` on zero results (not an error).
    Raises ``SemanticScholarQueryError`` if every retry is exhausted or the
    server rejects the request outright, ``SemanticScholarUnavailable`` on a
    network-level failure."""
    transport = transport or _default_transport
    sleep = sleep or time.sleep
    params = {"query": query, "fields": FIELDS, "limit": str(limit)}
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    headers = _headers()

    with _REQUEST_LOCK:
        attempt = 0
        while True:
            attempt += 1
            try:
                data = transport(url, headers)
                break
            except SemanticScholarRateLimited as exc:
                if attempt > MAX_RETRIES:
                    raise SemanticScholarQueryError(f"rate-limited (429) after {MAX_RETRIES} retries") from exc
                sleep(exc.retry_after)

    raw_results = (data or {}).get("data") or []
    results = [_normalize(p) for p in raw_results[:limit] if isinstance(p, dict)]
    return results or None

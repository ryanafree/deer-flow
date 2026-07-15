"""openalex_client.py — direct api.openalex.org works search (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": OpenAlex as an
explicit academic-search fallback when Semantic Scholar is unavailable or
exhausts its retries).

connectors.yaml already declares openalex <-> semantic_scholar as mutual
fallbacks and notes a free key has been required since 2026-02-24 -- this
client sends it when present but degrades to a keyless (slower) request
otherwise, matching the registry row's own contract.

Same injectable-``transport`` shape as the other connector clients in this
package; tests never hit the network.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

SEARCH_URL = "https://api.openalex.org/works"
CONNECT_TIMEOUT_S = 30

Transport = Callable[[str, dict[str, str]], Any]


class OpenAlexUnavailable(RuntimeError):
    """The HTTP request itself failed (network/DNS/timeout)."""


class OpenAlexQueryError(RuntimeError):
    """An established request completed but OpenAlex rejected it (including
    429) or returned unparseable JSON."""


def _default_transport(url: str, headers: dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise OpenAlexQueryError(f"OpenAlex request failed: {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise OpenAlexUnavailable(f"OpenAlex request failed: {exc}") from exc


def _normalize(work: dict[str, Any]) -> dict[str, Any]:
    doi = work.get("doi")
    url_or_id = doi or work.get("id")
    authorships = work.get("authorships") or []
    authors = [(a.get("author") or {}).get("display_name") for a in authorships if isinstance(a, dict) and (a.get("author") or {}).get("display_name")]
    primary_location = work.get("primary_location") or {}
    venue = (primary_location.get("source") or {}).get("display_name")
    return {
        "url_or_id": url_or_id,
        "title": work.get("display_name") or work.get("title"),
        "year": work.get("publication_year"),
        "authors": authors,
        "venue": venue,
        "snippet": None,  # OpenAlex returns abstract_inverted_index, not plain text; not reconstructed here
        "citation_count": work.get("cited_by_count"),
    }


def search_works(query: str, *, limit: int = 5, transport: Transport | None = None) -> list[dict[str, Any]] | None:
    """Search OpenAlex works matching ``query``. Returns ``None`` on zero
    results (not an error)."""
    transport = transport or _default_transport
    params = {"search": query, "per-page": str(limit)}
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"
    headers = {"Accept": "application/json"}

    data = transport(url, headers)
    raw_results = (data or {}).get("results") or []
    results = [_normalize(w) for w in raw_results[:limit] if isinstance(w, dict)]
    return results or None

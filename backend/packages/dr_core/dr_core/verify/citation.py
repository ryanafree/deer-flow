"""Legal citation gate (D8): CITE_RE port + a CourtListener citation-lookup client.

Ported from ~/Documents/Projects/DeepResearch/harness/dr.js:2363-2430 (the Tier 2
deterministic legal check) and data_fetch.py's ``courtlistener`` subcommand (the exact
request shape: POST form-encoded ``text`` to the v4 citation-lookup endpoint, bearer
``Token`` auth). The decision function (``decide_citation_status``) is pure and
network-free so it is unit-testable against a stubbed cite list; ``lookup_citations``
is the only network-touching piece, always run off the event loop via
``asyncio.to_thread``, with a hard timeout and no-raise contract: any failure --
missing token, timeout, malformed response -- returns ``None`` (no advance, citation
stays UNRESOLVED), never an exception.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from dr_core.models.enums import CitationStatus

# Permissive volume-reporter-page detector, ported verbatim from dr.js:2370.
CITE_RE = re.compile(r"\b\d{1,4}\s+[A-Z][A-Za-z.]*\.?(?:\s?\d?d)?\.?\s+\d{1,5}\b")

_COURTLISTENER_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"
_DEFAULT_TIMEOUT = 15.0
_MAX_TEXT_CHARS = 64000  # matches data_fetch.py's text[:64000] truncation


def extract_citations(text: str) -> list[str]:
    """Every citation-shaped substring in ``text``. dr.js's CITE_RE is a single
    permissive detector with no per-cite splitting logic beyond regex matches, so
    this is exactly ``CITE_RE.findall``."""
    return CITE_RE.findall(text or "")


def has_citation(text: str) -> bool:
    return bool(CITE_RE.search(text or ""))


def decide_citation_status(cites: list[dict] | None) -> CitationStatus | None:
    """Pure decision over a CourtListener citation-lookup result (or ``None`` on
    lookup failure). Per D8: per-cite 404/400 -> NOT_FOUND (terminal); found (200,
    no 404/400) -> RESOLVED; ambiguous (300) with no 404/400 -> stays UNRESOLVED
    (``None``); no cites / lookup failure -> stays UNRESOLVED (``None``). A claim
    with multiple cites where any is 404/400 is NOT_FOUND overall, mirroring
    dr.js's ``notFound.length ? 'not-found' : (ambiguous.length ? 'ambiguous' :
    'found')`` precedence (dr.js:2412)."""
    if not cites:
        return None
    statuses = [c.get("status") for c in cites]
    if any(s in (404, 400) for s in statuses):
        return CitationStatus.NOT_FOUND
    if any(s == 300 for s in statuses):
        return None
    if any(s == 200 for s in statuses):
        return CitationStatus.RESOLVED
    return None


def _lookup_citations_sync(text: str, token: str | None, timeout: float) -> list[dict] | None:
    """Blocking CourtListener call -- only ever invoked via ``asyncio.to_thread``
    from ``lookup_citations``. Never raises: every failure path returns ``None``."""
    if not token:
        return None
    try:
        data = urllib.parse.urlencode({"text": text[:_MAX_TEXT_CHARS]}).encode()
        req = urllib.request.Request(
            _COURTLISTENER_URL,
            data=data,
            headers={"Authorization": f"Token {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    cites: list[dict] = []
    for entry in payload:
        if isinstance(entry, dict):
            cites.append({"citation": entry.get("citation"), "status": entry.get("status")})
    return cites


async def lookup_citations(text: str, *, token: str | None = None, timeout: float = _DEFAULT_TIMEOUT) -> list[dict] | None:
    """Async, non-blocking CourtListener citation-lookup. ``token`` defaults to
    ``COURTLISTENER_API_TOKEN``; any failure (missing token, network error,
    timeout, malformed response) degrades to ``None`` rather than raising, per D8
    decision 6 (a hung source never stalls a run)."""
    resolved_token = token if token is not None else os.environ.get("COURTLISTENER_API_TOKEN")
    try:
        return await asyncio.to_thread(_lookup_citations_sync, text, resolved_token, timeout)
    except Exception:
        return None

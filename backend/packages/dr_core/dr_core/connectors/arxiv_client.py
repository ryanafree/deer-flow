"""arxiv_client.py — direct export.arxiv.org Atom search (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": arXiv as the
final explicit academic-search fallback, behind Semantic Scholar and
OpenAlex).

arXiv's API is keyless Atom XML (no JSON option); parsed with stdlib
``xml.etree.ElementTree`` -- no new dependency. connectors.yaml's arxiv row
asks callers to "be polite (1 req/3s)"; this client makes one request per
call and leaves call-site pacing to the caller (the academic_search fallback
chain only reaches arxiv after semantic_scholar and openalex have already
failed, so back-to-back arxiv calls are not the expected hot path).

Same injectable-``transport`` shape as the other connector clients in this
package, except the transport returns raw bytes (Atom XML) rather than a
parsed dict; tests never hit the network.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree

SEARCH_URL = "http://export.arxiv.org/api/query"
CONNECT_TIMEOUT_S = 30
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

Transport = Callable[[str], bytes]


class ArxivUnavailable(RuntimeError):
    """The HTTP request itself failed (network/DNS/timeout)."""


class ArxivQueryError(RuntimeError):
    """An established request completed but arXiv rejected it (including
    429) or returned unparseable XML."""


def _default_transport(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/atom+xml"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ArxivQueryError(f"arXiv request failed: {exc.code}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ArxivUnavailable(f"arXiv request failed: {exc}") from exc


def _collapse_whitespace(s: str) -> str:
    """arXiv Atom titles/summaries routinely wrap across lines with leading
    indentation in the raw XML (e.g. "Foo\n  Bar") -- a bare newline->space
    replace leaves the indentation's extra spaces behind. split()/join()
    collapses any run of whitespace (including newlines) to one space."""
    return " ".join(s.split())


def _normalize(entry: ElementTree.Element) -> dict[str, Any]:
    def text(tag: str) -> str | None:
        el = entry.find(f"atom:{tag}", _ATOM_NS)
        return el.text.strip() if el is not None and el.text else None

    authors = [a.findtext("atom:name", namespaces=_ATOM_NS) for a in entry.findall("atom:author", _ATOM_NS)]
    authors = [a.strip() for a in authors if a]
    published = text("published")
    year = int(published[:4]) if published and published[:4].isdigit() else None
    summary = text("summary")
    return {
        "url_or_id": text("id"),
        "title": _collapse_whitespace(text("title") or "") or None,
        "year": year,
        "authors": authors,
        "venue": "arXiv",
        "snippet": _collapse_whitespace(summary) if summary else None,
        "citation_count": None,
    }


def search_papers(query: str, *, limit: int = 5, transport: Transport | None = None) -> list[dict[str, Any]] | None:
    """Search arXiv for papers matching ``query`` (title/abstract full-text
    search). Returns ``None`` on zero results (not an error)."""
    transport = transport or _default_transport
    params = {"search_query": f"all:{query}", "start": "0", "max_results": str(limit)}
    url = f"{SEARCH_URL}?{urllib.parse.urlencode(params)}"

    raw = transport(url)
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ArxivQueryError(f"arXiv returned unparseable Atom XML: {exc}") from exc

    entries = root.findall("atom:entry", _ATOM_NS)
    results = [_normalize(e) for e in entries[:limit]]
    return results or None

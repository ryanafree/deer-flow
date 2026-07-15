"""Tests for dr_core.connectors.arxiv_client (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": arXiv as the
final academic-search fallback). HERMETIC: transport is injected and returns
raw Atom XML bytes, matching export.arxiv.org's real response shape."""

from __future__ import annotations

import pytest
from dr_core.connectors import arxiv_client as ac

_ATOM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
{entries}
</feed>
"""

_ENTRY_TEMPLATE = """
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <title>{title}</title>
    <summary>{summary}</summary>
    <published>{published}</published>
    <author><name>{author}</name></author>
  </entry>
"""


def _feed(entries: list[dict]) -> bytes:
    rendered = "".join(
        _ENTRY_TEMPLATE.format(
            arxiv_id=e.get("arxiv_id", "2401.00001"),
            title=e.get("title", "A Paper"),
            summary=e.get("summary", "An abstract."),
            published=e.get("published", "2024-01-15T00:00:00Z"),
            author=e.get("author", "A. Author"),
        )
        for e in entries
    )
    return _ATOM_TEMPLATE.format(entries=rendered).encode("utf-8")


class TestSearchPapers:
    def test_success_normalizes_results(self):
        def transport(url):
            assert "search_query=all%3Avolatility" in url
            return _feed([{"arxiv_id": "2401.00001", "title": "Volatility Study\n  Continued", "summary": "We find...", "published": "2024-03-10T00:00:00Z", "author": "A. Researcher"}])

        results = ac.search_papers("volatility", transport=transport)
        assert len(results) == 1
        r = results[0]
        assert r["url_or_id"] == "http://arxiv.org/abs/2401.00001"
        assert r["title"] == "Volatility Study Continued"
        assert r["year"] == 2024
        assert r["authors"] == ["A. Researcher"]
        assert r["venue"] == "arXiv"
        assert r["snippet"] == "We find..."

    def test_no_results_returns_none(self):
        transport = lambda url: _feed([])  # noqa: E731
        assert ac.search_papers("nothing", transport=transport) is None

    def test_limit_truncates(self):
        entries = [{"arxiv_id": f"240{i}.0000{i}", "title": f"P{i}"} for i in range(5)]
        transport = lambda url: _feed(entries)  # noqa: E731
        results = ac.search_papers("q", limit=2, transport=transport)
        assert len(results) == 2

    def test_unparseable_xml_raises_query_error(self):
        transport = lambda url: b"not xml at all <<<"  # noqa: E731
        with pytest.raises(ac.ArxivQueryError):
            ac.search_papers("q", transport=transport)

    def test_multiple_authors_collected(self):
        def transport(url):
            xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00002</id>
    <title>Multi Author Paper</title>
    <summary>S</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>A. One</name></author>
    <author><name>B. Two</name></author>
  </entry>
</feed>
"""
            return xml.encode("utf-8")

        results = ac.search_papers("q", transport=transport)
        assert results[0]["authors"] == ["A. One", "B. Two"]

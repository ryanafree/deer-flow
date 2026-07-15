"""Tests for dr_core.connectors.openalex_client (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": OpenAlex as an
explicit academic-search fallback). HERMETIC: transport is injected."""

from __future__ import annotations

import pytest
from dr_core.connectors import openalex_client as oc


class TestSearchWorks:
    def test_success_normalizes_results(self):
        def transport(url, headers):
            assert "search=" in url
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "doi": "https://doi.org/10.1000/abc",
                        "display_name": "A Study of Beta and IV",
                        "publication_year": 2023,
                        "cited_by_count": 5,
                        "authorships": [{"author": {"display_name": "J. Smith"}}],
                        "primary_location": {"source": {"display_name": "Journal of Vol"}},
                    }
                ]
            }

        results = oc.search_works("beta iv correlation", transport=transport)
        assert len(results) == 1
        r = results[0]
        assert r["url_or_id"] == "https://doi.org/10.1000/abc"
        assert r["title"] == "A Study of Beta and IV"
        assert r["year"] == 2023
        assert r["authors"] == ["J. Smith"]
        assert r["venue"] == "Journal of Vol"
        assert r["citation_count"] == 5

    def test_falls_back_to_openalex_id_when_no_doi(self):
        def transport(url, headers):
            return {"results": [{"id": "https://openalex.org/W1", "display_name": "T"}]}

        results = oc.search_works("q", transport=transport)
        assert results[0]["url_or_id"] == "https://openalex.org/W1"

    def test_no_results_returns_none(self):
        transport = lambda url, headers: {"results": []}  # noqa: E731
        assert oc.search_works("nothing", transport=transport) is None

    def test_limit_truncates(self):
        transport = lambda url, headers: {"results": [{"id": str(i), "display_name": f"P{i}"} for i in range(10)]}  # noqa: E731
        results = oc.search_works("q", limit=2, transport=transport)
        assert len(results) == 2

    def test_api_key_included_in_params_when_present(self, monkeypatch):
        monkeypatch.setenv("OPENALEX_API_KEY", "key123")
        captured = {}

        def transport(url, headers):
            captured["url"] = url
            return {"results": []}

        oc.search_works("q", transport=transport)
        assert "api_key=key123" in captured["url"]

    def test_no_api_key_omitted_from_params(self, monkeypatch):
        monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
        captured = {}

        def transport(url, headers):
            captured["url"] = url
            return {"results": []}

        oc.search_works("q", transport=transport)
        assert "api_key=" not in captured["url"]

    def test_http_error_raises_query_error(self):
        def transport(url, headers):
            raise oc.OpenAlexQueryError("429")

        with pytest.raises(oc.OpenAlexQueryError):
            oc.search_works("q", transport=transport)

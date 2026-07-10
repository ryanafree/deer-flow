"""Tests for dr_core.connectors.courtlistener_client (S9-C batch 2). All HTTP
calls are stubbed via the injectable `transport` param -- no real network in
this file."""

from __future__ import annotations

import pytest
from dr_core.connectors import courtlistener_client


@pytest.fixture(autouse=True)
def _token_env(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_TOKEN", "test-token")
    yield


class TestSearchOpinions:
    def test_success_normalizes_results_and_absolute_url(self):
        def transport(url, headers):
            assert headers["Authorization"] == "Token test-token"
            assert "type=o" in url
            return {
                "results": [
                    {"caseName": "Harlow v. Fitzgerald", "court": "scotus", "dateFiled": "1982-06-24", "citation": ["457 U.S. 800"], "cluster_id": 111, "absolute_url": "/opinion/111/harlow-v-fitzgerald/"}
                ]
            }

        results = courtlistener_client.search_opinions("qualified immunity", transport=transport)
        assert len(results) == 1
        r = results[0]
        assert r["case_name"] == "Harlow v. Fitzgerald"
        assert r["absolute_url"] == "https://www.courtlistener.com/opinion/111/harlow-v-fitzgerald/"

    def test_query_params_include_optional_filters(self):
        captured = {}

        def transport(url, headers):
            captured["url"] = url
            return {"results": []}

        courtlistener_client.search_opinions("x", court="scotus", filed_after="1980-01-01", filed_before="1990-01-01", transport=transport)
        assert "court=scotus" in captured["url"]
        assert "filed_after=1980-01-01" in captured["url"]
        assert "filed_before=1990-01-01" in captured["url"]

    def test_no_results_returns_none(self):
        transport = lambda url, headers: {"results": []}  # noqa: E731
        assert courtlistener_client.search_opinions("nonexistent case xyz", transport=transport) is None

    def test_max_results_truncates(self):
        transport = lambda url, headers: {"results": [{"caseName": f"Case {i}"} for i in range(5)]}  # noqa: E731
        results = courtlistener_client.search_opinions("x", max_results=2, transport=transport)
        assert len(results) == 2

    def test_missing_token_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("COURTLISTENER_TOKEN", raising=False)
        monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)
        with pytest.raises(courtlistener_client.CourtListenerUnavailable):
            courtlistener_client.search_opinions("x")

    def test_fallback_token_env_var_accepted(self, monkeypatch):
        monkeypatch.delenv("COURTLISTENER_TOKEN", raising=False)
        monkeypatch.setenv("COURTLISTENER_API_TOKEN", "fallback-token")

        def transport(url, headers):
            assert headers["Authorization"] == "Token fallback-token"
            return {"results": []}

        courtlistener_client.search_opinions("x", transport=transport)

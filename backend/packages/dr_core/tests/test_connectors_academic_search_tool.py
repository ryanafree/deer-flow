"""Tests for dr_core.connectors.tools.academic_search (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": one tool with a
semantic_scholar -> openalex -> arxiv fallback chain).

Kept in its own file (rather than appended to test_connectors_tools.py)
since that file is being actively edited by a concurrent thread landing the
ToolBindingRegistry work -- this file only touches
dr_core.connectors.{tools,semantic_scholar_client,openalex_client,arxiv_client},
none of which that thread owns.

HERMETIC: no live network. Each client module's own search function is
monkeypatched; this drives the REAL academic_search tool function and its
REAL fallback/registration logic."""

from __future__ import annotations

import json

from dr_core.connectors import arxiv_client, openalex_client, semantic_scholar_client
from dr_core.connectors import tools as tools_mod
from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS

_SS_RESULT = [{"url_or_id": "https://doi.org/10.1/ss", "title": "SS Paper", "year": 2024, "authors": ["A"], "venue": "V", "snippet": "s", "citation_count": 1}]
_OA_RESULT = [{"url_or_id": "https://openalex.org/W1", "title": "OA Paper", "year": 2023, "authors": ["B"], "venue": "V2", "snippet": None, "citation_count": 2}]
_ARXIV_RESULT = [{"url_or_id": "http://arxiv.org/abs/1", "title": "Arxiv Paper", "year": 2022, "authors": ["C"], "venue": "arXiv", "snippet": "s2", "citation_count": None}]


class TestToolRegistration:
    def test_registered_to_all_three_fallback_connectors(self):
        assert TOOL_TO_CONNECTORS["academic_search"] == frozenset({"semantic_scholar", "openalex", "arxiv"})

    def test_tool_name_distinct_from_community_hardcoded_names(self):
        assert tools_mod.ACADEMIC_SEARCH_TOOL_NAME not in {"web_search", "web_fetch"}

    def test_contributed_by_connector_tools_middleware(self):
        assert tools_mod.academic_search in tools_mod.DrConnectorToolsMiddleware.tools


class TestFallbackChain:
    def test_semantic_scholar_success_short_circuits_fallback(self, monkeypatch):
        called = {"openalex": False, "arxiv": False}
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: _SS_RESULT)
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: called.__setitem__("openalex", True))
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: called.__setitem__("arxiv", True))

        out = json.loads(tools_mod.academic_search.func("volatility"))
        assert out["ok"] is True
        assert out["source_system"] == "semantic_scholar"
        assert out["results"] == _SS_RESULT
        assert called == {"openalex": False, "arxiv": False}

    def test_semantic_scholar_rate_limited_falls_back_to_openalex(self, monkeypatch):
        def _raise(query, limit=5):
            raise semantic_scholar_client.SemanticScholarQueryError("rate-limited (429) after 3 retries")

        monkeypatch.setattr(semantic_scholar_client, "search_papers", _raise)
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: _OA_RESULT)
        arxiv_called = {"called": False}
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: arxiv_called.__setitem__("called", True))

        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is True
        assert out["source_system"] == "openalex"
        assert out["results"] == _OA_RESULT
        assert arxiv_called["called"] is False

    def test_semantic_scholar_and_openalex_both_fail_falls_back_to_arxiv(self, monkeypatch):
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: (_ for _ in ()).throw(semantic_scholar_client.SemanticScholarUnavailable("down")))
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: (_ for _ in ()).throw(openalex_client.OpenAlexQueryError("429")))
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: _ARXIV_RESULT)

        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is True
        assert out["source_system"] == "arxiv"
        assert out["results"] == _ARXIV_RESULT

    def test_semantic_scholar_empty_falls_through_without_raising(self, monkeypatch):
        # None (no results) is a normal outcome, not an error -- must also fall through.
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: None)
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: _OA_RESULT)
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: None)

        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is True
        assert out["source_system"] == "openalex"

    def test_all_three_fail_returns_ok_false(self, monkeypatch):
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: (_ for _ in ()).throw(semantic_scholar_client.SemanticScholarQueryError("x")))
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: (_ for _ in ()).throw(openalex_client.OpenAlexQueryError("x")))
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: (_ for _ in ()).throw(arxiv_client.ArxivQueryError("x")))

        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is False
        assert "fallback chain" in out["error"]

    def test_all_three_return_none_returns_ok_false(self, monkeypatch):
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: None)
        monkeypatch.setattr(openalex_client, "search_works", lambda query, limit=5: None)
        monkeypatch.setattr(arxiv_client, "search_papers", lambda query, limit=5: None)

        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is False

    def test_max_results_threaded_through_to_first_successful_source(self, monkeypatch):
        captured = {}

        def _ss(query, limit=5):
            captured["limit"] = limit
            return _SS_RESULT

        monkeypatch.setattr(semantic_scholar_client, "search_papers", _ss)
        tools_mod.academic_search.func("q", max_results=2)
        assert captured["limit"] == 2

    def test_result_payload_is_json_serializable_structured_search_shape(self, monkeypatch):
        # Matches the STRUCTURED_SEARCH ResultShape contract binding.py/middleware.py
        # expect: top-level ok/source_system + a results list of url_or_id/title dicts.
        monkeypatch.setattr(semantic_scholar_client, "search_papers", lambda query, limit=5: _SS_RESULT)
        out = json.loads(tools_mod.academic_search.func("q"))
        assert out["ok"] is True
        assert isinstance(out["results"], list)
        assert all("url_or_id" in r and "title" in r for r in out["results"])

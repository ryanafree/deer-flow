"""Tests for dr_core.verify.citation (D8 citation gate). Hermetic: no live HTTP --
the lookup client is exercised only through monkeypatched urlopen or by asserting
the decision function directly against stubbed cite lists.
"""

import urllib.error

from dr_core.models.enums import CitationStatus
from dr_core.verify.citation import decide_citation_status, extract_citations, has_citation, lookup_citations


class TestCiteRe:
    def test_matches_real_volume_reporter_page(self):
        text = "Officials get qualified immunity under Harlow v. Fitzgerald, 457 U.S. 800 (1982)."
        assert has_citation(text)
        assert extract_citations(text) == ["457 U.S. 800"]

    def test_matches_fabricated_but_plausible_cite(self):
        text = "The Supreme Court abolished qualified immunity in Roe v. Doe, 605 U.S. 217 (2025)."
        assert has_citation(text)
        assert extract_citations(text) == ["605 U.S. 217"]

    def test_no_match_on_plain_prose(self):
        text = "Qualified immunity shields officials from suit in most circumstances."
        assert not has_citation(text)
        assert extract_citations(text) == []

    def test_multiple_cites_in_one_text(self):
        text = "See 457 U.S. 800 and also 605 U.S. 217 for contrast."
        assert extract_citations(text) == ["457 U.S. 800", "605 U.S. 217"]


class TestDecideCitationStatus:
    def test_no_cites_stays_unresolved(self):
        assert decide_citation_status([]) is None
        assert decide_citation_status(None) is None

    def test_404_maps_to_not_found(self):
        assert decide_citation_status([{"citation": "605 U.S. 217", "status": 404}]) == CitationStatus.NOT_FOUND

    def test_400_maps_to_not_found(self):
        assert decide_citation_status([{"citation": "605 U.S. 217", "status": 400}]) == CitationStatus.NOT_FOUND

    def test_200_maps_to_resolved(self):
        assert decide_citation_status([{"citation": "457 U.S. 800", "status": 200}]) == CitationStatus.RESOLVED

    def test_ambiguous_300_stays_unresolved(self):
        assert decide_citation_status([{"citation": "1 U.S. 1", "status": 300}]) is None

    def test_any_not_found_dominates_mixed_results(self):
        cites = [{"citation": "457 U.S. 800", "status": 200}, {"citation": "605 U.S. 217", "status": 404}]
        assert decide_citation_status(cites) == CitationStatus.NOT_FOUND

    def test_not_found_dominates_over_ambiguous(self):
        cites = [{"citation": "1 U.S. 1", "status": 300}, {"citation": "605 U.S. 217", "status": 404}]
        assert decide_citation_status(cites) == CitationStatus.NOT_FOUND

    def test_unknown_status_stays_unresolved(self):
        assert decide_citation_status([{"citation": "x", "status": 999}]) is None


class TestLookupCitationsNoAdvanceOnFailure:
    """Every failure path returns None; never raises. No live network -- urlopen
    is monkeypatched or simply never reached (missing token)."""

    async def test_missing_token_returns_none_without_network(self, monkeypatch):
        monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)

        def _boom(*args, **kwargs):
            raise AssertionError("must not touch the network with no token")

        monkeypatch.setattr("urllib.request.urlopen", _boom)
        result = await lookup_citations("457 U.S. 800", token=None)
        assert result is None

    async def test_network_error_returns_none(self, monkeypatch):
        def _raise_urlerror(*args, **kwargs):
            raise urllib.error.URLError("boom")

        monkeypatch.setattr("urllib.request.urlopen", _raise_urlerror)
        result = await lookup_citations("457 U.S. 800", token="fake-token")
        assert result is None

    async def test_timeout_returns_none(self, monkeypatch):
        def _raise_timeout(*args, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
        result = await lookup_citations("457 U.S. 800", token="fake-token", timeout=0.01)
        assert result is None

    async def test_malformed_json_returns_none(self, monkeypatch):
        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"not json"

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp())
        result = await lookup_citations("457 U.S. 800", token="fake-token")
        assert result is None

    async def test_non_list_payload_returns_none(self, monkeypatch):
        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"error": "not a list"}'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp())
        result = await lookup_citations("457 U.S. 800", token="fake-token")
        assert result is None

    async def test_well_formed_payload_normalizes_to_citation_status_pairs(self, monkeypatch):
        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'[{"citation": "605 U.S. 217", "status": 404, "extra_junk": true}]'

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _FakeResp())
        result = await lookup_citations("605 U.S. 217", token="fake-token")
        assert result == [{"citation": "605 U.S. 217", "status": 404}]

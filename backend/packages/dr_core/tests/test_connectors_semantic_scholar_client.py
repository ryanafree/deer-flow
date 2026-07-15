"""Tests for dr_core.connectors.semantic_scholar_client (2026-07-11,
MYTHOS_REVIEW_2026-07-11.md P0 "Repair academic discovery": concurrency=1,
Retry-After-aware Semantic Scholar search).

HERMETIC: no live network. Both the HTTP transport and the retry sleeper are
injected."""

from __future__ import annotations

import threading
import time

import pytest
from dr_core.connectors import semantic_scholar_client as ssc


class TestSearchPapersSuccess:
    def test_success_normalizes_results(self):
        def transport(url, headers):
            assert "query=volatility" in url
            return {
                "data": [
                    {
                        "paperId": "abc123",
                        "title": "Volatility Disagreement as a Signal",
                        "abstract": "We study...",
                        "year": 2024,
                        "authors": [{"name": "A. Researcher"}],
                        "venue": "Journal of Finance",
                        "citationCount": 12,
                        "externalIds": {"DOI": "10.1000/xyz"},
                    }
                ]
            }

        results = ssc.search_papers("volatility", transport=transport)
        assert len(results) == 1
        r = results[0]
        assert r["url_or_id"] == "https://doi.org/10.1000/xyz"
        assert r["title"] == "Volatility Disagreement as a Signal"
        assert r["authors"] == ["A. Researcher"]
        assert r["year"] == 2024
        assert r["citation_count"] == 12

    def test_falls_back_to_semantic_scholar_url_when_no_doi(self):
        def transport(url, headers):
            return {"data": [{"paperId": "xyz789", "title": "T"}]}

        results = ssc.search_papers("q", transport=transport)
        assert results[0]["url_or_id"] == "https://www.semanticscholar.org/paper/xyz789"

    def test_no_results_returns_none(self):
        transport = lambda url, headers: {"data": []}  # noqa: E731
        assert ssc.search_papers("nothing", transport=transport) is None

    def test_limit_truncates(self):
        transport = lambda url, headers: {"data": [{"paperId": str(i), "title": f"P{i}"} for i in range(10)]}  # noqa: E731
        results = ssc.search_papers("q", limit=3, transport=transport)
        assert len(results) == 3

    def test_api_key_sent_when_present(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "sk-test")
        captured = {}

        def transport(url, headers):
            captured["headers"] = headers
            return {"data": []}

        ssc.search_papers("q", transport=transport)
        assert captured["headers"]["x-api-key"] == "sk-test"

    def test_no_api_key_omits_header(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        captured = {}

        def transport(url, headers):
            captured["headers"] = headers
            return {"data": []}

        ssc.search_papers("q", transport=transport)
        assert "x-api-key" not in captured["headers"]


class TestRetryAfterHandling:
    def test_429_then_success_retries_and_honors_retry_after(self):
        calls = []
        sleeps = []

        def transport(url, headers):
            if len(calls) == 0:
                calls.append("rate-limited")
                raise ssc.SemanticScholarRateLimited(retry_after=5.0)
            calls.append("ok")
            return {"data": [{"paperId": "1", "title": "T"}]}

        def sleeper(seconds):
            sleeps.append(seconds)

        results = ssc.search_papers("q", transport=transport, sleep=sleeper)
        assert calls == ["rate-limited", "ok"]
        assert sleeps == [5.0]
        assert results[0]["title"] == "T"

    def test_exhausts_retries_and_raises(self):
        attempts = {"n": 0}

        def transport(url, headers):
            attempts["n"] += 1
            raise ssc.SemanticScholarRateLimited(retry_after=0.01)

        with pytest.raises(ssc.SemanticScholarQueryError):
            ssc.search_papers("q", transport=transport, sleep=lambda s: None)
        # MAX_RETRIES additional attempts after the first -> MAX_RETRIES + 1 total
        assert attempts["n"] == ssc.MAX_RETRIES + 1

    def test_retry_after_parsing_is_bounded(self):
        assert ssc._parse_retry_after("5") == 5.0
        assert ssc._parse_retry_after(None) == ssc.DEFAULT_RETRY_AFTER_S
        assert ssc._parse_retry_after("not-a-number") == ssc.DEFAULT_RETRY_AFTER_S
        assert ssc._parse_retry_after("99999") == ssc.MAX_RETRY_AFTER_S
        assert ssc._parse_retry_after("-5") == 0.0


class TestConcurrencyCap:
    def test_requests_are_serialized_never_overlap(self):
        """Two threads calling search_papers concurrently must never have
        their transport calls overlap in time -- proves the module-level
        lock actually serializes requests (the review's "cap concurrency at
        1" requirement), not just documents an intent."""
        in_flight = {"count": 0, "max_seen": 0}
        guard = threading.Lock()

        def transport(url, headers):
            with guard:
                in_flight["count"] += 1
                in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["count"])
            time.sleep(0.05)
            with guard:
                in_flight["count"] -= 1
            return {"data": []}

        threads = [threading.Thread(target=ssc.search_papers, args=(f"q{i}",), kwargs={"transport": transport}) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert in_flight["max_seen"] == 1

    def test_lock_is_held_across_retry_backoff(self):
        """A second caller must not slip in a request while the first caller
        is sleeping between retries -- the lock has to wrap the whole retry
        loop, not just each individual transport call."""
        events: list[str] = []
        events_lock = threading.Lock()

        def make_transport(name):
            state = {"attempt": 0}

            def transport(url, headers):
                with events_lock:
                    events.append(f"{name}:start")
                state["attempt"] += 1
                if name == "first" and state["attempt"] == 1:
                    raise ssc.SemanticScholarRateLimited(retry_after=0.05)
                with events_lock:
                    events.append(f"{name}:done")
                return {"data": []}

            return transport

        def sleeper(seconds):
            time.sleep(seconds)

        results = {}

        def run(name, delay_before_start):
            time.sleep(delay_before_start)
            results[name] = ssc.search_papers(name, transport=make_transport(name), sleep=sleeper)

        t1 = threading.Thread(target=run, args=("first", 0))
        t2 = threading.Thread(target=run, args=("second", 0.01))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # "first" must fully complete (including its retry) before "second"
        # ever starts -- the lock serializes across the backoff sleep.
        first_done_idx = events.index("first:done")
        second_start_idx = events.index("second:start")
        assert first_done_idx < second_start_idx

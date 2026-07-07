"""C2 source-extraction hook tests — DrLedgerMiddleware.before_model (D5 ruling C).

HERMETIC: no live model, no network. Synthetic state["messages"] built from real
langchain_core message objects, mirroring the AIMessage-tool_call / ToolMessage
pairing pattern established in deerflow's DurableContextMiddleware tests.
"""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dr_core.graph.middleware import DrLedgerMiddleware
from dr_core.graph.state import merge_ledger, merge_run


def _web_search_call(tool_call_id: str = "call_search") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "q"}, "id": tool_call_id, "type": "tool_call"}],
    )


def _web_fetch_call(url: str, tool_call_id: str = "call_fetch") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_fetch", "args": {"url": url}, "id": tool_call_id, "type": "tool_call"}],
    )


class TestWebSearchExtraction:
    def test_three_results_yield_three_sources(self):
        results = [
            {"title": "A", "url": "https://a.example.com", "snippet": "..."},
            {"title": "B", "url": "https://b.example.com", "snippet": "..."},
            {"title": "C", "url": "https://c.example.com", "snippet": "..."},
        ]
        messages = [
            HumanMessage(content="search it"),
            _web_search_call(),
            ToolMessage(content=json.dumps(results), tool_call_id="call_search", name="web_search"),
        ]

        out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        assert out is not None
        sources = out["dr_sources"]
        assert len(sources) == 3
        urls = {record["url_or_id"] for record in sources.values()}
        assert urls == {"https://a.example.com", "https://b.example.com", "https://c.example.com"}
        for result in results:
            record = next(r for r in sources.values() if r["url_or_id"] == result["url"])
            assert record["title"] == result["title"]

    def test_deterministic_ids_for_same_url(self):
        results = [{"title": "A", "url": "https://a.example.com", "snippet": "..."}]
        messages_1 = [_web_search_call("call_1"), ToolMessage(content=json.dumps(results), tool_call_id="call_1", name="web_search")]
        messages_2 = [_web_search_call("call_2"), ToolMessage(content=json.dumps(results), tool_call_id="call_2", name="web_search")]

        out_1 = DrLedgerMiddleware().before_model({"messages": messages_1, "dr_run": {}}, None)
        out_2 = DrLedgerMiddleware().before_model({"messages": messages_2, "dr_run": {}}, None)

        assert list(out_1["dr_sources"].keys()) == list(out_2["dr_sources"].keys())

    def test_malformed_json_is_skipped_without_crash(self):
        messages = [_web_search_call(), ToolMessage(content="not json{{", tool_call_id="call_search", name="web_search")]

        out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        # The message is still marked processed (watermark advances); no sources emitted.
        assert out is not None
        assert out.get("dr_sources") is None or out["dr_sources"] == {}


class TestWebFetchExtraction:
    def test_url_paired_from_tool_call_args_title_from_heading(self):
        url = "https://example.com/page"
        messages = [
            HumanMessage(content="fetch it"),
            _web_fetch_call(url),
            ToolMessage(content="# Example Page\n\nsome body text", tool_call_id="call_fetch", name="web_fetch"),
        ]

        out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        assert out is not None
        sources = out["dr_sources"]
        assert len(sources) == 1
        record = next(iter(sources.values()))
        assert record["url_or_id"] == url
        assert record["title"] == "Example Page"

    def test_error_result_is_skipped_without_crash(self):
        messages = [
            _web_fetch_call("https://example.com/broken"),
            ToolMessage(content="Error: fetch failed", tool_call_id="call_fetch", name="web_fetch"),
        ]

        out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        assert out is not None
        assert out.get("dr_sources") is None or out["dr_sources"] == {}


class TestNonSourceToolIgnored:
    def test_bash_tool_message_ignored(self):
        messages = [
            AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "ls"}, "id": "call_bash", "type": "tool_call"}]),
            ToolMessage(content="file1\nfile2", tool_call_id="call_bash", name="bash"),
        ]

        out = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        assert out is None


class TestIdempotency:
    def test_applying_hook_twice_over_same_messages_does_not_raise_or_change_sources(self):
        url = "https://example.com/page"
        messages = [
            _web_fetch_call(url),
            ToolMessage(content="# Example Page\n\nbody", tool_call_id="call_fetch", name="web_fetch"),
        ]

        first = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)
        assert first is not None

        # Simulate the reducers folding the first update into state, then
        # re-run the hook over the SAME messages (e.g. a later graph step).
        state_after_first = {
            "messages": messages,
            "dr_run": first["dr_run"],
            "dr_sources": first["dr_sources"],
        }

        second = DrLedgerMiddleware().before_model(state_after_first, None)

        # No unprocessed source-bearing messages remain -> no update at all,
        # so merge_ledger is never even asked to reconcile a divergent write.
        assert second is None
        assert state_after_first["dr_sources"] == first["dr_sources"]

    def test_reducers_tolerate_a_forced_replay_of_the_same_update(self):
        """Belt-and-suspenders: even if the SAME (already-applied) update were
        folded in again, merge_ledger's identical-dict no-op path and
        merge_run's plain dict-update must not raise."""
        url = "https://example.com/page"
        messages = [
            _web_fetch_call(url),
            ToolMessage(content="# Example Page\n\nbody", tool_call_id="call_fetch", name="web_fetch"),
        ]
        first = DrLedgerMiddleware().before_model({"messages": messages, "dr_run": {}}, None)

        merged_sources = merge_ledger(first["dr_sources"], first["dr_sources"])
        merged_run = merge_run(first["dr_run"], first["dr_run"])

        assert merged_sources == first["dr_sources"]
        assert merged_run == first["dr_run"]


class TestCrossMessageUrlDedup:
    """The watermark only guards re-processing the SAME ToolMessage. The same
    URL routinely resurfaces across DIFFERENT ToolMessages (search-then-fetch,
    or two searches with overlapping results); each sighting would otherwise
    mint a fresh retrieved_at for the same deterministic source id, and
    merge_ledger raises on that same-field divergent write. These tests drive
    the hook's output through the REAL merge_ledger, folding step by step as
    the graph would, to prove that never happens."""

    def test_search_then_fetch_same_url_does_not_raise_and_keeps_first_retrieved_at(self):
        url = "https://example.com/page"
        search_results = [{"title": "Example Page", "url": url, "snippet": "..."}]
        search_messages = [
            _web_search_call("call_search"),
            ToolMessage(content=json.dumps(search_results), tool_call_id="call_search", name="web_search"),
        ]

        middleware = DrLedgerMiddleware()
        first = middleware.before_model({"messages": search_messages, "dr_run": {}}, None)
        assert first is not None

        dr_sources = merge_ledger(None, first["dr_sources"])
        dr_run = merge_run(None, first["dr_run"])
        first_retrieved_at = next(iter(dr_sources.values()))["retrieved_at"]

        # The model now fetches that same url in a later turn (different
        # tool_call_id, so the watermark alone would not catch it).
        all_messages = [
            *search_messages,
            _web_fetch_call(url, "call_fetch"),
            ToolMessage(content="# Example Page\n\nbody", tool_call_id="call_fetch", name="web_fetch"),
        ]
        state = {"messages": all_messages, "dr_run": dr_run, "dr_sources": dr_sources}

        second = middleware.before_model(state, None)

        assert second is not None
        # No raise: fold the second update's dr_sources (absent, since the
        # url was already known) through the real reducer.
        merged = merge_ledger(dr_sources, second.get("dr_sources"))
        assert merged == dr_sources
        assert next(iter(merged.values()))["retrieved_at"] == first_retrieved_at
        assert len(merged) == 1

    def test_two_web_searches_with_overlapping_url_emitted_once_no_raise(self):
        shared_url = "https://example.com/shared"
        results_1 = [{"title": "Shared", "url": shared_url, "snippet": "..."}]
        results_2 = [
            {"title": "Shared", "url": shared_url, "snippet": "..."},
            {"title": "Other", "url": "https://example.com/other", "snippet": "..."},
        ]
        messages_1 = [
            _web_search_call("call_search_1"),
            ToolMessage(content=json.dumps(results_1), tool_call_id="call_search_1", name="web_search"),
        ]

        middleware = DrLedgerMiddleware()
        first = middleware.before_model({"messages": messages_1, "dr_run": {}}, None)
        assert first is not None
        dr_sources = merge_ledger(None, first["dr_sources"])
        dr_run = merge_run(None, first["dr_run"])
        shared_retrieved_at = next(iter(dr_sources.values()))["retrieved_at"]

        messages_2 = [
            *messages_1,
            _web_search_call("call_search_2"),
            ToolMessage(content=json.dumps(results_2), tool_call_id="call_search_2", name="web_search"),
        ]
        second = middleware.before_model({"messages": messages_2, "dr_run": dr_run, "dr_sources": dr_sources}, None)

        assert second is not None
        merged = merge_ledger(dr_sources, second.get("dr_sources"))

        urls = {record["url_or_id"] for record in merged.values()}
        assert urls == {shared_url, "https://example.com/other"}
        shared_record = next(r for r in merged.values() if r["url_or_id"] == shared_url)
        assert shared_record["retrieved_at"] == shared_retrieved_at

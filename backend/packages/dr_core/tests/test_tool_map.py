"""Tests for dr_core.connectors.tool_map -- the tool-name -> connector-name(s)
binding DrProfileToolMiddleware reads (S9-B orchestrator ruling)."""

from __future__ import annotations

from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS, connectors_for_tool, register_tool


def test_seeded_community_tools_map_to_the_retrieval_backbone():
    assert connectors_for_tool("web_search") == frozenset({"tavily", "searxng", "crawl4ai", "jina_reader"})
    assert connectors_for_tool("web_fetch") == frozenset({"tavily", "searxng", "crawl4ai", "jina_reader"})


def test_unmapped_tool_name_returns_empty_frozenset():
    assert connectors_for_tool("bash") == frozenset()
    assert connectors_for_tool("record_claim") == frozenset()


def test_register_tool_adds_a_new_binding():
    try:
        register_tool("__test_only_tool", "wrds")
        assert connectors_for_tool("__test_only_tool") == frozenset({"wrds"})
    finally:
        TOOL_TO_CONNECTORS.pop("__test_only_tool", None)


def test_register_tool_is_additive_for_repeated_calls():
    try:
        register_tool("__test_only_multi", "fred")
        register_tool("__test_only_multi", "world_bank")
        assert connectors_for_tool("__test_only_multi") == frozenset({"fred", "world_bank"})
    finally:
        TOOL_TO_CONNECTORS.pop("__test_only_multi", None)

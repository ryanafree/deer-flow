"""tool_map.py — the tool-name -> connector-name(s) binding (S9-B orchestrator ruling).

Profiles (``profiles/*.yaml``) list a ``tool_allowlist`` of CONNECTOR names
only -- they know nothing about the bound LangChain tool objects a model
actually calls. ``DrProfileToolMiddleware`` (``dr_core/graph/profile_middleware.py``)
needs the reverse direction: given a bound tool's NAME, which connector(s)
back it, so it can check that at least one is in the active profile's
allowlist. This module is the ONE explicit map for that translation, owned by
the connector layer per the orchestrator ruling in HANDOFF_CONNECTORS_BC.md
(open question 1) -- profiles stay connector-name-only, no per-profile tool
maps.

Seeded entries:
  - ``web_search`` / ``web_fetch`` -- the community tavily/searxng/crawl4ai/
    jina_reader providers all hardcode these two tool names (upstream issue
    #1803; Option A finding, BUILD_LEDGER 2026-07-08), so at runtime there is
    no way to know which specific provider actually backed a given call.
    Both names map to the full set of Tier-0 unstructured-web connectors
    (``profiles: all`` in connectors.yaml) that could plausibly be behind
    them. Every profile's allowlist includes this whole set, so this mapping
    never itself causes a denial -- it exists so the map is complete, not to
    gate the retrieval backbone.

A tool name absent from this map is NOT a connector tool at all (bash,
record_claim, task, ask_clarification, write_todos, ...) and is out of scope
for profile enforcement -- ``DrProfileToolMiddleware`` passes those through
unfiltered. A tool name PRESENT in this map is in scope: it is allowed only
if at least one of its mapped connectors is in the active profile's
allowlist, denied otherwise (the "unknown/disallowed connector tool ->
default-deny" rule).

Stage C (structured connector tools, ``dr_core/connectors/tools.py``) MUST
call ``register_tool(tool_name, connector_name)`` for every new typed tool it
mints, at tool-creation time (module import), before any graph is built --
each typed tool binds 1:1 to the connector it was generated from.
"""

from __future__ import annotations

TOOL_TO_CONNECTORS: dict[str, frozenset[str]] = {
    "web_search": frozenset({"tavily", "searxng", "crawl4ai", "jina_reader"}),
    "web_fetch": frozenset({"tavily", "searxng", "crawl4ai", "jina_reader"}),
}


def connectors_for_tool(tool_name: str) -> frozenset[str]:
    """Connector names backing ``tool_name``, or an empty frozenset if
    ``tool_name`` is not a connector-backed tool (out of scope for profile
    enforcement -- callers should treat that as pass-through, not deny)."""
    return TOOL_TO_CONNECTORS.get(tool_name, frozenset())


def register_tool(tool_name: str, connector_name: str) -> None:
    """Stage C hook: bind a newly-minted structured tool's name to its
    connector. Additive -- safe to call more than once for the same name
    (e.g. module re-import in tests)."""
    TOOL_TO_CONNECTORS[tool_name] = connectors_for_tool(tool_name) | {connector_name}

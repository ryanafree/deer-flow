"""binding.py — ToolBinding registry (D11 P0, "normalize the connector boundary").

Resolves an ACTUAL bound tool name (as it appears on ``ToolMessage.name`` at
runtime) to the connector/parser metadata the source-capture hook needs, so
capture no longer depends on a hard-coded list of tool names known in
advance (MYTHOS_REVIEW_2026-07-11.md's primary integration defect).

Two independent naming schemes bind to the same connector, and this module is
the ONE place that understands both:

  - Non-MCP tools keep their literal names always: DeerFlow's community
    web_search/web_fetch wrappers (tavily/searxng/crawl4ai all hardcode these
    two names per ``connectors/tool_map.py``'s docstring) and dr_core's own
    Stage-C typed tools (wrds_query, edgar_company_facts, fred_series,
    courtlistener_search — ``connectors/tools.py``). None of these are loaded
    through ``deerflow.mcp.tools``'s ``MultiServerMCPClient``, so none of them
    are ever prefixed.
  - MCP tools: ``harness/deerflow/mcp/tools.py`` (around line 628) loads every
    MCP server's tools through ``MultiServerMCPClient(..., tool_name_prefix=
    True)``, which renames each tool to ``f"{server_name}_{original_tool_
    name}"`` (that module's own prefix-strip logic a few lines later,
    ``tool.name.startswith(f"{name}_")``, is the other half of this
    convention). dr_core does not control extensions_config.json's server
    keys and never sees the live MCP tool inventory, so the only robust
    resolution available here is a longest-prefix match of the runtime name
    against the connector names dr_core DOES know about (connectors.yaml),
    applied lazily to whatever name shows up on a ToolMessage.

Unknown names resolve to ``None`` — callers must fail closed (treat as
"not a source-bearing tool"), never guess at a shape for a name this
registry doesn't recognize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from dr_core.connectors.registry import Connector, load_connectors


class ResultShape(Enum):
    """How a resolved tool's JSON payload maps onto Source records."""

    SEARCH_URL = "search_url"  # list of {url, title} (web_search shape)
    FETCH_URL = "fetch_url"  # one page; the url comes from the CALL args, not the payload
    STRUCTURED_SINGLE = "structured_single"  # one {ok, url_or_id, title} payload
    STRUCTURED_SEARCH = "structured_search"  # {ok, results: [{url_or_id, title}, ...]}
    MCP_GENERIC = "mcp_generic"  # third-party MCP payload shape dr_core does not know


@dataclass(frozen=True)
class ToolBinding:
    """One resolved (bound tool name -> connector) fact, and enough to parse it."""

    runtime_tool_name: str
    connector: str | None  # canonical connector id, or None for the generic web tools
    source_system: str
    authority_tier: int
    result_shape: ResultShape


# Connectors whose tools are wired through DeerFlow's community integrations
# (tavily/searxng/crawl4ai — see tool_map.py's docstring, the authority for
# this list) rather than the generic MCP loader. Their calls always surface
# under the literal names web_search/web_fetch, never prefixed, so they are
# excluded from prefix-match candidacy below (matching them there would be
# dead code at best, a false-positive collision at worst).
_COMMUNITY_WIRED_CONNECTORS = frozenset({"tavily", "searxng", "crawl4ai"})

_STATIC_BINDINGS: dict[str, ToolBinding] = {
    "web_search": ToolBinding("web_search", None, "web", 3, ResultShape.SEARCH_URL),
    "web_fetch": ToolBinding("web_fetch", None, "web", 3, ResultShape.FETCH_URL),
    "wrds_query": ToolBinding("wrds_query", "wrds", "wrds", 1, ResultShape.STRUCTURED_SINGLE),
    "edgar_company_facts": ToolBinding("edgar_company_facts", "edgar", "edgar", 1, ResultShape.STRUCTURED_SINGLE),
    "fred_series": ToolBinding("fred_series", "fred", "fred", 1, ResultShape.STRUCTURED_SINGLE),
    "courtlistener_search": ToolBinding("courtlistener_search", "courtlistener", "courtlistener", 1, ResultShape.STRUCTURED_SEARCH),
    # academic_search (connectors/tools.py, the review's sibling "Repair
    # academic discovery" P0) is one Stage-C tool backed by a semantic_scholar
    # -> openalex -> arxiv fallback chain -- which connector actually served a
    # given call varies, so `connector` here is deliberately None (ambiguous
    # at the binding level); the payload's own `source_system` field is the
    # ground truth per call (see middleware.py's structured-tool parsers,
    # which now prefer the payload's source_system over this default).
    "academic_search": ToolBinding("academic_search", None, "academic", 1, ResultShape.STRUCTURED_SEARCH),
}

# Institutional/primary connectors get the same higher-than-web authority
# tier the Stage-C structured tools already use (middleware.py's docstring:
# "institutional primary data, not a crawled page"); practitioner artifacts
# sit between that and the generic web default.
_AUTHORITY_TIER_BY_EVIDENCE_CLASS = {"academic": 1, "primary_data": 1, "practitioner": 2, "news": 3}
_DEFAULT_MCP_AUTHORITY_TIER = 3

# Reverse mapping: connector name -> the literal bound tool name(s), for
# connectors whose runtime tool name dr_core knows exactly (Stage C typed
# tools + the two community-wired web tools). Connectors reached only through
# the generic MCP loader don't have a knowable literal name ahead of time
# (their bound name depends on tool_name_prefix applied to whatever the
# third-party MCP server itself calls the tool) -- describe_connector_tools
# falls back to the prefix pattern for those.
_KNOWN_LITERAL_TOOL_NAMES: dict[str, tuple[str, ...]] = {
    "wrds": ("wrds_query",),
    "edgar": ("edgar_company_facts",),
    "fred": ("fred_series",),
    "courtlistener": ("courtlistener_search",),
    "tavily": ("web_search", "web_fetch"),
    "searxng": ("web_search", "web_fetch"),
    "crawl4ai": ("web_search", "web_fetch"),
    # All three ride the Stage-C academic_search fallback chain (see
    # connectors/tools.py's _ACADEMIC_SEARCH_CHAIN) rather than their own raw
    # MCP tool name.
    "semantic_scholar": ("academic_search",),
    "openalex": ("academic_search",),
    "arxiv": ("academic_search",),
}


def describe_connector_tools(connector_name: str) -> str:
    """User-facing (directive-prose) hint for which bound tool name(s) back a
    connector: the exact name(s) when dr_core knows them statically, else the
    MCP tool_name_prefix pattern (module docstring) -- the exact third-party
    tool name is only known at runtime MCP discovery, which dr_core doesn't
    have access to."""
    literal = _KNOWN_LITERAL_TOOL_NAMES.get(connector_name)
    if literal:
        return " or ".join(literal)
    return f"{connector_name}_* (MCP-prefixed)"


class ToolBindingRegistry:
    """Lazily built, connectors.yaml-backed resolver: bound tool name -> ToolBinding | None."""

    def __init__(self, connectors: list[Connector] | None = None):
        connectors = connectors if connectors is not None else load_connectors()
        # Longest connector name first, so a connector name that happens to be
        # a prefix of another's can't shadow the more specific match.
        self._mcp_connectors: list[Connector] = sorted(
            (c for c in connectors if c.has_mcp() and c.name not in _COMMUNITY_WIRED_CONNECTORS),
            key=lambda c: len(c.name),
            reverse=True,
        )

    def resolve(self, runtime_tool_name: str) -> ToolBinding | None:
        """Resolve a ToolMessage.name to its ToolBinding, whether or not
        tool_name_prefix was applied. Returns None for anything this registry
        doesn't recognize -- callers must treat that as "not a source-bearing
        tool", never guess at a shape."""
        static = _STATIC_BINDINGS.get(runtime_tool_name)
        if static is not None:
            return static
        for connector in self._mcp_connectors:
            if runtime_tool_name.startswith(f"{connector.name}_"):
                tier = _AUTHORITY_TIER_BY_EVIDENCE_CLASS.get(connector.evidence_class, _DEFAULT_MCP_AUTHORITY_TIER)
                return ToolBinding(
                    runtime_tool_name=runtime_tool_name,
                    connector=connector.name,
                    source_system=connector.name,
                    authority_tier=tier,
                    result_shape=ResultShape.MCP_GENERIC,
                )
        return None


@lru_cache(maxsize=1)
def get_default_registry() -> ToolBindingRegistry:
    """Process-wide singleton over the vendored connectors.yaml. Tests that
    need a different connector set should construct ``ToolBindingRegistry``
    directly instead of touching this cache."""
    return ToolBindingRegistry()


_MCP_GENERIC_URL_KEYS = ("url", "url_or_id", "link", "landing_page_url", "html_url")
_MCP_GENERIC_TITLE_KEYS = ("title", "name", "case_name")


def mcp_generic_url_and_title(record: dict) -> tuple[str | None, str | None]:
    """Best-effort (url, title) pair for one MCP_GENERIC record dict, tried
    against a handful of conventional field names. Neither field found ->
    (None, None); the caller (middleware.py) skips records with no url."""
    url = next((record[key] for key in _MCP_GENERIC_URL_KEYS if isinstance(record.get(key), str) and record.get(key)), None)
    title = next((record[key] for key in _MCP_GENERIC_TITLE_KEYS if isinstance(record.get(key), str) and record.get(key)), None)
    return url, title


def parse_mcp_generic_records(content: str) -> list[dict]:
    """Best-effort record extraction for ResultShape.MCP_GENERIC: dr_core
    knows the CONNECTOR behind a prefixed MCP tool call but not the
    third-party server's own JSON shape. Accepts a bare list, or a dict
    carrying a list under a conventional results-ish key, of record dicts;
    falls back to treating the whole payload as one record. Returns [] for
    anything unparsable -- callers fail closed the same way the other
    structured parsers do on bad JSON."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return []

    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        for key in ("results", "items", "data", "papers", "works"):
            value = payload.get(key)
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        return [payload]
    return []

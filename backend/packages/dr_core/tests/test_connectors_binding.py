"""Tests for dr_core.connectors.binding (D11 P0, connector-boundary normalization).

HERMETIC: no live model, no network, no live MCP client. Builds a small
synthetic connector list (mirroring test_connectors_registry.py's fixture
style) instead of depending on the full vendored connectors.yaml, so these
tests pin the resolution RULES, not the current connector roster.
"""

from dr_core.connectors.binding import ResultShape, ToolBindingRegistry, describe_connector_tools, get_default_registry
from dr_core.connectors.registry import Connector


def _connector(name, *, access="mcp+rest", evidence_class=None) -> Connector:
    return Connector(
        name=name,
        type="graph",
        tier=1,
        profiles=[name],
        access=access,
        auth="none",
        launcher="uvx",
        durability="stable-api",
        rate_limit="free",
        last_verified="2026-06-23",
        fallback=None,
        evidence_class=evidence_class,
    )


def _registry() -> ToolBindingRegistry:
    connectors = [
        _connector("semantic_scholar", evidence_class="academic"),
        _connector("openalex", evidence_class="academic"),
        _connector("github", evidence_class="practitioner"),
        _connector("courtlistener_rest", access="rest"),  # rest-only: never MCP-prefixed
        _connector("tavily"),  # community-wired: excluded from prefix candidacy
    ]
    return ToolBindingRegistry(connectors)


class TestStaticBindings:
    """Non-MCP tools (community web tools, Stage-C typed tools) always keep
    their literal names -- resolution must not depend on connectors.yaml at
    all for these."""

    def test_web_search_resolves_unprefixed(self):
        binding = _registry().resolve("web_search")
        assert binding is not None
        assert binding.connector is None
        assert binding.result_shape is ResultShape.SEARCH_URL

    def test_wrds_query_resolves_unprefixed_to_structured_single(self):
        binding = _registry().resolve("wrds_query")
        assert binding.connector == "wrds"
        assert binding.result_shape is ResultShape.STRUCTURED_SINGLE
        assert binding.authority_tier == 1

    def test_courtlistener_search_resolves_unprefixed_to_structured_search(self):
        binding = _registry().resolve("courtlistener_search")
        assert binding.connector == "courtlistener"
        assert binding.result_shape is ResultShape.STRUCTURED_SEARCH


class TestPrefixedMcpResolution:
    """The primary D11 fix: MCP tool_name_prefix renames a connector's tools
    to f"{server_name}_{original_name}" (harness/mcp/tools.py:628); the
    registry must resolve that runtime name back to the connector by a
    longest-prefix match, without dr_core ever seeing the live tool
    inventory."""

    def test_semantic_scholar_prefixed_tool_resolves_to_connector(self):
        binding = _registry().resolve("semantic_scholar_search_papers")
        assert binding is not None
        assert binding.connector == "semantic_scholar"
        assert binding.source_system == "semantic_scholar"
        assert binding.result_shape is ResultShape.MCP_GENERIC
        # academic evidence_class -> institutional tier, same as Stage C tools.
        assert binding.authority_tier == 1

    def test_openalex_prefixed_tool_resolves_to_connector(self):
        binding = _registry().resolve("openalex_search_works")
        assert binding.connector == "openalex"

    def test_practitioner_connector_gets_tier_2(self):
        binding = _registry().resolve("github_search_repositories")
        assert binding.connector == "github"
        assert binding.authority_tier == 2

    def test_longest_prefix_wins_no_false_short_match(self):
        # A tool name that happens to start with a shorter connector's name
        # as a substring, but is not actually that connector's prefix, must
        # not resolve at all -- e.g. "semantic_scholarly_thing" is not
        # "semantic_scholar_"-prefixed.
        assert _registry().resolve("semantic_scholarly_thing") is None


class TestCommunityWiredExclusion:
    def test_tavily_is_not_a_prefix_candidate(self):
        # tavily is community-wired (always surfaces as web_search/web_fetch,
        # never its own prefixed name) -- a stray "tavily_anything" name must
        # not resolve via the connector-prefix path.
        assert _registry().resolve("tavily_anything") is None


class TestUnknownNamesFailClosed:
    def test_unrelated_tool_name_resolves_to_none(self):
        assert _registry().resolve("bash") is None
        assert _registry().resolve("record_claim") is None

    def test_rest_only_connector_never_matches_by_prefix(self):
        # courtlistener_rest has access="rest" (no MCP) -- must never be
        # offered as a prefix candidate even though its name is a superset
        # of another connector's.
        assert _registry().resolve("courtlistener_rest_search") is None


class TestDescribeConnectorTools:
    def test_known_literal_tool_name_is_exact(self):
        assert describe_connector_tools("wrds") == "wrds_query"

    def test_community_wired_connector_lists_both_web_tools(self):
        assert describe_connector_tools("tavily") == "web_search or web_fetch"

    def test_unknown_mcp_connector_falls_back_to_prefix_pattern(self):
        # github has no Stage-C typed tool -- it's only ever reached through
        # the generic MCP loader, so its live bound name isn't knowable ahead
        # of time.
        assert describe_connector_tools("github") == "github_* (MCP-prefixed)"

    def test_academic_search_backed_connectors_use_the_literal_tool_name(self):
        # semantic_scholar/openalex/arxiv all ride the Stage-C academic_search
        # fallback chain (connectors/tools.py) -- dr_core knows this exact
        # literal name, unlike a pure-MCP connector.
        for connector_name in ("semantic_scholar", "openalex", "arxiv"):
            assert describe_connector_tools(connector_name) == "academic_search"


class TestDefaultRegistrySingleton:
    def test_default_registry_resolves_real_vendored_connectors(self):
        # Sanity check against the real connectors.yaml (no synthetic
        # fixture): semantic_scholar is Tier 1 mcp+rest in the vendored file,
        # so its MCP-prefixed tool names must resolve through the singleton
        # the same way the DrLedgerMiddleware hook uses it.
        registry = get_default_registry()
        binding = registry.resolve("semantic_scholar_search_papers")
        assert binding is not None
        assert binding.connector == "semantic_scholar"

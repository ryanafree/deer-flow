"""Tests for dr_core.connectors.generator (S9)."""

import json

import yaml

from dr_core.connectors.generator import (
    build_mcp_fragment,
    build_policy_map,
    build_tools_fragment,
    generate,
    write_generated,
)
from dr_core.connectors.registry import load_connectors


class TestMcpFragment:
    def test_only_tier_0_and_1_mcp_capable_connectors_included(self):
        connectors = load_connectors()
        fragment = build_mcp_fragment(connectors)
        servers = fragment["mcpServers"]
        assert "tavily" in servers  # tier 0, mcp+rest
        assert "docling" in servers  # tier 0, mcp only
        assert "courtlistener" in servers  # tier 1, mcp+rest
        # Tier 2, rest-only connectors must never appear as MCP servers.
        assert "fred" not in servers
        assert "jina_reader" not in servers  # rest-only, no mcp

    def test_stdio_launcher_shape(self):
        connectors = load_connectors()
        fragment = build_mcp_fragment(connectors)
        docling = fragment["mcpServers"]["docling"]
        assert docling["type"] == "stdio"
        assert docling["command"] == "uvx"
        assert isinstance(docling["args"], list)
        assert docling["enabled"] is False

    def test_http_launcher_shape(self):
        connectors = load_connectors()
        fragment = build_mcp_fragment(connectors)
        courtlistener = fragment["mcpServers"]["courtlistener"]
        assert courtlistener["type"] == "http"
        assert courtlistener["url"] == "https://mcp.courtlistener.com"

    def test_auth_env_included_when_declared(self):
        connectors = load_connectors()
        fragment = build_mcp_fragment(connectors)
        openalex = fragment["mcpServers"]["openalex"]
        assert openalex["env"] == {"OPENALEX_API_KEY": "$OPENALEX_API_KEY"}
        docling = fragment["mcpServers"]["docling"]
        assert "env" not in docling  # auth: none


class TestToolsFragment:
    def test_rest_capable_connectors_included(self):
        connectors = load_connectors()
        fragment = build_tools_fragment(connectors)
        names = {t["name"] for t in fragment["tools"]}
        assert "fred_fetch" in names
        assert "tavily_fetch" in names  # mcp+rest still has a REST substrate
        assert "docling_fetch" not in names  # mcp-only, no REST

    def test_auth_env_field_present_when_needed(self):
        connectors = load_connectors()
        fragment = build_tools_fragment(connectors)
        fred = next(t for t in fragment["tools"] if t["name"] == "fred_fetch")
        assert fred["auth_env"] == "FRED_API_KEY"
        world_bank = next(t for t in fragment["tools"] if t["name"] == "world_bank_fetch")
        assert "auth_env" not in world_bank


class TestPolicyMap:
    def test_defaults_by_durability(self):
        connectors = load_connectors()
        policies = build_policy_map(connectors)
        idx = {c.name: c for c in connectors}

        substrate_name = next(n for n, c in idx.items() if c.durability == "substrate")
        stable_name = next(n for n, c in idx.items() if c.durability == "stable-api")
        fragile_name = next(n for n, c in idx.items() if c.durability == "fragile-wrapper")

        assert policies[substrate_name] == {"timeout_s": 20, "retries": 2, "breaker_threshold": 3}
        assert policies[stable_name] == {"timeout_s": 15, "retries": 1, "breaker_threshold": 3}
        assert policies[fragile_name] == {"timeout_s": 10, "retries": 0, "breaker_threshold": 1}

    def test_per_connector_override_layers_on_top_of_default(self):
        connectors = load_connectors()
        name = connectors[0].name
        policies = build_policy_map(connectors, overrides={name: {"timeout_s": 99}})
        assert policies[name]["timeout_s"] == 99
        # Untouched fields keep the durability default.
        default_retries = build_policy_map(connectors)[name]["retries"]
        assert policies[name]["retries"] == default_retries


class TestGenerateAndWrite:
    def test_generate_returns_all_three_artifacts(self):
        result = generate()
        assert set(result.keys()) == {"mcp_fragment", "tools_fragment", "policy_map"}

    def test_write_generated_produces_well_formed_files(self, tmp_path):
        out_dir = str(tmp_path / "generated")
        paths = write_generated(out_dir)

        with open(paths["mcp_fragment"]) as f:
            mcp_doc = json.load(f)
        assert "mcpServers" in mcp_doc

        with open(paths["tools_fragment"]) as f:
            tools_doc = yaml.safe_load(f)
        assert "tools" in tools_doc
        assert isinstance(tools_doc["tools"], list)

        with open(paths["policy_map"]) as f:
            policy_doc = json.load(f)
        assert "tavily" in policy_doc
        assert "timeout_s" in policy_doc["tavily"]

    def test_write_generated_never_touches_real_repo_config(self, tmp_path):
        """Generator must only ever write into the given out_dir -- never the repo's
        real config.yaml / extensions_config.json."""
        out_dir = str(tmp_path / "generated")
        write_generated(out_dir)
        repo_root = tmp_path.parents[0]
        assert not (repo_root / "config.yaml").exists()
        assert not (repo_root / "extensions_config.json").exists()

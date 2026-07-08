"""generator.py — emit config fragments from the connector registry, for human review only.

NEVER auto-applies to the repo's real config.yaml / extensions_config.json. Emits
three artifacts to an output directory:

  - ``extensions_config.fragment.json`` -- an ``mcpServers`` fragment for tier<=1
    MCP-capable connectors (access ``mcp``/``mcp+rest``, launcher != ``none``).
    Exact MCP package specs (npx/uvx args) are placeholders a human must fill in;
    this generator does not invent package names connectors.yaml does not record.
  - ``tools.fragment.yaml`` -- a ``tools:`` list fragment (config.yaml shape) for
    every REST-capable connector (access ``rest``/``mcp+rest``), one entry per
    connector's REST substrate.
  - ``connector_policy.json`` -- per-connector ``{timeout_s, retries,
    breaker_threshold}``, defaulted from the connector's ``durability`` rating and
    overridable per-connector.

Durability -> policy defaults (S9 contract):
  substrate:        20s timeout, 2 retries, breaker_threshold 3
  stable-api:       15s timeout, 1 retry,  breaker_threshold 3
  fragile-wrapper:  10s timeout, 0 retries, breaker_threshold 1
"""

from __future__ import annotations

import argparse
import json
import os

import yaml

from dr_core.connectors.registry import Connector, load_connectors

DURABILITY_POLICY_DEFAULTS = {
    "substrate": {"timeout_s": 20, "retries": 2, "breaker_threshold": 3},
    "stable-api": {"timeout_s": 15, "retries": 1, "breaker_threshold": 3},
    "fragile-wrapper": {"timeout_s": 10, "retries": 0, "breaker_threshold": 1},
}

DEFAULT_OUT_DIR = os.path.expanduser("~/Documents/Projects/deerflow-dr/deer-flow/build-logs/connectors-generated/")


def build_mcp_fragment(connectors: list[Connector]) -> dict:
    """mcpServers fragment for tier<=1 MCP-capable connectors (extensions_config.json shape)."""
    servers: dict[str, dict] = {}
    for c in connectors:
        if c.tier > 1 or not c.has_mcp() or c.launcher == "none":
            continue
        entry: dict = {
            "enabled": False,
            "description": f"{c.name} ({c.type}, tier {c.tier}) -- from connectors.yaml",
        }
        if c.launcher in ("uvx", "npx"):
            entry["type"] = "stdio"
            entry["command"] = c.launcher
            # Placeholder: connectors.yaml records a logical endpoint id, not the
            # exact package spec this launcher needs. A human fills in the real
            # package/version before enabling.
            entry["args"] = [f"TODO-package-spec-for-{c.name}", f"# endpoint hint: {c.endpoint}"]
        elif c.launcher == "http":
            entry["type"] = "http"
            entry["url"] = c.endpoint
        if c.auth and c.auth != "none":
            entry["env"] = {c.auth: f"${c.auth}"}
        servers[c.name] = entry
    return {"mcpServers": servers}


def build_tools_fragment(connectors: list[Connector]) -> dict:
    """config.yaml tools[] fragment for every REST-capable connector."""
    tools = []
    for c in connectors:
        if not c.has_rest():
            continue
        base = c.rest_endpoint or c.endpoint
        tool: dict = {
            "name": f"{c.name}_fetch",
            "group": "web",
            "base_url": base,
            "tier": c.tier,
        }
        if c.auth and c.auth != "none":
            tool["auth_env"] = c.auth
        tools.append(tool)
    return {"tools": tools}


def build_policy_map(connectors: list[Connector], overrides: dict[str, dict] | None = None) -> dict[str, dict]:
    """Per-connector {timeout_s, retries, breaker_threshold}, defaulted by durability,
    with optional per-connector-name overrides layered on top."""
    overrides = overrides or {}
    policies: dict[str, dict] = {}
    for c in connectors:
        base = dict(DURABILITY_POLICY_DEFAULTS[c.durability])
        base.update(overrides.get(c.name, {}))
        policies[c.name] = base
    return policies


def generate(connectors: list[Connector] | None = None, policy_overrides: dict[str, dict] | None = None) -> dict:
    """Build all three fragments in memory (no I/O). Returns a dict with keys
    'mcp_fragment', 'tools_fragment', 'policy_map'."""
    connectors = connectors if connectors is not None else load_connectors()
    return {
        "mcp_fragment": build_mcp_fragment(connectors),
        "tools_fragment": build_tools_fragment(connectors),
        "policy_map": build_policy_map(connectors, policy_overrides),
    }


def write_generated(out_dir: str, connectors: list[Connector] | None = None, policy_overrides: dict[str, dict] | None = None) -> dict[str, str]:
    """Write the three fragments to out_dir for human review. Returns the written paths.
    Never touches the repo's real config.yaml / extensions_config.json."""
    os.makedirs(out_dir, exist_ok=True)
    artifacts = generate(connectors, policy_overrides)

    paths = {
        "mcp_fragment": os.path.join(out_dir, "extensions_config.fragment.json"),
        "tools_fragment": os.path.join(out_dir, "tools.fragment.yaml"),
        "policy_map": os.path.join(out_dir, "connector_policy.json"),
    }
    with open(paths["mcp_fragment"], "w") as f:
        json.dump(artifacts["mcp_fragment"], f, indent=2)
        f.write("\n")
    with open(paths["tools_fragment"], "w") as f:
        yaml.safe_dump(artifacts["tools_fragment"], f, sort_keys=False)
    with open(paths["policy_map"], "w") as f:
        json.dump(artifacts["policy_map"], f, indent=2)
        f.write("\n")
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory for the three generated fragments")
    ap.add_argument("--connectors-yaml", default=None, help="override the vendored connectors.yaml path")
    args = ap.parse_args(argv)

    connectors = load_connectors(args.connectors_yaml)
    paths = write_generated(args.out, connectors)
    print(f"wrote {len(connectors)} connectors' worth of config fragments to {args.out} (NOT auto-applied):")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

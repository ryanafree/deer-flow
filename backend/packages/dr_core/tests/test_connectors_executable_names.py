"""Executable-name config sanity tests (2026-07-11, MYTHOS_REVIEW_2026-07-11.md
P0 "Repair academic discovery": Docling and OpenBB connectors could never
launch because the configured executable name did not match the console
script the installed PyPI package actually exports).

Verified locally against real `uv`/PyPI output already captured in
deer-flow/backend/build-logs/q4-diag.log and bench-b1-run1.log:
  "An executable named `docling-mcp` is not provided by package
   `docling-mcp`. ... Use `uvx --from docling-mcp docling-mcp-server` instead."
  "An executable named `openbb-mcp-server` is not provided by package
   `openbb-mcp-server`. ... Use `uvx --from openbb-mcp-server openbb-mcp` instead."

Two things are checked: connectors.yaml (the documentation/registry source of
truth) records the corrected executable name, and extensions_config.json (the
file DeerFlow's MCP loader actually launches from -- see
deerflow/mcp/tools.py) invokes uvx with the matching `--from <package>
<executable>` form. A bare `uvx docling-mcp` / `uvx openbb-mcp-server` can
never work for either package, so this pins the two-argument form, not just
the corrected name string."""

from __future__ import annotations

import json
from pathlib import Path

from dr_core.connectors.registry import by_name, load_connectors

# extensions_config.json lives at the deer-flow repo root:
# .../deer-flow/backend/packages/dr_core/tests/this_file.py -> .../deer-flow/extensions_config.json
# (parents[0]=tests, [1]=dr_core, [2]=packages, [3]=backend, [4]=deer-flow)
_EXTENSIONS_CONFIG_PATH = Path(__file__).resolve().parents[4] / "extensions_config.json"


class TestConnectorsYamlExecutableNames:
    def test_docling_endpoint_is_the_real_console_script(self):
        idx = by_name(load_connectors())
        assert idx["docling"].endpoint == "docling-mcp-server"

    def test_openbb_endpoint_is_the_real_console_script(self):
        idx = by_name(load_connectors())
        assert idx["openbb"].endpoint == "openbb-mcp"


class TestExtensionsConfigJsonLaunchesTheRealExecutable:
    def _load(self) -> dict:
        assert _EXTENSIONS_CONFIG_PATH.is_file(), f"expected extensions_config.json at {_EXTENSIONS_CONFIG_PATH}"
        with open(_EXTENSIONS_CONFIG_PATH) as f:
            return json.load(f)

    def test_docling_invokes_uvx_from_docling_mcp_docling_mcp_server(self):
        servers = self._load()["mcpServers"]
        docling = servers["docling"]
        assert docling["command"] == "uvx"
        assert docling["args"] == [
            "--from",
            "docling-mcp",
            "docling-mcp-server",
            "--transport",
            "stdio",
        ]

    def test_openbb_invokes_uvx_from_openbb_mcp_server_openbb_mcp(self):
        servers = self._load()["mcpServers"]
        openbb = servers["openbb"]
        assert openbb["command"] == "uvx"
        assert openbb["args"] == [
            "--from",
            "openbb-mcp-server",
            "openbb-mcp",
            "--transport",
            "stdio",
        ]

    def test_stdio_servers_explicitly_select_stdio_transport(self):
        """Both packages default to HTTP despite DeerFlow declaring stdio."""
        servers = self._load()["mcpServers"]
        for name in ("docling", "openbb"):
            server = servers[name]
            assert server["type"] == "stdio"
            assert server["args"][-2:] == ["--transport", "stdio"]

    def test_docling_uses_neutral_working_directory(self):
        """Docling rejects unrelated keys in DeerFlow's backend .env file."""
        docling = self._load()["mcpServers"]["docling"]
        assert docling["cwd"] == "/tmp"

    def test_docling_and_openbb_args_are_not_the_old_bare_broken_form(self):
        """Regression guard: a bare single-element args list (the pre-fix
        shape) is exactly what produced the "executable not provided by
        package" failure in build-logs/q4-diag.log -- pin that it can't
        silently regress back to that shape."""
        servers = self._load()["mcpServers"]
        assert servers["docling"]["args"] != ["docling-mcp"]
        assert servers["openbb"]["args"] != ["openbb-mcp-server"]

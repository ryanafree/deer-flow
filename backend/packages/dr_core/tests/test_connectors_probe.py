"""Tests for dr_core.connectors.probe (S9). All HTTP is stubbed -- no real network calls."""

import socket
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from dr_core.connectors.probe import build_probe_request, probe_all, probe_connector
from dr_core.connectors.registry import Connector


def _connector(**overrides) -> Connector:
    defaults = dict(
        name="test_connector",
        type="unstructured",
        tier=0,
        profiles="all",
        access="rest",
        auth="none",
        launcher="none",
        durability="stable-api",
        rate_limit="free",
        last_verified="2026-06-23",
        fallback=None,
        endpoint="https://example.test",
        rest_endpoint="https://example.test",
        probe_url="https://example.test/probe",
        probe_auth=None,
    )
    defaults.update(overrides)
    return Connector(**defaults)


class TestBuildProbeRequest:
    def test_no_probe_auth_returns_url_unchanged(self):
        row = _connector(probe_url="https://example.test/x")
        url, headers = build_probe_request(row, env={})
        assert url == "https://example.test/x"
        assert headers == {}

    def test_query_auth_appends_param(self):
        row = _connector(auth="API_KEY", probe_url="https://example.test/x", probe_auth="query:api_key")
        url, headers = build_probe_request(row, env={"API_KEY": "secret"})
        assert url == "https://example.test/x?api_key=secret"
        assert headers == {}

    def test_query_auth_with_existing_query_string_uses_ampersand(self):
        row = _connector(auth="API_KEY", probe_url="https://example.test/x?foo=1", probe_auth="query:api_key")
        url, _ = build_probe_request(row, env={"API_KEY": "secret"})
        assert url == "https://example.test/x?foo=1&api_key=secret"

    def test_header_auth(self):
        row = _connector(auth="TOK", probe_url="https://example.test/x", probe_auth="header:X-Api-Key")
        _, headers = build_probe_request(row, env={"TOK": "abc"})
        assert headers == {"X-Api-Key": "abc"}

    def test_bearer_auth(self):
        row = _connector(auth="TOK", probe_url="https://example.test/x", probe_auth="bearer")
        _, headers = build_probe_request(row, env={"TOK": "abc"})
        assert headers == {"Authorization": "Bearer abc"}

    def test_token_auth(self):
        row = _connector(auth="TOK", probe_url="https://example.test/x", probe_auth="token")
        _, headers = build_probe_request(row, env={"TOK": "abc"})
        assert headers == {"Authorization": "Token abc"}

    def test_auth_scheme_present_but_env_empty_leaves_url_unchanged(self):
        row = _connector(auth="TOK", probe_url="https://example.test/x", probe_auth="bearer")
        url, headers = build_probe_request(row, env={})
        assert url == "https://example.test/x"
        assert headers == {}


class TestProbeConnectorNeverRaises:
    def test_no_probe_url_returns_skip(self):
        row = _connector(probe_url=None, launcher="none")
        status, detail = probe_connector(row, env={})
        assert status == "SKIP"

    def test_timeout_returns_down_not_raise(self):
        row = _connector(probe_url="https://example.test/x")
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            status, detail = probe_connector(row, env={}, timeout=0.01)
        assert status == "DOWN"

    def test_dns_failure_returns_down_not_raise(self):
        row = _connector(probe_url="https://nonexistent.invalid/x")
        err = urllib.error.URLError("Name or service not known")
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=err):
            status, detail = probe_connector(row, env={})
        assert status == "DOWN"

    def test_ok_response(self):
        row = _connector(probe_url="https://example.test/x")
        cm = MagicMock()
        cm.__enter__.return_value.status = 200
        with patch("dr_core.connectors.probe.urllib.request.urlopen", return_value=cm):
            status, detail = probe_connector(row, env={})
        assert status == "OK"
        assert detail == "200"

    def test_rate_limit_429(self):
        row = _connector(probe_url="https://example.test/x")
        err = urllib.error.HTTPError("https://example.test/x", 429, "Too Many Requests", {}, None)
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=err):
            status, detail = probe_connector(row, env={})
        assert status == "RATE"

    def test_auth_failure_with_key_present_reports_auth(self):
        row = _connector(auth="API_KEY", probe_url="https://example.test/x")
        err = urllib.error.HTTPError("https://example.test/x", 401, "Unauthorized", {}, None)
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=err):
            status, detail = probe_connector(row, env={"API_KEY": "present"})
        assert status == "AUTH"

    def test_auth_failure_with_key_absent_reports_needs_key(self):
        row = _connector(auth="API_KEY", probe_url="https://example.test/x")
        err = urllib.error.HTTPError("https://example.test/x", 401, "Unauthorized", {}, None)
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=err):
            status, detail = probe_connector(row, env={})
        assert status == "NEEDS-KEY"

    def test_generic_exception_returns_down_not_raise(self):
        row = _connector(probe_url="https://example.test/x")
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=RuntimeError("boom")):
            status, detail = probe_connector(row, env={})
        assert status == "DOWN"


class TestProbeAllScoping:
    def test_default_scope_is_tier_0(self):
        connectors = [_connector(name="t0", tier=0), _connector(name="t1", tier=1)]
        with patch("dr_core.connectors.probe.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.status = 200
            results = probe_all(connectors)
        assert [r["name"] for r in results] == ["t0"]

    def test_tier_scope(self):
        connectors = [_connector(name="t0", tier=0), _connector(name="t1", tier=1)]
        with patch("dr_core.connectors.probe.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.status = 200
            results = probe_all(connectors, tier=1)
        assert [r["name"] for r in results] == ["t1"]

    def test_profile_scope_includes_tier_0_plus_profile_matches(self):
        connectors = [
            _connector(name="t0", tier=0, profiles="all"),
            _connector(name="legal-only", tier=1, profiles=["legal"]),
            _connector(name="financial-only", tier=1, profiles=["financial"]),
        ]
        with patch("dr_core.connectors.probe.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.status = 200
            results = probe_all(connectors, profile="legal")
        names = {r["name"] for r in results}
        assert names == {"t0", "legal-only"}

    def test_all_scope(self):
        connectors = [_connector(name="t0", tier=0), _connector(name="t2", tier=2)]
        with patch("dr_core.connectors.probe.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.status = 200
            results = probe_all(connectors, do_all=True)
        assert {r["name"] for r in results} == {"t0", "t2"}


@pytest.mark.parametrize("cli_flag", ["--json", None])
def test_main_exits_nonzero_on_down(cli_flag, capsys):
    from dr_core.connectors import probe as probe_mod

    argv = ["--tier", "0"] if cli_flag is None else ["--tier", "0", cli_flag]
    with patch.object(probe_mod, "load_connectors", return_value=[_connector(tier=0)]):
        with patch("dr_core.connectors.probe.urllib.request.urlopen", side_effect=RuntimeError("boom")):
            code = probe_mod.main(argv)
    assert code == 1

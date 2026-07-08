"""Tests for dr_core.connectors.registry (S9)."""

import pytest
from dr_core.connectors.registry import ConnectorValidationError, by_name, load_connectors


def _write_yaml(tmp_path, rows):
    import yaml

    path = tmp_path / "connectors.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"connectors": rows}, f)
    return str(path)


def _good_row(**overrides):
    row = dict(
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
    )
    row.update(overrides)
    return row


class TestLoadVendoredConnectors:
    def test_vendored_connectors_yaml_parses_and_validates(self):
        connectors = load_connectors()
        assert len(connectors) >= 30
        names = {c.name for c in connectors}
        assert "tavily" in names
        assert "courtlistener" in names
        assert "edgar" in names

    def test_by_name_indexes_by_name(self):
        connectors = load_connectors()
        index = by_name(connectors)
        assert index["tavily"].tier == 0
        assert index["tavily"].profiles == "all"

    def test_govinfo_probe_auth_parsed(self):
        connectors = load_connectors()
        govinfo = by_name(connectors)["govinfo"]
        assert govinfo.probe_auth == "query:api_key"


class TestClosedEnumValidation:
    def test_good_row_loads(self, tmp_path):
        path = _write_yaml(tmp_path, [_good_row()])
        connectors = load_connectors(path)
        assert connectors[0].name == "test_connector"

    def test_sql_is_a_valid_access_value(self, tmp_path):
        path = _write_yaml(tmp_path, [_good_row(access="sql")])
        connectors = load_connectors(path)
        assert connectors[0].access == "sql"

    @pytest.mark.parametrize(
        "field,bad_value",
        [
            ("type", "not-a-type"),
            ("tier", 5),
            ("access", "carrier-pigeon"),
            ("durability", "vibes"),
            ("launcher", "telepathy"),
        ],
    )
    def test_unknown_enum_value_is_hard_error(self, tmp_path, field, bad_value):
        path = _write_yaml(tmp_path, [_good_row(**{field: bad_value})])
        with pytest.raises(ConnectorValidationError):
            load_connectors(path)

    def test_missing_required_field_is_hard_error(self, tmp_path):
        row = _good_row()
        del row["durability"]
        path = _write_yaml(tmp_path, [row])
        with pytest.raises(ConnectorValidationError):
            load_connectors(path)

    def test_one_bad_row_does_not_silently_drop_and_continue(self, tmp_path):
        """A single bad row among good ones still raises -- no silent partial load."""
        path = _write_yaml(tmp_path, [_good_row(name="good"), _good_row(name="bad", tier=99)])
        with pytest.raises(ConnectorValidationError) as excinfo:
            load_connectors(path)
        assert "bad" in str(excinfo.value)


class TestConnectorHelpers:
    def test_used_by_all_profile(self):
        connectors = load_connectors()
        tavily = by_name(connectors)["tavily"]
        assert tavily.used_by("legal")
        assert tavily.used_by("anything")

    def test_used_by_scoped_profile(self):
        connectors = load_connectors()
        edgar = by_name(connectors)["edgar"]
        assert edgar.used_by("financial")
        assert not edgar.used_by("legal")

    def test_has_rest_and_has_mcp(self):
        connectors = load_connectors()
        idx = by_name(connectors)
        assert idx["tavily"].has_rest()
        assert idx["tavily"].has_mcp()
        assert idx["jina_reader"].has_rest()
        assert not idx["jina_reader"].has_mcp()
        assert idx["docling"].has_mcp()
        assert not idx["docling"].has_rest()

    def test_wrds_is_sql_access_with_no_mcp_or_rest_fragment(self):
        connectors = load_connectors()
        wrds = by_name(connectors)["wrds"]
        assert wrds.access == "sql"
        assert wrds.tier == 1
        assert wrds.used_by("financial")
        assert not wrds.has_rest()
        assert not wrds.has_mcp()

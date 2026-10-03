"""Tests for dr_core.connectors.wrds_client (S9-C batch 1). All psycopg2 calls
are stubbed -- no real network/database connection in this file."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from dr_core.connectors import wrds_client


@pytest.fixture(autouse=True)
def _reset_conn():
    wrds_client.reset_connection()
    yield
    wrds_client.reset_connection()


class TestPeriodParsing:
    def test_year_only_yields_full_calendar_year(self):
        start, end = wrds_client._period_to_date_range("2023")
        assert start == "2023-01-01"
        assert end == "2023-12-31"

    def test_year_month_yields_that_month(self):
        start, end = wrds_client._period_to_date_range("2023-02")
        assert start == "2023-02-01"
        assert end == "2023-02-28"

    def test_year_month_handles_december_year_rollover(self):
        start, end = wrds_client._period_to_date_range("2023-12")
        assert start == "2023-12-01"
        assert end == "2023-12-31"

    def test_full_date_yields_a_window_around_it(self):
        start, end = wrds_client._period_to_date_range("2023-09-30")
        assert start < "2023-09-30" < end

    def test_unparseable_period_raises_value_error(self):
        with pytest.raises(ValueError):
            wrds_client._period_to_date_range("not-a-period")

    def test_fyear_extracts_leading_year(self):
        assert wrds_client._period_to_fyear("2023") == 2023
        assert wrds_client._period_to_fyear("2023-09-30") == 2023

    def test_fyear_unparseable_raises_value_error(self):
        with pytest.raises(ValueError):
            wrds_client._period_to_fyear("bogus")


class TestGetConnection:
    def test_missing_credentials_raises_wrds_unavailable(self, monkeypatch):
        monkeypatch.delenv("WRDS_USERNAME", raising=False)
        monkeypatch.delenv("WRDS_PASSWORD", raising=False)
        with pytest.raises(wrds_client.WrdsUnavailable):
            wrds_client.get_connection()

    def test_connect_failure_is_normalized_to_wrds_unavailable(self, monkeypatch):
        monkeypatch.setenv("WRDS_USERNAME", "u")
        monkeypatch.setenv("WRDS_PASSWORD", "p")
        with patch("dr_core.connectors.wrds_client.psycopg2.connect", side_effect=OSError("connection refused")):
            with pytest.raises(wrds_client.WrdsUnavailable):
                wrds_client.get_connection()

    def test_successful_connect_is_cached_across_calls(self, monkeypatch):
        monkeypatch.setenv("WRDS_USERNAME", "u")
        monkeypatch.setenv("WRDS_PASSWORD", "p")
        fake_conn = MagicMock(closed=False)
        with patch("dr_core.connectors.wrds_client.psycopg2.connect", return_value=fake_conn) as mock_connect:
            first = wrds_client.get_connection()
            second = wrds_client.get_connection()
        assert first is second
        mock_connect.assert_called_once()

    def test_reset_connection_forces_a_fresh_connect(self, monkeypatch):
        monkeypatch.setenv("WRDS_USERNAME", "u")
        monkeypatch.setenv("WRDS_PASSWORD", "p")
        fake_conn_1 = MagicMock(closed=False)
        fake_conn_2 = MagicMock(closed=False)
        with patch("dr_core.connectors.wrds_client.psycopg2.connect", side_effect=[fake_conn_1, fake_conn_2]):
            first = wrds_client.get_connection()
            wrds_client.reset_connection()
            second = wrds_client.get_connection()
        assert first is not second

    def test_credentials_never_appear_in_the_raised_error(self, monkeypatch):
        monkeypatch.setenv("WRDS_USERNAME", "supersecretuser")
        monkeypatch.setenv("WRDS_PASSWORD", "supersecretpass")
        with patch("dr_core.connectors.wrds_client.psycopg2.connect", side_effect=OSError("boom")):
            with pytest.raises(wrds_client.WrdsUnavailable) as exc_info:
                wrds_client.get_connection()
        assert "supersecretuser" not in str(exc_info.value)
        assert "supersecretpass" not in str(exc_info.value)


def _mock_conn(row: dict | None):
    conn = MagicMock()
    cursor_cm = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    cursor_cm.__enter__.return_value = cursor
    cursor_cm.__exit__.return_value = False
    conn.cursor.return_value = cursor_cm
    return conn, cursor


class TestFetchCrspPrice:
    def test_row_found_is_returned_as_plain_dict(self):
        row = {"permno": 14593, "date": "2023-12-29", "prc": 192.53, "ret": 0.0, "vol": 1000, "shrout": 15550061, "ticker": "AAPL", "comnam": "APPLE INC"}
        conn, cursor = _mock_conn(row)
        result = wrds_client.fetch_crsp_price("AAPL", "2023", conn=conn)
        assert result == row
        assert cursor.execute.call_args[0][1]["ticker"] == "AAPL"

    def test_no_row_returns_none_not_an_error(self):
        conn, _ = _mock_conn(None)
        assert wrds_client.fetch_crsp_price("ZZZZ", "2023", conn=conn) is None

    def test_ticker_is_uppercased(self):
        conn, cursor = _mock_conn(None)
        wrds_client.fetch_crsp_price("aapl", "2023", conn=conn)
        assert cursor.execute.call_args[0][1]["ticker"] == "AAPL"

    def test_query_failure_raises_wrds_query_error(self):
        conn = MagicMock()
        conn.cursor.side_effect = RuntimeError("server error")
        with pytest.raises(wrds_client.WrdsQueryError):
            wrds_client.fetch_crsp_price("AAPL", "2023", conn=conn)


class TestFetchCompustatFundamentals:
    def test_row_found_is_returned_as_plain_dict(self):
        row = {"gvkey": "001690", "tic": "AAPL", "conm": "APPLE INC", "datadate": "2023-09-30", "fyear": 2023, "at": 352583000000.0, "revt": 383285000000.0, "ni": 96995000000.0, "sale": 383285000000.0}
        conn, cursor = _mock_conn(row)
        result = wrds_client.fetch_compustat_fundamentals("AAPL", "2023", conn=conn)
        assert result == row
        assert cursor.execute.call_args[0][1]["fyear"] == 2023

    def test_no_row_returns_none_not_an_error(self):
        conn, _ = _mock_conn(None)
        assert wrds_client.fetch_compustat_fundamentals("ZZZZ", "2023", conn=conn) is None

    def test_standard_indfmt_filters_are_in_the_query(self):
        conn, cursor = _mock_conn(None)
        wrds_client.fetch_compustat_fundamentals("AAPL", "2023", conn=conn)
        sql = cursor.execute.call_args[0][0]
        assert "INDL" in sql and "STD" in sql and "'D'" in sql and "'C'" in sql

    def test_query_failure_raises_wrds_query_error(self):
        conn = MagicMock()
        conn.cursor.side_effect = RuntimeError("server error")
        with pytest.raises(wrds_client.WrdsQueryError):
            wrds_client.fetch_compustat_fundamentals("AAPL", "2023", conn=conn)


class TestMetricSelector:
    """P-09 item 1: the single-scalar metric selector. Pure functions over
    already-fetched rows -- no DB access in this class."""

    def test_vocabulary_is_exactly_the_mandated_five(self):
        assert set(wrds_client.WRDS_METRICS) == {"at", "revt", "ni", "sale", "price"}

    def test_deliberately_excluded_crsp_fields_are_not_selectable(self):
        # ret/vol/shrout carry unit semantics this layer cannot prove safe
        # (ret is a decimal fraction, shrout is in thousands, vol's Nasdaq
        # double-counting era) -- excluded rather than risk a false MISMATCH.
        for name in ("ret", "vol", "shrout"):
            assert name not in wrds_client.WRDS_METRICS

    def test_unknown_metric_raises_value_error(self):
        with pytest.raises(ValueError):
            wrds_client.select_metric_value("ebitda", crsp=None, compustat={"revt": 1.0})

    def test_compustat_millions_are_converted_to_base_usd(self):
        got = wrds_client.select_metric_value("revt", crsp=None, compustat={"revt": 383285.0})
        assert got == {"metric": "revt", "value": 383285.0 * 1e6, "unit": "USD"}

    def test_every_compustat_metric_converts_from_millions(self):
        row = {"at": 352583.0, "revt": 383285.0, "ni": 96995.0, "sale": 383285.0}
        for metric, millions in row.items():
            got = wrds_client.select_metric_value(metric, crsp=None, compustat=row)
            assert got["value"] == millions * 1e6
            assert got["unit"] == "USD"

    def test_crsp_price_is_usd_per_share_unconverted(self):
        got = wrds_client.select_metric_value("price", crsp={"prc": 192.53}, compustat=None)
        assert got == {"metric": "price", "value": 192.53, "unit": "USD"}

    def test_negative_crsp_price_is_normalized_to_its_magnitude(self):
        # CRSP encodes a bid/ask midpoint (no trade that day) as a NEGATIVE
        # prc; the magnitude is still the price estimate. Minting the signed
        # value would guarantee a false MISMATCH against any honest claim.
        got = wrds_client.select_metric_value("price", crsp={"prc": -192.53}, compustat=None)
        assert got["value"] == 192.53

    def test_missing_row_yields_none(self):
        assert wrds_client.select_metric_value("revt", crsp={"prc": 1.0}, compustat=None) is None
        assert wrds_client.select_metric_value("price", crsp=None, compustat={"revt": 1.0}) is None

    def test_null_field_in_a_present_row_yields_none(self):
        assert wrds_client.select_metric_value("revt", crsp=None, compustat={"fyear": 2023, "revt": None}) is None

    def test_unparseable_field_yields_none_never_raises(self):
        assert wrds_client.select_metric_value("revt", crsp=None, compustat={"revt": "n/a"}) is None

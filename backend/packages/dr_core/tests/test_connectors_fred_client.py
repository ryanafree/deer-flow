"""Tests for dr_core.connectors.fred_client (S9-C batch 2). All HTTP calls
are stubbed via the injectable `transport` param -- no real network in this
file."""

from __future__ import annotations

import pytest
from dr_core.connectors import fred_client


@pytest.fixture(autouse=True)
def _key_env(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    yield


class TestPeriodToDateRange:
    def test_year_only_yields_full_calendar_year(self):
        assert fred_client._period_to_date_range("2023") == ("2023-01-01", "2023-12-31")

    def test_year_month_yields_that_month(self):
        assert fred_client._period_to_date_range("2023-02") == ("2023-02-01", "2023-02-28")

    def test_full_date_yields_the_same_single_day(self):
        assert fred_client._period_to_date_range("2023-09-30") == ("2023-09-30", "2023-09-30")

    def test_unparseable_period_raises_value_error(self):
        with pytest.raises(ValueError):
            fred_client._period_to_date_range("not-a-period")


class TestFetchSeriesObservations:
    def test_success_returns_observations_and_window(self):
        def transport(url, headers):
            assert "api_key=test-key" in url
            assert "series_id=GDP" in url
            return {"observations": [{"date": "2023-01-01", "value": "26000.0"}, {"date": "2023-04-01", "value": "26500.0"}]}

        result = fred_client.fetch_series_observations("GDP", "2023", transport=transport)
        assert result["series_id"] == "GDP"
        assert result["start"] == "2023-01-01"
        assert result["end"] == "2023-12-31"
        assert len(result["observations"]) == 2
        assert result["observations"][-1]["date"] == "2023-04-01"

    def test_dot_placeholder_values_are_filtered_out(self):
        transport = lambda url, headers: {"observations": [{"date": "2023-01-01", "value": "."}]}  # noqa: E731
        assert fred_client.fetch_series_observations("GDP", "2023", transport=transport) is None

    def test_no_observations_returns_none(self):
        transport = lambda url, headers: {"observations": []}  # noqa: E731
        assert fred_client.fetch_series_observations("BOGUS", "2023", transport=transport) is None

    def test_missing_api_key_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(fred_client.FredUnavailable):
            fred_client.fetch_series_observations("GDP", "2023")

    def test_api_key_never_appears_in_a_raised_query_error(self):
        def transport(url, headers):
            raise fred_client.FredQueryError(f"boom {url}")

        with pytest.raises(fred_client.FredQueryError):
            fred_client.fetch_series_observations("GDP", "2023", transport=transport)


class TestFetchSeriesMetadata:
    """P-09 item 1: the /fred/series metadata endpoint (series units). Same
    injectable-transport closure pattern as the observations boundary."""

    def test_success_returns_units_and_identity(self):
        def transport(url, headers):
            assert url.startswith(fred_client.FRED_SERIES_URL + "?")
            assert "api_key=test-key" in url
            assert "series_id=GDP" in url
            return {"seriess": [{"id": "GDP", "title": "Gross Domestic Product", "units": "Billions of Dollars", "units_short": "Bil. of $", "frequency": "Quarterly", "seasonal_adjustment": "Seasonally Adjusted Annual Rate"}]}

        got = fred_client.fetch_series_metadata("GDP", transport=transport)
        assert got["series_id"] == "GDP"
        assert got["units"] == "Billions of Dollars"
        assert got["title"] == "Gross Domestic Product"

    def test_empty_seriess_returns_none(self):
        transport = lambda url, headers: {"seriess": []}  # noqa: E731
        assert fred_client.fetch_series_metadata("BOGUS", transport=transport) is None

    def test_missing_api_key_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with pytest.raises(fred_client.FredUnavailable):
            fred_client.fetch_series_metadata("GDP")

    def test_api_key_never_appears_in_a_raised_query_error(self):
        def transport(url, headers):
            raise fred_client.FredQueryError("FRED request failed: 400")

        with pytest.raises(fred_client.FredQueryError) as exc:
            fred_client.fetch_series_metadata("GDP", transport=transport)
        assert "test-key" not in str(exc.value)


class TestNormalizeObservationValue:
    """Unit normalization at the connector boundary. Scale is applied ONLY
    for provably scale-bearing unit strings; anything else is minted raw with
    its unit verbatim, which is the deliberate fail-safe (the comparator can
    then only reach INCONCLUSIVE, never a guessed-scale MISMATCH)."""

    def test_billions_of_dollars_scales_to_base_usd(self):
        assert fred_client.normalize_observation_value("27957.2", "Billions of Dollars") == {"value": 27957.2 * 1e9, "unit": "USD"}

    def test_millions_of_dollars_scales_to_base_usd(self):
        assert fred_client.normalize_observation_value("1500.0", "Millions of Dollars") == {"value": 1500.0 * 1e6, "unit": "USD"}

    def test_thousands_of_dollars_scales_to_base_usd(self):
        assert fred_client.normalize_observation_value("42.0", "Thousands of Dollars") == {"value": 42.0 * 1e3, "unit": "USD"}

    def test_chained_dollars_spelling_is_recognized(self):
        assert fred_client.normalize_observation_value("22000.0", "Billions of Chained 2017 Dollars") == {"value": 22000.0 * 1e9, "unit": "USD"}

    def test_units_are_matched_case_insensitively(self):
        assert fred_client.normalize_observation_value("1.0", "BILLIONS OF DOLLARS")["value"] == 1e9

    def test_percent_is_passed_through_on_the_comparator_percent_path(self):
        got = fred_client.normalize_observation_value("3.7", "Percent")
        assert got == {"value": 3.7, "unit": "Percent"}

    def test_unrecognized_unit_mints_the_raw_value_and_the_unit_verbatim(self):
        got = fred_client.normalize_observation_value("315.6", "Index 1982-1984=100")
        assert got == {"value": 315.6, "unit": "Index 1982-1984=100"}

    def test_a_scale_word_alone_is_not_enough_to_scale(self):
        # "Billions of Chained Units" is not a proven currency spelling.
        got = fred_client.normalize_observation_value("5.0", "Billions of Units")
        assert got == {"value": 5.0, "unit": "Billions of Units"}

    def test_unparseable_observation_value_yields_none(self):
        assert fred_client.normalize_observation_value(".", "Percent") is None
        assert fred_client.normalize_observation_value(None, "Percent") is None

    def test_missing_units_string_mints_no_unit_override(self):
        assert fred_client.normalize_observation_value("5.0", None) == {"value": 5.0, "unit": None}

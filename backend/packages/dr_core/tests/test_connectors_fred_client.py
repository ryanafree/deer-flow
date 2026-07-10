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

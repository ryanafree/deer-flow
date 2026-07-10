"""Tests for dr_core.connectors.edgar_client (S9-C batch 2). All HTTP calls
are stubbed via the injectable `transport` param -- no real network in this
file."""

from __future__ import annotations

import pytest
from dr_core.connectors import edgar_client


@pytest.fixture(autouse=True)
def _ua_env(monkeypatch):
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "test-agent test@example.com")
    yield


class TestResolveCik:
    def test_numeric_input_is_treated_as_cik_no_lookup(self):
        cik10, name = edgar_client.resolve_cik("320193")

        assert cik10 == "0000320193"
        assert name is None

    def test_ticker_resolves_via_company_tickers_lookup(self):
        def transport(url, headers):
            assert url == edgar_client.COMPANY_TICKERS_URL
            assert headers["User-Agent"] == "test-agent test@example.com"
            return {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

        cik10, name = edgar_client.resolve_cik("aapl", transport=transport)
        assert cik10 == "0000320193"
        assert name == "Apple Inc."

    def test_unknown_ticker_raises_query_error(self):
        transport = lambda url, headers: {"0": {"cik_str": 1, "ticker": "ZZZZ", "title": "Nope"}}  # noqa: E731
        with pytest.raises(edgar_client.EdgarQueryError):
            edgar_client.resolve_cik("AAPL", transport=transport)

    def test_missing_user_agent_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
        with pytest.raises(edgar_client.EdgarUnavailable):
            edgar_client.resolve_cik("AAPL")


class TestFetchCompanyConcept:
    def _companyconcept_payload(self):
        return {
            "label": "Revenues",
            "entityName": "Apple Inc.",
            "units": {
                "USD": [
                    # Quarterly figure -- should be skipped (span < 300 days).
                    {"start": "2023-07-01", "end": "2023-09-30", "val": 89498000000, "fy": 2023, "fp": "Q4", "form": "10-Q", "filed": "2023-11-02", "accn": "0000320193-23-000106"},
                    # Annual FY2023 figure from the FY2023 10-K.
                    {"start": "2022-09-25", "end": "2023-09-30", "val": 383285000000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "accn": "0000320193-23-000106"},
                    # Prior-year annual figure carried in the SAME 10-K (fy field would
                    # mislead if used for disambiguation -- end date is what matters).
                    {"start": "2021-09-26", "end": "2022-09-24", "val": 394328000000, "fy": 2023, "fp": "FY", "form": "10-K", "filed": "2023-11-03", "accn": "0000320193-23-000106"},
                ]
            },
        }

    def test_selects_the_annual_fact_matching_period_by_end_date(self):
        transport = lambda url, headers: self._companyconcept_payload()  # noqa: E731
        fact = edgar_client.fetch_company_concept("0000320193", "Revenues", "2023", transport=transport)

        assert fact is not None
        assert fact["value"] == 383285000000
        assert fact["end"] == "2023-09-30"
        assert fact["form"] == "10-K"

    def test_disambiguates_by_end_date_not_filing_fy_field(self):
        """Both 10-K rows share fy=2023 (the FILING's fiscal year) but have
        different end dates -- period="2022" must resolve to the PRIOR-year
        figure by its own end date, not the filing's fy."""
        transport = lambda url, headers: self._companyconcept_payload()  # noqa: E731
        fact = edgar_client.fetch_company_concept("0000320193", "Revenues", "2022", transport=transport)

        assert fact is not None
        assert fact["value"] == 394328000000
        assert fact["end"] == "2022-09-24"

    def test_no_matching_year_returns_none(self):
        transport = lambda url, headers: self._companyconcept_payload()  # noqa: E731
        assert edgar_client.fetch_company_concept("0000320193", "Revenues", "2019", transport=transport) is None

    def test_quarterly_only_data_returns_none(self):
        data = {"units": {"USD": [{"start": "2023-07-01", "end": "2023-09-30", "val": 1, "form": "10-Q", "filed": "2023-11-02"}]}}
        transport = lambda url, headers: data  # noqa: E731
        assert edgar_client.fetch_company_concept("0000320193", "Revenues", "2023", transport=transport) is None

    def test_bad_period_raises_value_error(self):
        transport = lambda url, headers: self._companyconcept_payload()  # noqa: E731
        with pytest.raises(ValueError):
            edgar_client.fetch_company_concept("0000320193", "Revenues", "not-a-period", transport=transport)

    def test_missing_user_agent_raises_unavailable(self, monkeypatch):
        monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
        with pytest.raises(edgar_client.EdgarUnavailable):
            edgar_client.fetch_company_concept("0000320193", "Revenues", "2023")

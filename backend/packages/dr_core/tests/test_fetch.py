"""Offline smoke tests for dr_core.fetch.structured (Phase 1 port of
DeepResearch's harness/data_fetch.py; the original had no companion test —
see build-logs/task-port-data-fetch.md).

No network calls: urllib entry points (_get_json / _post_form) are monkeypatched
with small fixtures. Covers argparse wiring, env() secret-non-leak, provenance
scrubbing (no api_key in returned `query`), and response parsing (EDGAR fiscal-
year disambiguation + dedup, FRED observation filtering, CourtListener 429/cite
normalization).

Run: cd deer-flow/backend && PYTHONPATH=packages/harness uv run pytest packages/dr_core/tests/test_fetch.py -v
"""

from __future__ import annotations

import urllib.error

import pytest
from dr_core.fetch import structured

# ---------------------------------------------------------------------------
# env() — secret handling
# ---------------------------------------------------------------------------


def test_env_reads_value_without_printing(tmp_path, monkeypatch, capsys):
    env_file = tmp_path / ".env"
    env_file.write_text('MY_SECRET_KEY="sekrit-value-123"\n')
    monkeypatch.setattr(structured, "ENV_PATH", str(env_file))

    val = structured.env("MY_SECRET_KEY")

    assert val == "sekrit-value-123"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_env_missing_key_returns_empty(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\n")
    monkeypatch.setattr(structured, "ENV_PATH", str(env_file))

    assert structured.env("MISSING_KEY") == ""


def test_env_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(structured, "ENV_PATH", str(tmp_path / "nonexistent.env"))

    assert structured.env("ANYTHING") == ""


def test_missing_credential_fails_closed_without_network_call(monkeypatch):
    """No credential -> ok=False, and no attempt to reach the network."""
    monkeypatch.setattr(structured, "env", lambda key: "")

    def _boom(*a, **kw):
        raise AssertionError("must not call the network without a credential")

    monkeypatch.setattr(structured, "_get_json", _boom)
    monkeypatch.setattr(structured, "_post_form", _boom)

    assert structured.fetch_edgar(ticker="AAPL") == {"ok": False, "error": "SEC_EDGAR_USER_AGENT not set in ~/.claude/.env"}
    assert structured.fetch_fred(series="UNRATE") == {"ok": False, "error": "FRED_API_KEY not set in ~/.claude/.env"}
    assert structured.fetch_courtlistener(text="foo") == {"ok": False, "error": "COURTLISTENER_TOKEN not set in ~/.claude/.env"}


# ---------------------------------------------------------------------------
# EDGAR — response parsing + fiscal-year disambiguation + dedup
# ---------------------------------------------------------------------------

_TICKERS_FIXTURE = {"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}}

_FACTS_FIXTURE = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        # annual 10-K figure for FY2023 (end-date year disambiguates FY, not the `fy` field)
                        {"form": "10-K", "fp": "FY", "start": "2023-01-01", "end": "2023-12-31", "val": 1000, "fy": 2024, "accn": "acc-old", "filed": "2024-01-01"},
                        # same period re-reported in a later filing -> dedup should keep this one (newest filed)
                        {"form": "10-K", "fp": "FY", "start": "2023-01-01", "end": "2023-12-31", "val": 1000, "fy": 2024, "accn": "acc-new", "filed": "2024-06-01"},
                        # a 10-Q (partial period) -> excluded by the form filter
                        {"form": "10-Q", "fp": "Q1", "start": "2024-01-01", "end": "2024-03-31", "val": 200, "fy": 2024, "accn": "acc-q", "filed": "2024-04-01"},
                    ]
                },
            }
        }
    }
}


def _fake_get_json(tickers=_TICKERS_FIXTURE, facts=_FACTS_FIXTURE):
    def _fake(url, headers=None):
        if "company_tickers.json" in url:
            return tickers
        if "companyfacts" in url:
            return facts
        raise AssertionError(f"unexpected url: {url}")

    return _fake


def test_fetch_edgar_parses_annual_facts_and_dedupes(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent (test@example.com)")
    monkeypatch.setattr(structured, "_get_json", _fake_get_json())

    result = structured.fetch_edgar(ticker="aapl", concept_keyword="revenue", fy=2023)

    assert result["ok"] is True
    assert result["source_system"] == "edgar"
    assert result["cik"] == "0000320193"
    assert result["ticker"] == "AAPL"
    # 10-Q excluded by form filter, and the two duplicate 10-K periods collapse to one (newest filed kept)
    assert len(result["matches"]) == 1
    assert result["matches"][0]["accn"] == "acc-new"
    assert result["matches"][0]["value"] == 1000


def test_fetch_edgar_fiscal_year_disambiguated_by_end_date_not_fy_field(monkeypatch):
    """The SEC `fy` field (2024, the filing's fiscal year) must NOT be used to filter;
    only the data point's END date year (2023) determines its fiscal year."""
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")
    monkeypatch.setattr(structured, "_get_json", _fake_get_json())

    matched_2023 = structured.fetch_edgar(ticker="AAPL", fy=2023)
    matched_2024 = structured.fetch_edgar(ticker="AAPL", fy=2024)

    assert matched_2023["ok"] is True and len(matched_2023["matches"]) == 1
    assert matched_2024["ok"] is False  # no data point actually ends in 2024


def test_fetch_edgar_ticker_not_found(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")
    monkeypatch.setattr(structured, "_get_json", _fake_get_json())

    result = structured.fetch_edgar(ticker="NOPE")

    assert result == {"ok": False, "error": "ticker NOPE not found in SEC company_tickers"}


def test_fetch_edgar_http_error_from_company_tickers(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")

    def _raise(url, headers=None):
        raise urllib.error.HTTPError(url, 503, "unavailable", hdrs=None, fp=None)

    monkeypatch.setattr(structured, "_get_json", _raise)

    result = structured.fetch_edgar(ticker="AAPL")

    assert result == {"ok": False, "error": "company_tickers fetch failed: 503"}


# ---------------------------------------------------------------------------
# B6 fault injection — a hung/unreachable source can never stall or crash a run
# ---------------------------------------------------------------------------


def test_fetch_edgar_url_error_from_company_tickers_fails_closed(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")

    def _raise(url, headers=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(structured, "_get_json", _raise)

    result = structured.fetch_edgar(ticker="AAPL")

    assert result["ok"] is False
    assert "company_tickers fetch failed" in result["error"]


def test_fetch_edgar_socket_timeout_from_companyfacts_fails_closed(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")

    def _flaky(url, headers=None):
        if "companyfacts" in url:
            raise TimeoutError("timed out")
        return _TICKERS_FIXTURE

    monkeypatch.setattr(structured, "_get_json", _flaky)

    result = structured.fetch_edgar(ticker="AAPL")

    assert result["ok"] is False
    assert "companyfacts fetch failed" in result["error"]


def test_fetch_fred_dns_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "k")

    def _raise(url, headers=None):
        raise urllib.error.URLError(OSError("nodename nor servname provided"))

    monkeypatch.setattr(structured, "_get_json", _raise)

    result = structured.fetch_fred(series="UNRATE")

    assert result == {"ok": False, "error": "FRED fetch failed: <urlopen error nodename nor servname provided>", "series": "UNRATE"}


def test_fetch_courtlistener_url_error_fails_closed(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-token")

    def _raise(url, fields, headers=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(structured, "_post_form", _raise)

    result = structured.fetch_courtlistener(text="foo")

    assert result["ok"] is False
    assert "citation-lookup failed" in result["error"]


def test_cli_fred_url_error_still_prints_ok_false_and_exits_zero(monkeypatch, capsys):
    """The always-exits-0 contract must hold for a genuine network exception, not just
    a handled ok=false return -- this drives the exception through main(), unlike
    test_cli_prints_error_record_and_still_exits_zero which mocks the fetch fn itself."""
    monkeypatch.setattr(structured, "env", lambda key: "k")

    def _raise(url, headers=None):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(structured, "_get_json", _raise)
    parser = structured._build_parser()
    args = parser.parse_args(["fred", "--series", "UNRATE"])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 0
    assert '"ok": false' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# m2 — malformed dates in SEC data must not raise before the None-check
# ---------------------------------------------------------------------------


def test_fetch_edgar_malformed_end_date_does_not_raise(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-agent")
    facts_bad_date = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {
                        "USD": [
                            # end date has an invalid month/day; _day_ordinal returns None for it
                            {"form": "10-K", "fp": "FY", "start": "2026-01-01", "end": "2026-13-99", "val": 500, "fy": 2026, "accn": "acc-bad", "filed": "2026-06-01"},
                        ]
                    },
                }
            }
        }
    }
    monkeypatch.setattr(structured, "_get_json", _fake_get_json(facts=facts_bad_date))

    result = structured.fetch_edgar(ticker="AAPL", concept_keyword="revenue")

    assert result["ok"] is True
    assert result["matches"][0]["accn"] == "acc-bad"


# ---------------------------------------------------------------------------
# FRED — observation parsing + provenance scrubbing
# ---------------------------------------------------------------------------


def test_fetch_fred_filters_dot_observations_and_scrubs_key(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "super-secret-fred-key")
    captured_urls = []

    def _fake(url, headers=None):
        captured_urls.append(url)
        return {"observations": [
            {"date": "2020-01-01", "value": "3.5"},
            {"date": "2020-02-01", "value": "."},  # missing observation, filtered
        ]}

    monkeypatch.setattr(structured, "_get_json", _fake)

    result = structured.fetch_fred(series="UNRATE", year=2020)

    assert result["ok"] is True
    assert result["observations"] == [{"date": "2020-01-01", "value": "3.5"}]
    # the actual request URL legitimately carries the key; the returned provenance must not
    assert "super-secret-fred-key" in captured_urls[0]
    assert "super-secret-fred-key" not in result["query"]
    assert "api_key" not in result["query"]


def test_fetch_fred_latest_ignored_when_year_given(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "k")
    monkeypatch.setattr(structured, "_get_json", lambda url, headers=None: {
        "observations": [{"date": "2020-01-01", "value": "1"}, {"date": "2020-06-01", "value": "2"}],
    })

    result = structured.fetch_fred(series="UNRATE", year=2020, latest=True)

    assert len(result["observations"]) == 2  # latest only applies when year is None


def test_fetch_fred_latest_true_year_none_slices_to_last(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "k")
    monkeypatch.setattr(structured, "_get_json", lambda url, headers=None: {
        "observations": [{"date": "2020-01-01", "value": "1"}, {"date": "2020-06-01", "value": "2"}],
    })

    result = structured.fetch_fred(series="UNRATE", latest=True)

    assert result["observations"] == [{"date": "2020-06-01", "value": "2"}]


def test_fetch_fred_no_observations_fails(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "k")
    monkeypatch.setattr(structured, "_get_json", lambda url, headers=None: {"observations": []})

    result = structured.fetch_fred(series="UNRATE")

    assert result == {"ok": False, "error": "no observations returned", "series": "UNRATE"}


# ---------------------------------------------------------------------------
# CourtListener — citation normalization + 429 handling
# ---------------------------------------------------------------------------


def test_fetch_courtlistener_normalizes_citations(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-token")
    monkeypatch.setattr(structured, "_post_form", lambda url, fields, headers=None: [
        {
            "citation": "410 U.S. 113",
            "normalized_citations": ["410 U.S. 113"],
            "status": 200,
            "clusters": [{"case_name": "Roe v. Wade", "absolute_url": "/opinion/108713/roe-v-wade/"}],
            "start_index": 0,
            "end_index": 12,
        }
    ])

    result = structured.fetch_courtlistener(text="see 410 U.S. 113")

    assert result["ok"] is True
    cite = result["cites"][0]
    assert cite["case_name"] == "Roe v. Wade"
    assert cite["absolute_url"] == "https://www.courtlistener.com/opinion/108713/roe-v-wade/"
    assert cite["clusters_count"] == 1


def test_fetch_courtlistener_reads_textfile(tmp_path, monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-token")
    monkeypatch.setattr(structured, "_post_form", lambda url, fields, headers=None: [])
    textfile = tmp_path / "brief.txt"
    textfile.write_text("some legal text")

    result = structured.fetch_courtlistener(textfile=str(textfile))

    assert result == {"ok": True, "source_system": "courtlistener", "query": "citation-lookup (Eyecite) over submitted text", "cites": []}


def test_fetch_courtlistener_no_text_fails(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-token")

    result = structured.fetch_courtlistener()

    assert result == {"ok": False, "error": "no --text or --textfile supplied"}


def test_fetch_courtlistener_rate_limited(monkeypatch):
    monkeypatch.setattr(structured, "env", lambda key: "test-token")

    def _raise(url, fields, headers=None):
        raise urllib.error.HTTPError(url, 429, "too many requests", hdrs=None, fp=None)

    monkeypatch.setattr(structured, "_post_form", _raise)

    result = structured.fetch_courtlistener(text="foo")

    assert result == {"ok": False, "error": "rate-limited (429): CourtListener daily/throttle cap reached", "rate_limited": True}


# ---------------------------------------------------------------------------
# argparse wiring + CLI parity wrapper
# ---------------------------------------------------------------------------


def test_parser_defaults_for_each_subcommand():
    parser = structured._build_parser()

    edgar_args = parser.parse_args(["edgar", "--ticker", "AAPL"])
    assert edgar_args.func is structured._cli_edgar
    assert edgar_args.concept is None
    assert edgar_args.concept_keyword == "revenue"
    assert edgar_args.fy is None
    assert edgar_args.unit == "USD"
    assert edgar_args.max == 8

    fred_args = parser.parse_args(["fred", "--series", "UNRATE"])
    assert fred_args.func is structured._cli_fred
    assert fred_args.year is None
    assert fred_args.latest is False

    cl_args = parser.parse_args(["courtlistener", "--text", "hi"])
    assert cl_args.func is structured._cli_courtlistener
    assert cl_args.textfile is None


def test_parser_requires_a_subcommand():
    parser = structured._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_edgar_prints_json_and_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(structured, "fetch_edgar", lambda **kw: {"ok": True, "ticker": kw["ticker"]})
    parser = structured._build_parser()
    args = parser.parse_args(["edgar", "--ticker", "AAPL"])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert '"ticker": "AAPL"' in out


def test_cli_prints_error_record_and_still_exits_zero(monkeypatch, capsys):
    """Even a failed fetch (ok=false) must exit 0 -- the caller reads the JSON, not the shell code."""
    monkeypatch.setattr(structured, "fetch_fred", lambda **kw: {"ok": False, "error": "boom"})
    parser = structured._build_parser()
    args = parser.parse_args(["fred", "--series", "UNRATE"])

    with pytest.raises(SystemExit) as exc_info:
        args.func(args)

    assert exc_info.value.code == 0
    assert '"ok": false' in capsys.readouterr().out

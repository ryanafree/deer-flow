# S1 live acceptance — WRDS/FRED data_ref value capture (2026-08-29)

Stage contract: `build-logs/STAGE_CONTRACT_wrdsfred_selector_2026-08-29.md`, process step 2.
Executor: Opus 5 (medium). Repo HEAD `0ae1e08d`, branch `deerflow-dr`. Nothing committed.

Credentials read from `backend/.env` via `python-dotenv`. All four probes are read-only and
bounded (one WRDS connection, three CRSP/Compustat rows via a single `wrds_query` call each;
two FRED series, one observations call plus one `/fred/series` metadata call each). Probe
output is scrubbed of `FRED_API_KEY` / `WRDS_USERNAME` / `WRDS_PASSWORD` before being written
here, and re-verified absent after the fact.

## Outcome summary

| Probe | What it exercises | Result |
|---|---|---|
| A | `wrds_query("AAPL", "2023", metric="revt")` → synthetic claim quoting the same figure | `Comparison.MATCH` → `DataProvenance.MATCHED` |
| B | `fred_series("GDP", "2023")` with `units="Billions of Dollars"` normalized to base USD → synthetic claim | `Comparison.MATCH` → `DataProvenance.MATCHED` |
| C | Deliberate wrong figure ($500.0 billion) against probe A's real record | `Comparison.MISMATCH` → `DataProvenance.MISMATCH` |
| D | Unrecognized-unit fail-safe: `fred_series("A191RL1Q225SBEA", "2023")`, units "Percent Change from Preceding Period" | `Comparison.INCONCLUSIVE` → `audit_provenance` returns `None` (claim stays UNAUDITED) |

Notes worth recording:

- Probe A's live Compustat `revt` came back as the string `"383285.0000"` (millions). The
  connector's `select_metric_value` converted it to `383285000000.0` base USD, which the
  comparator matched against a claim written as "$383.285 billion" — the millions-to-dollars
  conversion is exercised live, not only in the mocked tests.
- Probe B's live GDP figure is `28424.722` (2023-10-01), minted as `28424722000000.0` USD.
  The 2023 Q4 level has been revised upward since the mocked fixtures were written (27957.2);
  the probe builds its claim text from the returned value, so the check is on the live number.
- Probe D is the fail-safe demonstration the contract asks for: FRED's own unit string is not
  a provable scale, so the connector mints the raw value with the unit verbatim, and the
  percent-shaped claim lands on INCONCLUSIVE rather than a fabricated contradiction.
- The first WRDS attempt in this session hit `connection failed: timeout expired` at the
  module's 15 s `CONNECT_TIMEOUT_S`; a direct psycopg2 probe then connected in 10.6 s and the
  re-run succeeded. Transient WRDS latency, not a code path — recorded because it is close to
  the configured budget and may be worth raising later.

## Verbatim probe output

```text
  from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
== PROBE A: wrds_query(AAPL, 2023, metric='revt') ==
--- wrds revt ---
{
  "ok": true,
  "source_system": "wrds",
  "source_class": "primary_database",
  "url_or_id": "wrds://crsp-compustat/AAPL/2023",
  "title": "WRDS CRSP/Compustat AAPL 2023",
  "ticker": "AAPL",
  "period": "2023",
  "crsp": {
    "permno": 14593,
    "date": "2023-12-29",
    "prc": "192.53000",
    "ret": "-0.005424",
    "vol": "42120662",
    "shrout": 15460223.0,
    "ticker": "AAPL",
    "comnam": "APPLE INC"
  },
  "compustat": {
    "gvkey": "001690",
    "tic": "AAPL",
    "conm": "APPLE INC",
    "datadate": "2023-09-30",
    "fyear": 2023,
    "at": "352583.0000",
    "revt": "383285.0000",
    "ni": "96995.0000",
    "sale": "383285.0000"
  },
  "data_ref": {
    "period": "2023",
    "source_class": "primary_database",
    "metric": "revt",
    "value": 383285000000.0,
    "unit": "USD"
  },
  "metric": "revt",
  "value": 383285000000.0,
  "unit": "USD"
}
claim: Apple's FY2023 total revenue was $383.285 billion (Compustat revt).
comparison, provenance: (<Comparison.MATCH: 'match'>, <DataProvenance.MATCHED: 'matched'>)

== PROBE C: deliberate WRONG figure against the same record ==
claim: Apple's FY2023 total revenue was $500.0 billion.
comparison, provenance: (<Comparison.MISMATCH: 'mismatch'>, <DataProvenance.MISMATCH: 'mismatch'>)

== PROBE B: fred_series('GDP', '2023') ==
--- fred GDP ---
{
  "ok": true,
  "source_system": "fred",
  "source_class": "official_stat",
  "url_or_id": "https://api.stlouisfed.org/fred/series/observations?series_id=GDP&observation_start=2023-01-01&observation_end=2023-12-31&file_type=json",
  "title": "FRED GDP 2023",
  "series_id": "GDP",
  "latest_observation": {
    "date": "2023-10-01",
    "value": "28424.722"
  },
  "data_ref": {
    "period": "2023-10-01",
    "source_class": "official_stat",
    "series_id": "GDP",
    "value": 28424722000000.0,
    "unit": "USD"
  },
  "units": "Billions of Dollars",
  "series_title": "Gross Domestic Product",
  "observations_count": 4
}
claim: US GDP was $28.425 trillion as of 2023-10-01 per FRED.
comparison, provenance: (<Comparison.MATCH: 'match'>, <DataProvenance.MATCHED: 'matched'>)

== PROBE D: unrecognized-unit fail-safe -- fred_series('A191RL1Q225SBEA', '2023') ==
--- fred pct-change ---
{
  "ok": true,
  "source_system": "fred",
  "source_class": "official_stat",
  "url_or_id": "https://api.stlouisfed.org/fred/series/observations?series_id=A191RL1Q225SBEA&observation_start=2023-01-01&observation_end=2023-12-31&file_type=json",
  "title": "FRED A191RL1Q225SBEA 2023",
  "series_id": "A191RL1Q225SBEA",
  "latest_observation": {
    "date": "2023-10-01",
    "value": "3.4"
  },
  "data_ref": {
    "period": "2023-10-01",
    "source_class": "official_stat",
    "series_id": "A191RL1Q225SBEA",
    "value": 3.4,
    "unit": "Percent Change from Preceding Period"
  },
  "units": "Percent Change from Preceding Period",
  "series_title": "Real Gross Domestic Product",
  "observations_count": 4
}
claim: Real GDP grew 3.4% at an annual rate in Q4 2023.
comparison, provenance: (<Comparison.INCONCLUSIVE: 'inconclusive'>, None)
```

Probe script (as run): `/private/tmp/claude-501/-Users-ryanfree-Documents-Projects-deerflow-dr-deer-flow/8992072a-1d86-4083-8eed-fcdc9bf62e52/scratchpad/probe.py`, retained in the session scratchpad only.

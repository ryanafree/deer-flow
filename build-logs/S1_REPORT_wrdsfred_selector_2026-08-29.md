# S1 report — WRDS/FRED data_ref value capture (P-09 next-item 1)

Date 2026-08-29. Executor: Opus 5 at medium effort. Repo `deerflow-dr/deer-flow`, branch
`deerflow-dr`, HEAD `0ae1e08d`. Contract:
`build-logs/STAGE_CONTRACT_wrdsfred_selector_2026-08-29.md`. **Nothing committed; `.git`
untouched.** The ~18 pre-existing uncommitted entries from the 2026-08-27 session are
unchanged.

## Files changed

All under `backend/packages/dr_core/`, plus the two report files in `build-logs/`.

| File | Change | Lines |
|---|---|---|
| `dr_core/connectors/wrds_client.py` | New selector section: `WRDS_METRICS`, `COMPUSTAT_MILLIONS_TO_USD`, `USD_UNIT`, `select_metric_value()` | +62 at 179-240 |
| `dr_core/connectors/fred_client.py` | `FRED_SERIES_URL`; `fetch_series_metadata()`; unit-normalization section with `normalize_observation_value()` | +1 at 30, +66 at 99-164 |
| `dr_core/connectors/tools.py` | `wrds_query` gains `metric`; docstring echo language for `value`/`unit`/`metric`; data_ref fold-in; `fred_series` metadata fetch + docstring + data_ref fold-in | 89, 101-135, 159, 170-185, 270-325 |
| `tests/test_connectors_wrds_client.py` | New `TestMetricSelector` (10 tests) | +51 at 155-205 |
| `tests/test_connectors_fred_client.py` | New `TestFetchSeriesMetadata` (4) + `TestNormalizeObservationValue` (10) | +76 at 65-140 |
| `tests/test_connectors_tools.py` | `_mock_fred` defaults the metadata boundary off; new `TestWrdsMetricSelectorTool` (7) + `TestFredSeriesUnitMetadata` (5) | +211 at 33, 359-361, 596-802 |

## Test delta

Baseline 815 passed + 3 skipped → **852 passed + 3 skipped**, zero failures, skips unchanged.
37 new tests. Canonical suite only (`from backend/: .venv/bin/python -m pytest
packages/dr_core/`); `make test` never run. Verbatim tail:

```text
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
852 passed, 3 skipped, 1 warning in 4.41s
```

`ruff check` and `ruff format --check` clean on all six touched files (no reformatting needed).

## Live probes

Full output in `build-logs/wrdsfred_selector_live_acceptance_2026-08-29.md`. All four
demonstrated:

- **a. WRDS MATCH** — `wrds_query("AAPL","2023",metric="revt")` returned live Compustat
  `revt="383285.0000"` (millions), minted `value=383285000000.0 unit="USD" metric="revt"`; a
  claim reading "$383.285 billion" → `Comparison.MATCH` → `DataProvenance.MATCHED`.
- **b. FRED MATCH** — `fred_series("GDP","2023")`, units "Billions of Dollars", latest
  observation `28424.722` → `value=28424722000000.0 unit="USD"`; claim → MATCH → MATCHED.
- **c. Deliberate MISMATCH** — "$500.0 billion" against probe a's real record → MISMATCH →
  `DataProvenance.MISMATCH`. The comparator demonstrably sees the value.
- **d. Fail-safe** — `fred_series("A191RL1Q225SBEA","2023")`, units "Percent Change from
  Preceding Period" (no provable scale) → raw value, unit verbatim; a percent-shaped claim →
  `Comparison.INCONCLUSIVE`, `audit_provenance` returns `None` (UNAUDITED, not MISMATCH).

Credentials came from `backend/.env` via `python-dotenv`; the log was scrubbed of
`FRED_API_KEY`/`WRDS_USERNAME`/`WRDS_PASSWORD` and re-verified absent afterwards.

## Judgment calls inside the contract's latitude

1. **CRSP `ret`/`vol`/`shrout` excluded**, as settled decision 2 permits. `ret` is a decimal
   fraction (0.0123, not 1.23%), `shrout` is in thousands of shares, and `vol`'s Nasdaq
   double-counting convention is not carried in the row. None can be proven false-MISMATCH-safe
   here. Stated in the tool docstring and pinned by a test.
2. **Negative CRSP `prc` is minted as its magnitude.** CRSP encodes a bid/ask midpoint (no trade
   that day) as a negative price; minting the sign would guarantee a false MISMATCH against any
   honest claim. Documented and tested.
3. **A selected metric with no backing value returns `ok:false`,** not a value-less success. A
   value-less "success" invites the model to quote a figure from the compound record that the
   audit cannot then check. Unknown metric names are validated at the tool boundary *before* any
   DB call, so a typo costs nothing.
4. **Unit spellings** (settled decision 5): the comparator recognizes only percent aliases as an
   explicit unit; everything else falls to its default "count". So the connectors mint exactly
   two MATCHED-reachable spellings — `"USD"` (edgar_client's own XBRL spelling) for scale-bearing
   currency, and FRED's `"Percent"` verbatim, which is in `_PERCENT_UNIT_ALIASES`.
5. **FRED scale table is a strict regex**, `^(thousands|millions|billions|trillions) of
   (?:chained \d{4} )?dollars$`, case-insensitive. "Billions of Units" deliberately does NOT
   scale — a scale word alone is not proof of a currency series. Pinned by a test.
6. **FRED metadata failure is non-fatal**: `fred_series` falls back to the pre-P-09 value-less
   data_ref rather than failing the call. That path is INCONCLUSIVE by construction, so it cannot
   produce a wrong answer, and it keeps `fred_series` working if `/fred/series` is down.
7. **`_mock_fred` in `test_connectors_tools.py` now stubs the metadata boundary to `None` by
   default.** Without it, every pre-existing FRED tool test would have made a real network call.
   That also turns the two pre-existing FRED assertions into the metadata-unavailable regression:
   they still assert the legacy value-less data_ref, unchanged.

## One nuance worth the orchestrator's attention (not a deviation)

Settled decision 4 says an unrecognized FRED unit string "resolves to INCONCLUSIVE". That is
true for a differently-shaped claim (the percent-vs-count case probe d demonstrates), but not
universally: since the comparator maps any non-percent unit label to its default "count", a
claim that quotes the same unscaled number in the same shape still reaches MATCH. That is the
correct outcome — no scale was guessed on either side, so the figures are like-for-like — and
making it INCONCLUSIVE instead would have required changing `provenance_compare.py`, which
settled decision 1 forbids. The guarantee actually delivered is the one that matters: an
unrecognized unit can never produce a *scale-artifact* MISMATCH.

## Not done

Nothing in the contract was skipped. No `AGENTS.md`/`README.md` update was made: the repo's
documentation policy covers the upstream deer-flow modules, and `dr_core` connector detail is
carried by `BUILD_LEDGER.md`/`FORK_DELTA.md` at the project root, which are the orchestrator's
to write at session close.

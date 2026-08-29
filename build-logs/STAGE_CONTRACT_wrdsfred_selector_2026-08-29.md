# Stage contract S1 — WRDS/FRED data_ref value capture (P-09 next-item 1)

Date: 2026-08-29. Orchestrator: Fable portfolio session. Executor seat: Opus 5 at medium effort (Ryan's 2026-08-17 ruling, scope corrected 2026-08-18). Repo: /Users/ryanfree/Documents/Projects/deerflow-dr/deer-flow (git root; parent dir is NOT a repo), branch `deerflow-dr`, HEAD 0ae1e08d, working tree already carries ~18 uncommitted entries from the 2026-08-27 session — leave them exactly as they are.

## Goal

`wrds_query` gains an optional single-scalar metric selector; `fred_series` gains series-unit metadata; both mint data_refs carrying `value` (+ `unit`, + metric/series identity) so `audit_provenance` can reach MATCHED deterministically for WRDS- and FRED-backed claims — without weakening the D10/D11 invariant that absent or unparseable evidence yields INCONCLUSIVE, never MISMATCH.

## Inputs (exact)

- `backend/packages/dr_core/dr_core/connectors/tools.py` — wrds_query ~88-150; the edgar_company_facts pattern to mirror ~154-214 (data_ref fold-in at ~204-214, docstring echo language ~171-175)
- `backend/packages/dr_core/dr_core/connectors/wrds_client.py` — fetch_crsp_price / fetch_compustat_fundamentals (~136, ~164)
- `backend/packages/dr_core/dr_core/connectors/fred_client.py` — fetch_series_observations (~85), FRED_OBSERVATIONS_URL (~29, 92)
- `backend/packages/dr_core/dr_core/verify/provenance.py` ~55-92 and `verify/provenance_compare.py` (compare_claim_to_data_ref ~136-149, compare_quantities ~121-133, DEFAULT_RELATIVE_TOLERANCE ~51)
- `backend/packages/dr_core/dr_core/graph/claim_tool.py` ~143-161 (retained-snapshot exact-match rule)
- `/Users/ryanfree/Documents/Projects/deerflow-dr/DECISIONS.md` — D10, D11 only
- `/Users/ryanfree/Documents/Projects/deerflow-dr/BUILD_LEDGER.md` — 2026-07-18 Item-1 scope call (~1737-1748) only
- Tests: `backend/packages/dr_core/tests/test_connectors_wrds_client.py`, `test_connectors_edgar_client.py` (transport-closure pattern ~24-32), `test_connectors_fred_client.py`, `test_verify_provenance.py`, snapshot/claim-tool tests as needed

## Do-Not-Load

Upstream deer-flow app trees (frontend/web/src outside dr_core), any .venv contents, node_modules, build-logs/ bulk (this contract and your own outputs only), git history beyond `git status`/`git diff` of your own edits.

## Settled decisions (binding — escalate in your report rather than deviate)

1. **Additive and backward compatible.** `wrds_query(metric=None)` keeps today's compound behavior and its value-less data_ref byte-for-byte (regression test proves it). No semantic change to provenance.py or provenance_compare.py — the connector boundary is the only place values are shaped.
2. **WRDS selector**: explicit optional `metric` argument dispatching to exactly ONE scalar field. Mandatory vocabulary: Compustat `at`, `revt`, `ni`, `sale` + CRSP `price`. Values minted in BASE units (Compustat reports millions — convert to USD via ×1e6; document and test the conversion; price is USD/share). CRSP `ret`/`vol`/`shrout`: include ONLY if you can prove no-false-MISMATCH unit semantics with tests; otherwise exclude them and say so in the docstring. Unknown metric name → hard error at the tool boundary, never a silent full-record fallback.
3. **data_ref additions mirror EDGAR's keys**: `value`, `unit`, plus `metric` for WRDS (do NOT reuse `concept` — that key means XBRL vocabulary). FRED data_ref gains `value`, `unit` (+ series identity if not already present).
4. **FRED**: new fetch function against the `/fred/series` metadata endpoint (this codebase has never called it), own injectable `transport` param, own mocked tests. Unit normalization at the connector boundary ONLY for provably scale-bearing units ("Billions of Dollars" → value×1e9 with a currency unit; Millions/Thousands likewise; "Percent" passed through for the comparator's percent path). Unrecognized unit strings: mint value raw + unit verbatim, which the comparator resolves to INCONCLUSIVE — that is the deliberate fail-safe; never guess a scale.
5. **Unit spellings**: only mint unit strings the existing comparator provably handles — read provenance_compare.py and the EDGAR path to find what MATCHED-reachable units look like; do not assume.
6. **Docstrings**: instruct the model to echo `value`/`unit`/`metric` verbatim into record_claim data_refs (mirror the EDGAR language at tools.py ~171-175). claim_tool's retained-snapshot exact-match (~159-161) makes any drift a record-time rejection — that is intended.
7. **TDD**: failing tests first, per suite conventions (closure transport fixtures for HTTP clients; MagicMock conn for WRDS). Every normalization/conversion has a test. At least one test per connector asserting the false-MISMATCH class CANNOT occur (e.g. a revenue claim can no longer be compared against a price payload once metric is selected; unrecognized FRED unit → INCONCLUSIVE, not MISMATCH).

## Process

1. Failing tests → implement → green. Canonical suite ONLY: from `backend/`, `.venv/bin/python -m pytest packages/dr_core/`. NEVER `make test` (hangs on ~6.9k upstream tests). Ruff on touched files per repo config.
2. **Live acceptance** (credentials already in `backend/.env`; read-only API calls; keep probes bounded):
   a. `wrds_query` with `metric="revt"` for a real ticker/period → data_ref carries value/unit; feed a synthetic claim quoting the same figure through `compare_claim_to_data_ref` → MATCH.
   b. `fred_series` for GDP (or another scale-bearing series) → unit captured and normalized; same synthetic-claim probe → MATCH.
   c. One deliberate wrong-figure probe → MISMATCH (proves the comparator actually sees the value).
   d. One unrecognized-unit probe → comparator returns None / claim stays UNAUDITED (proves the fail-safe).
   Log full probe outputs to `build-logs/wrdsfred_selector_live_acceptance_2026-08-29.md`.
3. **COMMIT NOTHING.** No `git add`/`commit`/`stash`; do not touch `.git`. (iCloud churn rule: commit only at session close, and that decision is the orchestrator's.)
4. Report (≤800 words) to `build-logs/S1_REPORT_wrdsfred_selector_2026-08-29.md`: files changed with line ranges, test delta (baseline 815 passed + 3 skipped → new counts), verbatim suite tail, live probe outcomes, judgment calls made inside the contract's latitude, anything you could not do and why.

## Verify (definition of done)

- Canonical suite: zero failures; passed count strictly above 815; skips unchanged at 3 (or explained).
- ruff clean on touched files.
- `metric=None` regression test proves the legacy data_ref is unchanged.
- Live probes: WRDS MATCH, FRED MATCH, deliberate MISMATCH, fail-safe INCONCLUSIVE — all four demonstrated and logged.
- No edits outside `backend/packages/dr_core/` and `build-logs/`.
- Both report files exist at the named paths.

#!/usr/bin/env python3
"""score_run.py — mechanical scorer for the dr_core seeded-trap eval set.

Ported from the FROZEN harness's evals/score_run.py (DeepResearch project) per
build-logs/task-port-evalfixtures.md. Runtime-agnostic: reads ONLY a produced run
folder's artifacts (manifest.json, claims.jsonl, sources.jsonl, report.md, per
PHASE1-SHARED-CONTRACT.md) and checks them against a named fixture's `expect` block
in profile_questions.yaml (copied verbatim — it is data). No model-graded rubric:
every assertion is a count or a flag check. No dr.js, no engine, no network calls.

Usage:
  uv run python3 -m dr_core.eval.score_run --fixture tech_unreproduced --run <run-folder>
  uv run python3 -m dr_core.eval.score_run --all --runs-dir <runs-dir>

Exit code 0 if all checked fixtures pass, 1 otherwise.

Rewires (build-logs/task-port-evalfixtures.md "Rewires" + PHASE1-SHARED-CONTRACT.md):
  - claims/sources are loaded as dr_core.models.Claim/Source instances (claims.jsonl
    line order IS the citation ordinal), not the harness's plain dr.js-era dicts.
  - Eligibility of a cited claim [n] is a property of THAT claim (claim.citation_status,
    claim.gate_flags, claim.verification.status) — the "claim-keyed" repoint. The old
    harness's separate manifest `citation_gate` block does not exist under dr_core;
    `citation_gate.found`/`not_found` and `counts.citations_not_found` are now derived
    directly from each claim's own `citation_status` (RESOLVED / NOT_FOUND).
  - Gate flags compare against `Claim.gate_flags` (dr_core.models.enums.GateFlag,
    underscore-valued, e.g. "numeric_without_primary_trace") after normalizing the
    fixture's hyphenated flag names (e.g. "numeric-without-primary-trace") — the
    fixture yaml is copied verbatim and was never rewritten to match the enum spelling.
  - `counts.claims_flagged` / `counts.claims_killed` / `counts.structured_sources` are
    not dr_core manifest fields (write_run's own `counts` aggregate is publication-status
    shaped: supported/not_verified/contested/excluded/killed_on_refute/caveats — see
    dr_core.run.write_run._claim_status_counts) — they are computed here directly off
    claims/sources, with `claims_killed` reading manifest counts' `killed_on_refute` and
    `structured_sources` REINTERPRETED as structured CLAIMS (`claim.data_ref is not
    None`): dr_core's structured/quote grounding split lives on Claim
    (derive.structured_grounded), and dr_core's Source model has no structured/press
    distinction at all.
  - `gate_flags_absent_for_class` keys a claim's source by `Source.source_system` as
    the stand-in for the harness's `source_class` (press/preprint/vendor_docs/...):
    dr_core's Source model has no `source_class` field, and `source_system` is the one
    free-text per-source label available to carry an equivalent tag.
  - Dropped: `--result` (scoring an engine's raw task-output file) and `load_run_from_result`
    — there is no dr.js engine producing that shape anymore; a produced run folder is the
    only input. `--runs-dir` no longer defaults into the old DeepResearch runs tree; it
    defaults from `DR_CORE_RUNS_DIR` (mirroring dr_core.run.write_run's own default), and
    must otherwise be supplied explicitly.
  - D10 (S10 scorer/ledger defect): claims now load from `ledger.jsonl` (the FULL claim
    ledger write_run now writes — every claim, any eligibility state) when present, so
    gate_flags/citation_status/counts checks can see excluded/killed claims too, not just
    the eligible-only `claims.jsonl` the scorer used to be limited to. Run folders written
    before this change have no `ledger.jsonl`; `load_run_from_folder` falls back to
    `claims.jsonl` for those (printing a note) — scorability for older folders is reduced
    to what `claims.jsonl` alone can show, same as before this change.
"""

import argparse
import glob
import json
import os
import sys

from dr_core.models import CitationStatus, Claim, Source

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "profile_questions.yaml")


def load_fixtures():
    import yaml

    with open(FIXTURES) as f:
        return (yaml.safe_load(f) or {}).get("fixtures", {})


def load_run_from_folder(folder):
    with open(os.path.join(folder, "manifest.json")) as f:
        manifest = json.load(f)
    ledger_path = os.path.join(folder, "ledger.jsonl")
    if os.path.isfile(ledger_path):
        claims_path = ledger_path
    else:
        # Pre-D10 run folder (written before ledger.jsonl existed): fall back to the
        # eligible-only claims.jsonl -- excluded/killed claims are invisible to the
        # scorer for these older folders.
        print(f"# note: {os.path.basename(folder)} has no ledger.jsonl; scoring against claims.jsonl (eligible-only)")
        claims_path = os.path.join(folder, "claims.jsonl")
    claims = [Claim.model_validate(json.loads(line)) for line in open(claims_path) if line.strip()]
    sources_path = os.path.join(folder, "sources.jsonl")
    sources = [Source.model_validate(json.loads(line)) for line in open(sources_path) if line.strip()] if os.path.isfile(sources_path) else []
    report = ""
    rp = os.path.join(folder, "report.md")
    if os.path.isfile(rp):
        report = open(rp).read()
    return manifest, claims, sources, report


def flags_of(claim):
    """claim.gate_flags is a list[GateFlag] (StrEnum, underscore-valued)."""
    return [f.value for f in (claim.gate_flags or [])]


def _normalize_flag(name):
    """Fixture flags are hyphenated (harness spelling); GateFlag values are underscored."""
    return name.replace("-", "_")


def _derive_count(key, manifest_counts, claims, sources):
    """Resolves a `counts.<key>_min` assertion's `key` against dr_core's artifact shapes
    (see the module docstring's Rewires note); falls back to manifest['counts'][key] for
    anything write_run already aggregates unchanged."""
    if key == "sources":
        return len(sources)
    if key == "structured_sources":
        return sum(1 for c in claims if c.data_ref is not None)
    if key == "claims_flagged":
        return sum(1 for c in claims if c.gate_flags)
    if key == "claims_killed":
        return manifest_counts.get("killed_on_refute", 0) or 0
    if key == "citations_not_found":
        return sum(1 for c in claims if c.citation_status == CitationStatus.NOT_FOUND)
    return manifest_counts.get(key, 0) or 0


def score(name, expect, manifest, claims, sources, report):
    checks = []  # (ok, message)
    counts = manifest.get("counts") or {}
    class_by_id = {s.id: s.source_system for s in sources}

    # --- counts.<key>_min -> derived count >= value ---------------------------
    for k, v in (expect.get("counts") or {}).items():
        if not k.endswith("_min"):
            checks.append((False, f"unknown counts assertion '{k}'"))
            continue
        key = k[:-4]
        got = _derive_count(key, counts, claims, sources)
        checks.append((got >= v, f"counts.{key} >= {v} (got {got})"))

    # --- gate_flags_min: flag appears on >= n claims --------------------------
    for flag, n in (expect.get("gate_flags_min") or {}).items():
        target = _normalize_flag(flag)
        got = sum(1 for c in claims if target in flags_of(c))
        checks.append((got >= n, f"flag '{flag}' on >= {n} claim(s) (got {got})"))

    # --- gate_flags_absent_for_class: no claim from a source of <class> -------
    #     may carry any of the listed flags (control claims must stay clean).
    for sclass, flags in (expect.get("gate_flags_absent_for_class") or {}).items():
        targets = {_normalize_flag(f) for f in flags}
        offenders = []
        for c in claims:
            if class_by_id.get(c.source_id) == sclass:
                bad = [f for f in flags_of(c) if f in targets]
                if bad:
                    offenders.append((c.text[:40], bad))
        checks.append((not offenders, f"no {sclass} claim carries {flags} (offenders: {offenders})"))

    # --- citation_gate.{found_min,not_found_min} (derived from claim.citation_status) ---
    cg_exp = expect.get("citation_gate") or {}
    if cg_exp:
        found = sum(1 for c in claims if c.citation_status == CitationStatus.RESOLVED)
        not_found = sum(1 for c in claims if c.citation_status == CitationStatus.NOT_FOUND)
        if "found_min" in cg_exp:
            checks.append((found >= cg_exp["found_min"], f"citation_gate.found >= {cg_exp['found_min']} (got {found})"))
        if "not_found_min" in cg_exp:
            checks.append((not_found >= cg_exp["not_found_min"], f"citation_gate.not_found >= {cg_exp['not_found_min']} (got {not_found})"))

    # --- report_min_chars -----------------------------------------------------
    if "report_min_chars" in expect:
        got = len(report or "")
        checks.append((got >= expect["report_min_chars"], f"report >= {expect['report_min_chars']} chars (got {got})"))

    return checks


def report_result(name, checks):
    passed = all(ok for ok, _ in checks)
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
    for ok, msg in checks:
        print(f"    {'ok  ' if ok else 'FAIL'} {msg}")
    return passed


def _has_seeded_trap(folder):
    """A seeded-trap acceptance run plants sources with source_system 'seeded-trap'; real
    research runs never do. The trap scorer must score the trap run, not a newer same-
    profile research run."""
    sj = os.path.join(folder, "sources.jsonl")
    try:
        with open(sj) as f:
            return "seeded-trap" in f.read()
    except OSError:
        return False


def find_run(runs_dir, profile):
    matches = sorted(glob.glob(os.path.join(os.path.expanduser(runs_dir), f"*-{profile}-*")))
    if not matches:
        return None
    # Prefer the newest folder that actually contains seeded traps; fall back to newest overall.
    seeded = [m for m in matches if _has_seeded_trap(m)]
    return (seeded or matches)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", help="fixture name from profile_questions.yaml")
    ap.add_argument("--run", help="run folder to score")
    ap.add_argument("--all", action="store_true", help="score every fixture against the newest run folder for its profile")
    ap.add_argument("--runs-dir", default=os.environ.get("DR_CORE_RUNS_DIR", "runs"))
    args = ap.parse_args()

    fixtures = load_fixtures()
    results = []

    if args.all:
        for name, fx in fixtures.items():
            folder = find_run(args.runs_dir, fx["profile"])
            if not folder:
                print(f"SKIP  {name} (no run folder for profile {fx['profile']})")
                results.append(False)
                continue
            m, c, s, r = load_run_from_folder(folder)
            print(f"# {name}  <-  {os.path.basename(folder)}")
            results.append(report_result(name, score(name, fx.get("expect") or {}, m, c, s, r)))
    else:
        if not args.fixture or not args.run:
            ap.error("provide --fixture and --run, or use --all")
        fx = fixtures.get(args.fixture)
        if not fx:
            ap.error(f"unknown fixture '{args.fixture}'")
        m, c, s, r = load_run_from_folder(args.run)
        results.append(report_result(args.fixture, score(args.fixture, fx.get("expect") or {}, m, c, s, r)))

    print(f"\n{sum(results)}/{len(results)} fixture(s) pass")
    sys.exit(0 if results and all(results) else 1)


if __name__ == "__main__":
    main()

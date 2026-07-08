"""write_run.py — the dr_core run-folder / manifest writer.

Ported from the retired dr.js-era harness's `harness/write_run.py` (DeepResearch project).
That writer stamped the run folder for the dr.js engine's JSON result; this one does the
same job for dr_core, off a dr_core ledger (Claim/Source/Requirement instances or their
model_dump dicts) instead of dr.js's plain-object shapes.

Kept from the original (the run-folder contract):
  - report.md, sources.jsonl, claims.jsonl, requirements.jsonl, conflicts.jsonl,
    coverage.jsonl, manifest.json
  - a deterministic back-matter appendix (methodology / verification summary /
    requirement coverage / disclaimers) appended to the report body
  - per-stage token rollup from a checkpoint directory (`--checkpoint-dir`)
  - run_env: file hashes + host tool versions, for reproducibility

Phase 1 closure (build-logs/task-phase1-consolidate.md item 1): `conflicts.jsonl` and
`coverage.jsonl` were previously not written at all, so a run folder alone could not
reconstruct contested-via-conflict eligibility or the requirement->claim evidence
linkage. Both are now optional kwargs (`conflicts=`, `coverage=`), serialized the same
way as the other artifacts — one `Conflict`/`CoverageMapping.model_dump(mode="json")`
per line, caller-given order, omitted when absent — per PHASE1-SHARED-CONTRACT.md.

D10 (S10 acceptance defect, scorer/ledger contract): `claims` stays the eligible-only,
citation-ordinal-ordered set that `claims.jsonl` has always been (the Part III
invariant — never resorted, never widened). A new `ledger_claims=` kwarg (defaulting to
`claims` for backward compatibility) carries the FULL claim ledger — every claim
regardless of eligibility, including excluded/killed ones — written to `ledger.jsonl` in
stable claim_id order. `ledger.jsonl` is intentionally NOT ordinal-synced with
`claims.jsonl`; it exists so tooling (score_run.py) can see claims that never made it
into the eligible set.

Rewired for this fork (see build-logs/task-port-writerun.md):
  - No dr.js: the `--deployed-engine` / engine-hash / engine-version-mismatch logic and
    the sibling BUILD.json lookup are gone. Engine identity is now this package's
    version + the repo's git rev (`build_run_env`).
  - RUNS_DEFAULT is no longer a DeepResearch path; it is a parameter/env
    (`--runs-dir` / `DR_CORE_RUNS_DIR`), defaulting to a `runs/` dir under this project.
  - claims/sources/requirements are dr_core.models instances (or model_dump dicts);
    serialization is one `model.model_dump(mode="json")` per JSONL line, in the caller's
    given order — that order IS the citation ordinal (PHASE1-SHARED-CONTRACT.md) and is
    never resorted here.
  - The per-claim evidence ledger and the numbered Sources list are NOT built here
    anymore: PHASE1-SHARED-CONTRACT.md assigns claim-ordinal citation resolution and the
    derived source list to the renderer (dr_core.render, built separately). This writer's
    appendix keeps only the citation-scheme-agnostic parts (methodology, an aggregate
    verification summary via publication_status/claim_caveat, requirement coverage,
    disclaimers).
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dr_core.models import (
    Claim,
    Conflict,
    CoverageMapping,
    Requirement,
    Source,
    VerificationStatus,
    claim_caveat,
    is_grounded,
    materiality_of,
    publication_status,
)
from dr_core.models.enums import PublicationStatus

_HERE = Path(__file__).resolve()
# dr_core/run/write_run.py -> dr_core (inner pkg) -> dr_core (pkg root) -> packages -> backend
# -> deer-flow -> deerflow-dr (project root). Best-effort: falls back to cwd if the
# package is ever installed somewhere without that ancestry.
try:
    _PROJECT_ROOT = _HERE.parents[6]
except IndexError:
    _PROJECT_ROOT = Path.cwd()


def _default_runs_dir() -> str:
    return os.environ.get("DR_CORE_RUNS_DIR") or str(_PROJECT_ROOT / "runs")


def slugify(text, max_words=6):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return "-".join(words[:max_words]) or "run"


def durability_map(connectors_yaml=None):
    """Best-effort source_system -> durability from connectors.yaml (skipped if PyYAML
    or the file is absent — dr_core's fetch/ connectors config is a separate port)."""
    path = connectors_yaml or os.path.join(os.path.dirname(os.path.abspath(__file__)), "connectors.yaml")
    try:
        import yaml

        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        return {c["name"]: c.get("durability") for c in doc.get("connectors", [])}
    except Exception:
        return {}


def _short(text, n=90):
    t = " ".join((text or "").split())
    return (t[: n - 1] + "…") if len(t) > n else t


def _cell(text):
    """Make a value safe for a markdown table cell."""
    return " ".join(str(text or "").split()).replace("|", "\\|")


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _cmd_version(args):
    """Best-effort version/rev capture; never fails the run."""
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=5)
        text = (out.stdout or out.stderr or "").strip()
        return text.splitlines()[0] if text else None
    except Exception:
        return None


def build_run_env(result_engine_version=None):
    """Reproducibility identity for the run: this dr_core build (package version + git
    rev of the checkout it ran from) plus host tool versions. Replaces the dr.js
    deployed-engine hash/mismatch check (build-logs/task-port-writerun.md): dr_core is
    a single installed package, not a workflow-deployed script, so there is no separate
    "deployed vs. result" engine to reconcile — the mismatch flag is dropped."""
    try:
        dr_core_version = importlib.metadata.version("dr-core")
    except importlib.metadata.PackageNotFoundError:
        dr_core_version = None
    return {
        "engine": "dr_core",
        "writer_sha256": _sha256_file(str(_HERE)),
        "dr_core_version": dr_core_version,
        "git_rev": _cmd_version(["git", "-C", str(_HERE.parent), "rev-parse", "--short", "HEAD"]),
        "result_engine_version": result_engine_version,
        "python_version": sys.version.split()[0],
        "claude_version": _cmd_version(["claude", "--version"]),
    }


def _coerce(model_cls, item):
    if isinstance(item, model_cls):
        return item
    if isinstance(item, dict):
        return model_cls(**item)
    raise TypeError(f"expected {model_cls.__name__} or dict, got {type(item).__name__}")


def _prepare_sources(sources, ts_iso):
    """Coerce to Source, filling retrieved_at when a dict omits it (the field is
    required on the model; the original writer filled retrieval_timestamp the same
    way, defaulting it to the run's stamp)."""
    prepared = []
    for item in sources:
        if isinstance(item, Source):
            prepared.append(item)
            continue
        d = dict(item)
        d.setdefault("retrieved_at", ts_iso)
        prepared.append(Source(**d))
    return prepared


def write_jsonl(path, model_cls, items):
    """One model.model_dump(mode="json") per line, in the given order. Callers must
    never resort this list for claims: line order IS the citation ordinal."""
    with open(path, "w") as f:
        for item in items:
            obj = _coerce(model_cls, item)
            f.write(json.dumps(obj.model_dump(mode="json")) + "\n")


def _claim_status_counts(claims, conflicts=()):
    """Aggregate verification summary via the shared derive functions (never
    re-derived locally, per PHASE1-SHARED-CONTRACT.md). materiality is approximated
    from the claim's own importance rating (materiality_of) since write_run has no
    CoverageMapping/Requirement linkage to run the fuller derived_materiality — good
    enough for a summary count; the renderer's per-claim ledger can use the fuller
    derivation where it has that context."""
    counts = {"claims": len(claims), "supported": 0, "not_verified": 0, "contested": 0, "excluded": 0, "killed_on_refute": 0, "caveats": 0}
    for c in claims:
        m = materiality_of(c.importance)
        grounded = is_grounded(c, m)
        status = publication_status(c, materiality=m, grounded=grounded, conflicts=conflicts)
        counts[status.value if isinstance(status, PublicationStatus) else str(status)] = counts.get(status.value, 0) + 1
        if c.verification.status == VerificationStatus.KILLED_ON_REFUTE:
            counts["killed_on_refute"] += 1
        if claim_caveat(c) is not None:
            counts["caveats"] += 1
    # Renderer-facing aliases (M1): render_report.py reads claims_verified/claims_killed/
    # claims_flagged, which is a different shape than this aggregate's publication-status
    # keys above. claims_flagged mirrors eval/score_run.py's own derivation (claims
    # carrying at least one gate flag) so the two counts never drift apart.
    counts["claims_verified"] = counts["supported"]
    counts["claims_killed"] = counts["killed_on_refute"]
    counts["claims_flagged"] = sum(1 for c in claims if c.gate_flags)
    return counts


def build_appendix(claims, sources, requirements, *, profile, model, ts_iso, connectors_used=None, conflicts=()):
    """Deterministic, citation-scheme-agnostic back matter. Does NOT build the
    per-claim evidence ledger or the numbered Sources list (PHASE1-SHARED-CONTRACT.md
    assigns claim-ordinal citation resolution and the derived source list to the
    renderer, built separately in dr_core.render)."""
    counts = _claim_status_counts(claims, conflicts)
    L = ["\n---\n", "## Appendix — Methodology, Verification & Requirement Coverage\n"]
    L.append(
        "*Provenance and validation detail, kept separate from the findings above. The per-claim "
        "evidence ledger and numbered source list are rendered from claims.jsonl/sources.jsonl by "
        "the report renderer; this appendix carries the aggregate summary.*\n"
    )
    L.append(f"**Methodology.** Profile: {profile} · engine: dr_core · model: {model} · generated {ts_iso}. Connectors: {', '.join(connectors_used or []) or '—'}.\n")
    L.append(
        f"**Verification.** {counts['claims']} claims · {counts['supported']} supported · "
        f"{counts['contested']} contested · {counts['excluded']} excluded · {counts['killed_on_refute']} "
        f"killed on refutation · {counts['caveats']} carrying a disclosed caveat.\n"
    )

    if requirements:
        L.append("### Requirement coverage\n")
        L.append("The explicit asks this run tracked, and whether each was answered.\n")
        L.append("| Req | Ask | Kind | Must-cover | Terminal state | Attempts |")
        L.append("|-----|-----|------|-----------|-----------------|----------|")
        for r in requirements:
            mc = "yes" if r.must_cover else "—"
            L.append(f"| {r.id} | {_cell(_short(r.text, 70))} | {r.kind.value} | {mc} | {_cell(r.terminal_state.value if r.terminal_state else '—')} | {r.attempts} |")
        L.append("")

    disc = []
    if profile == "health":
        disc.append("This is an evidence synthesis for informational purposes, not medical advice.")
    if profile == "legal":
        disc.append("First-pass public sources (CourtListener / govinfo), not Westlaw/Lexis; verify critical authorities manually.")
    if profile == "financial":
        disc.append("Figures are stated with their data vintage; confirm against the primary filings before relying on them.")
    if disc:
        L.append("**Disclaimers.** " + " ".join(disc) + "\n")

    return "\n".join(L)


def extract_actual_metrics(raw):
    """Workflow envelope metrics, if the front door was given a task-output envelope
    (a "result" key wrapping the engine's return) rather than the raw engine object."""
    if not isinstance(raw, dict) or ("result" not in raw and "manifest" in raw):
        return None

    def _num(name):
        v = raw.get(name)
        return v if isinstance(v, (int, float)) else None

    return {
        "agentCount": _num("agentCount"),
        "totalTokens": _num("totalTokens"),
        "totalToolCalls": _num("totalToolCalls"),
        "durationMs": _num("durationMs"),
    }


def write_run(
    *,
    report_body,
    sources,
    claims,
    requirements,
    conflicts=None,
    coverage=None,
    ledger_claims=None,
    profile,
    question,
    model="claude-sonnet-5",
    runs_dir=None,
    open_questions=None,
    manifest_in=None,
    checkpoint_dir=None,
    connectors_yaml=None,
    actual_metrics=None,
):
    """Write the dated run folder and return its path. `claims` order is FROZEN as the
    citation ordinal (line 1 -> ordinal 1); never resort it. `conflicts`/`coverage` are
    optional (model instances or dicts, caller-given order) — see module docstring.
    `ledger_claims` (D10) is the full claim ledger (all claims, any eligibility state);
    defaults to `claims` when omitted. Written to `ledger.jsonl` in stable claim_id
    order — NOT the citation ordinal, see module docstring."""
    runs_dir = runs_dir or _default_runs_dir()
    manifest_in = dict(manifest_in or {})

    now = datetime.now().astimezone()
    ts_iso = now.isoformat(timespec="seconds")
    ts_folder = now.strftime("%Y-%m-%d-%H%M%S")
    folder = os.path.join(runs_dir, f"{ts_folder}-{profile}-{slugify(question)}")
    os.makedirs(folder, exist_ok=True)

    prepared_sources = _prepare_sources(sources, ts_iso)
    prepared_claims = [_coerce(Claim, c) for c in claims]
    prepared_requirements = [_coerce(Requirement, r) for r in requirements]
    prepared_conflicts = [_coerce(Conflict, c) for c in (conflicts or [])]
    prepared_coverage = [_coerce(CoverageMapping, c) for c in (coverage or [])]
    ledger_source = claims if ledger_claims is None else ledger_claims
    prepared_ledger_claims = sorted((_coerce(Claim, c) for c in ledger_source), key=lambda c: c.claim_id)

    # --- report.md (body) + deterministic back-matter appendix ---------------
    parts = [(report_body or "(no report)").rstrip()]
    if open_questions and "## Open questions" not in (report_body or ""):
        parts.append("\n## Open questions\n\n" + "\n".join(f"- {q}" for q in open_questions))
    parts.append(
        build_appendix(
            prepared_claims,
            prepared_sources,
            prepared_requirements,
            profile=profile,
            model=model,
            ts_iso=ts_iso,
            connectors_used=manifest_in.get("connectors_used"),
            conflicts=prepared_conflicts,
        )
    )
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write("\n".join(parts).rstrip() + "\n")

    # --- sources.jsonl / claims.jsonl / requirements.jsonl / conflicts.jsonl /
    # coverage.jsonl -------------------------------------------------------------
    write_jsonl(os.path.join(folder, "sources.jsonl"), Source, prepared_sources)
    write_jsonl(os.path.join(folder, "claims.jsonl"), Claim, prepared_claims)
    # ledger.jsonl (D10): the FULL claim ledger, one claim per line, in stable claim_id
    # order -- deliberately NOT the citation-ordinal order claims.jsonl carries, and
    # deliberately not omitted-when-empty like conflicts/coverage below, so tooling can
    # always distinguish "no claims at all" from "file absent".
    write_jsonl(os.path.join(folder, "ledger.jsonl"), Claim, prepared_ledger_claims)
    if prepared_requirements:
        write_jsonl(os.path.join(folder, "requirements.jsonl"), Requirement, prepared_requirements)
    if prepared_conflicts:
        write_jsonl(os.path.join(folder, "conflicts.jsonl"), Conflict, prepared_conflicts)
    if prepared_coverage:
        write_jsonl(os.path.join(folder, "coverage.jsonl"), CoverageMapping, prepared_coverage)

    durs = durability_map(connectors_yaml)
    durability_by_source = {s.source_system: durs.get(s.source_system) for s in prepared_sources if durs.get(s.source_system) is not None}

    # --- checkpoints/ (staged research/verify/synth runs) ---------------------
    checkpoint_info = None
    stage_actuals = {}
    if checkpoint_dir:
        if not os.path.isdir(checkpoint_dir):
            sys.exit(f"write_run: --checkpoint-dir {checkpoint_dir} does not exist")
        stages_present = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".json"))
        dest = os.path.join(folder, "checkpoints")
        shutil.move(checkpoint_dir, dest)
        for fname, stage in (("01-research.json", "research"), ("02-verify.json", "verify"), ("03-synth.json", "synth")):
            p = os.path.join(dest, fname)
            if not os.path.exists(p):
                continue
            try:
                with open(p) as fh:
                    stage_actuals[stage] = (json.load(fh) or {}).get("_actuals")
            except (json.JSONDecodeError, OSError):
                stage_actuals[stage] = None
        synth_incomplete = "(no report)" == (report_body or "").strip()
        checkpoint_info = {"archived_dir": "checkpoints", "stages_present": stages_present, "synth_incomplete": synth_incomplete}

    # --- manifest.json ---------------------------------------------------------
    manifest = dict(manifest_in)
    manifest["timestamp"] = ts_iso
    manifest["model"] = model
    manifest["slug"] = slugify(question)
    manifest.setdefault("profile", profile)
    manifest.setdefault("question", question)
    manifest["run_env"] = build_run_env(manifest_in.get("engine_version"))
    manifest["counts"] = _claim_status_counts(prepared_claims, prepared_conflicts)
    if durability_by_source:
        manifest["durability_by_source"] = durability_by_source
    if checkpoint_info is not None:
        manifest["checkpoint"] = checkpoint_info
    if actual_metrics is not None:
        manifest.setdefault("runtime", {})
        manifest["runtime"]["actual"] = actual_metrics
    if checkpoint_dir and stage_actuals:
        totals = [a.get("totalTokens") for a in stage_actuals.values() if isinstance(a, dict) and isinstance(a.get("totalTokens"), (int, float))]
        manifest.setdefault("runtime", {})
        manifest["runtime"]["actual_by_stage"] = stage_actuals
        manifest["runtime"]["actual_total_tokens"] = int(sum(totals)) if totals else None
    with open(os.path.join(folder, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="path to a JSON file: the raw dr_core result OR a task-output envelope (a 'result' key)")
    ap.add_argument("--profile", default=None, help="defaults to the result manifest's profile")
    ap.add_argument("--question", default=None, help="defaults to the result manifest's question")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--runs-dir", default=None, help="default: $DR_CORE_RUNS_DIR, else <project root>/runs")
    ap.add_argument("--checkpoint-dir", default=None, help="a checkpoint directory (staged research/verify/synth runs); moved into <run-folder>/checkpoints/")
    ap.add_argument("--connectors-yaml", default=None)
    args = ap.parse_args()

    with open(args.result) as f:
        raw = json.load(f)
    actual_metrics = extract_actual_metrics(raw)
    res = raw
    if isinstance(res, dict) and "manifest" not in res and "result" in res:
        res = res["result"]
        if isinstance(res, str):
            res = json.loads(res)

    manifest_in = res.get("manifest") or {}
    profile = args.profile or manifest_in.get("profile")
    question = args.question or manifest_in.get("question")
    if not profile or not question:
        sys.exit("write_run.py: --profile and --question must be given or present in the result manifest")

    folder = write_run(
        report_body=res.get("report"),
        sources=res.get("sources") or [],
        claims=res.get("claims") or [],
        requirements=res.get("requirements") or [],
        conflicts=res.get("conflicts") or [],
        coverage=res.get("coverage") or [],
        profile=profile,
        question=question,
        model=args.model,
        runs_dir=args.runs_dir,
        open_questions=res.get("open_questions") or [],
        manifest_in=manifest_in,
        checkpoint_dir=args.checkpoint_dir,
        connectors_yaml=args.connectors_yaml,
        actual_metrics=actual_metrics,
    )
    print(folder)


if __name__ == "__main__":
    main()

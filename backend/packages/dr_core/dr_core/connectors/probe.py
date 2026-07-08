"""probe.py — minimal reachability prober for dr_core's connector registry (S9).

Ports the auth-handling contract of the old harness's
``~/Documents/Projects/DeepResearch/harness/probe.py`` (query/header/bearer/token
``probe_auth`` schemes) at a much smaller scope: dr_core only needs a pass/fail
table to gate this step, not the full NEEDS-KEY/RATE/AUTH/BADROW taxonomy that
harness tool depends on for `install.sh --probe`.

Contract: 10s timeout, never raises (a bad row or network failure is reported as
a DOWN/BADROW status row, not an exception), stdlib `urllib` only (no new
dependency).

Run via: ``python -m dr_core.connectors.probe --tier 0``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

from dr_core.connectors.registry import Connector, ConnectorValidationError, load_connectors

TIMEOUT_S = 10.0

STATUSES_FAIL = {"DOWN", "AUTH", "BADROW"}


def _env_value(name: str | None, env: dict | None) -> str:
    if not name or name == "none":
        return ""
    env = env if env is not None else os.environ
    return (env.get(name) or "").strip()


def build_probe_request(row: Connector, env: dict | None = None) -> tuple[str | None, dict]:
    """Apply probe_auth (query/header/bearer/token) to the probe URL, if the
    connector declares one and the env var is populated. Returns (url, headers)."""
    url = row.probe_url
    headers: dict[str, str] = {}
    auth_val = _env_value(row.auth, env)
    if row.probe_auth and auth_val and url:
        kind, _, name = row.probe_auth.partition(":")
        if kind == "query":
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{name}={auth_val}"
        elif kind == "header":
            headers[name] = auth_val
        elif kind == "bearer":
            headers["Authorization"] = f"Bearer {auth_val}"
        elif kind == "token":
            headers["Authorization"] = f"Token {auth_val}"
    return url, headers


def _fetch(url: str, headers: dict, timeout: float) -> tuple[str, str]:
    """One GET attempt. Never raises -- any exception becomes a DOWN row."""
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "dr-core-connector-probe/0.1", "Accept": "*/*", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return "OK", str(resp.status)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return "RATE", "429"
        if e.code in (401, 403):
            return "AUTH", str(e.code)
        return "OK", f"{e.code} (reachable)"
    except Exception as e:  # noqa: BLE001 -- probe must never crash on one bad row
        return "DOWN", str(e)[:60]


def probe_connector(row: Connector, env: dict | None = None, timeout: float = TIMEOUT_S) -> tuple[str, str]:
    """Return (status, detail) for one connector. Never raises."""
    has_auth = (not row.auth or row.auth == "none") or bool(_env_value(row.auth, env))
    if not row.probe_url:
        return ("SKIP", "no safe probe_url; launcher/auth check only")
    url, headers = build_probe_request(row, env)
    if not url:
        return ("SKIP", "no probe_url after auth substitution")
    status, detail = _fetch(url, headers, timeout)
    if status == "AUTH" and not has_auth:
        return ("NEEDS-KEY", f"endpoint up, {row.auth} empty")
    if status == "OK" and not has_auth:
        return ("NEEDS-KEY", f"{detail}, {row.auth} empty")
    return (status, detail)


def probe_all(connectors: list[Connector], *, tier: int | None = None, profile: str | None = None, do_all: bool = False, env: dict | None = None, timeout: float = TIMEOUT_S) -> list[dict]:
    """Probe every connector in scope. Scope defaults to Tier 0 (matches the old
    harness's probe.py default) when neither tier/profile/all is given."""
    if do_all:
        scope = list(connectors)
    elif tier is not None:
        scope = [c for c in connectors if c.tier == tier]
    elif profile is not None:
        scope = [c for c in connectors if c.tier == 0 or c.used_by(profile)]
    else:
        scope = [c for c in connectors if c.tier == 0]

    results = []
    for row in scope:
        status, detail = probe_connector(row, env=env, timeout=timeout)
        results.append({"name": row.name, "tier": row.tier, "type": row.type, "access": row.access, "status": status, "detail": detail})
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", type=int, default=None, help="probe only tier N (0, 1, 2)")
    ap.add_argument("--profile", default=None, help="probe Tier 0 plus connectors used by this profile")
    ap.add_argument("--all", action="store_true", dest="do_all", help="probe every connector")
    ap.add_argument("--json", action="store_true", dest="as_json", help="emit JSON instead of a table")
    ap.add_argument("--connectors-yaml", default=None)
    args = ap.parse_args(argv)

    try:
        connectors = load_connectors(args.connectors_yaml)
    except ConnectorValidationError as e:
        print(f"BADROW: {e}", file=sys.stderr)
        return 1

    results = probe_all(connectors, tier=args.tier, profile=args.profile, do_all=args.do_all)

    if args.as_json:
        print(json.dumps(results, indent=2))
    else:
        label = "all" if args.do_all else (f"tier {args.tier}" if args.tier is not None else (f"profile {args.profile}" if args.profile else "tier 0"))
        print(f"\ndr_core connector probe — scope: {label}  ({len(results)} connectors)\n")
        print(f"  {'STATUS':<10} {'TIER':<5} {'NAME':<22} {'TYPE':<13} DETAIL")
        print(f"  {'-' * 10} {'-' * 5} {'-' * 22} {'-' * 13} {'-' * 30}")
        for r in results:
            print(f"  {r['status']:<10} {str(r['tier']):<5} {r['name']:<22} {r['type']:<13} {r['detail']}")
        ok = sum(1 for r in results if r["status"] == "OK")
        warn = sum(1 for r in results if r["status"] in ("NEEDS-KEY", "RATE", "SKIP"))
        fail = sum(1 for r in results if r["status"] in STATUSES_FAIL)
        print(f"\n  {ok} ok · {warn} warn (needs-key/rate/skip) · {fail} fail (auth/down/badrow)\n")

    return 1 if any(r["status"] in STATUSES_FAIL for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

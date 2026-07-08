"""registry.py — typed, validated connector rows parsed from connectors.yaml (S9).

Ports the closed-enum discipline the vendored connectors.yaml documents in its own
header: ``type``/``tier``/``access``/``durability``/``launcher`` are closed sets and
an unknown value is a hard error, never a silently-dropped or coerced row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
DEFAULT_CONNECTORS_YAML = _HERE / "connectors.yaml"

VALID_TYPE = {"unstructured", "structured", "graph"}
VALID_TIER = {0, 1, 2}
VALID_ACCESS = {"mcp", "rest", "mcp+rest"}
VALID_DURABILITY = {"substrate", "stable-api", "fragile-wrapper"}
VALID_LAUNCHER = {"uvx", "npx", "http", "none"}

REQUIRED_FIELDS = (
    "name",
    "type",
    "tier",
    "profiles",
    "access",
    "auth",
    "launcher",
    "durability",
    "rate_limit",
    "last_verified",
    "fallback",
)


class ConnectorValidationError(ValueError):
    """A connectors.yaml row failed closed-enum or required-field validation.

    Raised for the whole load, listing every bad row, so a schema drift in one
    connector cannot silently pass the rest through.
    """


@dataclass(frozen=True)
class Connector:
    """One validated connectors.yaml row. Field names mirror the YAML schema verbatim."""

    name: str
    type: str
    tier: int
    profiles: str | list[str]  # "all" (Tier 0) or a list of profile names
    access: str
    auth: str
    launcher: str
    durability: str
    rate_limit: str
    last_verified: str
    fallback: str | None
    endpoint: str | None = None
    rest_endpoint: str | None = None
    probe_url: str | None = None
    probe_auth: str | None = None
    throughput_note: str | None = None

    def used_by(self, profile: str) -> bool:
        """True if this connector is in scope for the given profile name."""
        if self.profiles == "all":
            return True
        return isinstance(self.profiles, list) and profile in self.profiles

    def has_rest(self) -> bool:
        return self.access in ("rest", "mcp+rest")

    def has_mcp(self) -> bool:
        return self.access in ("mcp", "mcp+rest")


def _row_problems(row: dict) -> list[str]:
    problems = [f"missing {field}" for field in REQUIRED_FIELDS if field not in row]
    if row.get("type") not in VALID_TYPE:
        problems.append(f"type={row.get('type')!r}")
    if row.get("tier") not in VALID_TIER:
        problems.append(f"tier={row.get('tier')!r}")
    if row.get("access") not in VALID_ACCESS:
        problems.append(f"access={row.get('access')!r}")
    if row.get("durability") not in VALID_DURABILITY:
        problems.append(f"durability={row.get('durability')!r}")
    if row.get("launcher") not in VALID_LAUNCHER:
        problems.append(f"launcher={row.get('launcher')!r}")
    return problems


def load_connectors(path: str | Path | None = None) -> list[Connector]:
    """Parse and validate every row in connectors.yaml.

    Raises ``ConnectorValidationError`` listing every offending row if any closed
    enum is violated or a required field is missing -- never returns a partial,
    silently-repaired list.
    """
    doc_path = Path(path) if path else DEFAULT_CONNECTORS_YAML
    with open(doc_path) as f:
        doc = yaml.safe_load(f) or {}
    rows = doc.get("connectors", [])

    errors: list[str] = []
    connectors: list[Connector] = []
    for row in rows:
        problems = _row_problems(row)
        if problems:
            errors.append(f"{row.get('name', '<unnamed>')}: {'; '.join(problems)}")
            continue
        connectors.append(
            Connector(
                name=row["name"],
                type=row["type"],
                tier=row["tier"],
                profiles=row["profiles"],
                access=row["access"],
                auth=row["auth"],
                launcher=row["launcher"],
                durability=row["durability"],
                rate_limit=row["rate_limit"],
                last_verified=row["last_verified"],
                fallback=row.get("fallback"),
                endpoint=row.get("endpoint"),
                rest_endpoint=row.get("rest_endpoint"),
                probe_url=row.get("probe_url"),
                probe_auth=row.get("probe_auth"),
                throughput_note=row.get("throughput_note"),
            )
        )

    if errors:
        raise ConnectorValidationError("connectors.yaml validation failed:\n" + "\n".join(errors))
    return connectors


def by_name(connectors: list[Connector]) -> dict[str, Connector]:
    return {c.name: c for c in connectors}

"""profiles/ — five per-domain research profiles (S9).

Each YAML distills the corresponding old harness profile guide
(``~/Documents/Projects/DeepResearch/harness/profiles/<name>/CLAUDE.md``) plus
connectors.yaml's ``profiles:`` field into a minimal machine-readable shape: a
tool allowlist (by connector name), model tiering, and a default research
depth. This module owns loading + schema validation only; wiring a profile's
allowlist into the graph as an enforced restriction is a later step (dr_run's
``profile`` key, default "general", is the runtime selector once that lands).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent

KNOWN_PROFILES = ("general", "financial", "legal", "tech", "health")

REQUIRED_FIELDS = ("name", "tool_allowlist", "model_tiers", "depth_default")
REQUIRED_MODEL_TIERS = ("gruntwork", "verify", "top")


class ProfileError(ValueError):
    """A profile YAML failed schema validation, or an unknown profile name was requested."""


def load_profile(name: str) -> dict:
    """Load and validate one profile by name.

    Raises ``ValueError`` (unknown profile name, listing the known ones) or
    ``ProfileError`` (known name, but the YAML fails schema validation).
    """
    if name not in KNOWN_PROFILES:
        raise ValueError(f"unknown profile {name!r}; known profiles: {', '.join(KNOWN_PROFILES)}")

    path = _HERE / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ProfileError(f"profile {name!r} missing required field(s): {', '.join(missing)}")

    model_tiers = data.get("model_tiers") or {}
    missing_tiers = [tier for tier in REQUIRED_MODEL_TIERS if tier not in model_tiers]
    if missing_tiers:
        raise ProfileError(f"profile {name!r} model_tiers missing key(s): {', '.join(missing_tiers)}")

    if not isinstance(data.get("tool_allowlist"), list):
        raise ProfileError(f"profile {name!r} tool_allowlist must be a list")

    if data["name"] != name:
        raise ProfileError(f"profile file {name}.yaml declares name={data['name']!r}, expected {name!r}")

    return data


def load_all_profiles() -> dict[str, dict]:
    return {name: load_profile(name) for name in KNOWN_PROFILES}

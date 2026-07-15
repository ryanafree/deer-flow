"""Evidence-class resolution (SPEC_evidence_routing_2026-07-10.md, Stage 2).

Deterministic, not model-judged (settled question 7): a source's evidence
class is looked up, never inferred by an LLM. Resolution precedence is FROZEN
by the spec's Interfaces section: connector ``evidence_class`` first, the
``evidence_domains.yaml`` host-suffix fallback second (``source_system ==
"web"`` only), default ``"news"`` last.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

import yaml

from dr_core.connectors.registry import Connector

_HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE_DOMAINS_YAML = _HERE / "evidence_domains.yaml"

_DOMAIN_MAP_CACHE: dict[str, str] | None = None
_DEFAULT_CLASS_CACHE: str | None = None


def _load_domain_map(path: str | Path | None = None) -> tuple[dict[str, str], str]:
    """Returns (pattern -> class map, default class). Cached only for the
    default (vendored) file -- an explicit path always re-reads, mirroring
    ``connectors.registry.load_connectors``'s ``path=None`` caching contract."""
    global _DOMAIN_MAP_CACHE, _DEFAULT_CLASS_CACHE
    if path is None and _DOMAIN_MAP_CACHE is not None:
        return _DOMAIN_MAP_CACHE, _DEFAULT_CLASS_CACHE  # type: ignore[return-value]

    doc_path = Path(path) if path else DEFAULT_EVIDENCE_DOMAINS_YAML
    with open(doc_path) as f:
        doc = yaml.safe_load(f) or {}

    mapping: dict[str, str] = {}
    for cls, domains in (doc.get("classes") or {}).items():
        for pattern in domains:
            mapping[str(pattern)] = cls
    default_class = doc.get("default", "news")

    if path is None:
        _DOMAIN_MAP_CACHE, _DEFAULT_CLASS_CACHE = mapping, default_class
    return mapping, default_class


def _host_of(url_or_id: str) -> str:
    netloc = urlparse(url_or_id).netloc
    return (netloc or url_or_id).lower()


def _matches_pattern(host: str, pattern: str) -> bool:
    stripped = pattern.lstrip("*").lstrip(".").lower()
    return host == stripped or host.endswith("." + stripped)


def resolve_domain_class(host: str, domain_map: Mapping[str, str], *, default: str = "news") -> str:
    """Host-suffix lookup against ``domain_map`` (pattern -> class); the
    longest matching pattern wins so a specific host (e.g. ``academic.oup.com``)
    is not shadowed by a broader one that happens to iterate first."""
    best: tuple[int, str] | None = None
    for pattern, cls in domain_map.items():
        if _matches_pattern(host, pattern):
            length = len(pattern)
            if best is None or length > best[0]:
                best = (length, cls)
    return best[1] if best else default


def resolve_evidence_class(
    source_system: str,
    url_or_id: str | None,
    connectors_by_name: Mapping[str, Connector],
    *,
    domains_path: str | Path | None = None,
) -> str:
    """Resolve one source's evidence class. ``source_system`` is the
    ``Source.source_system`` value (a connector name, or ``"web"`` for
    generic search/fetch results)."""
    if source_system != "web":
        connector = connectors_by_name.get(source_system)
        if connector is not None and connector.evidence_class:
            return connector.evidence_class
        return "news"

    domain_map, default_class = _load_domain_map(domains_path)
    host = _host_of(url_or_id or "")
    return resolve_domain_class(host, domain_map, default=default_class)


def profile_tools_for_evidence_class(
    evidence_class: str,
    allowlist: Sequence[str],
    connectors_by_name: Mapping[str, Connector],
) -> list[str]:
    """Intersection of a profile's tool allowlist with connectors whose OWN
    ``evidence_class`` field (never the domain fallback -- these are named
    tools, not arbitrary web results) matches ``evidence_class``. Returns []
    for ``"any"`` (nothing specific to recommend) and preserves allowlist
    order so the retry-prompt hint is deterministic."""
    if evidence_class == "any":
        return []
    return [name for name in allowlist if (connector := connectors_by_name.get(name)) is not None and connector.evidence_class == evidence_class]

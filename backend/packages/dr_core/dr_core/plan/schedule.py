"""schedule_queries -- the D11 item-4 shared planning implementation.

ONE implementation, TWO wirings (D11): the entry planning node
(``plan_requirements_node``, deliverable-preset path) schedules the FIRST
research pass, and the eligibility gate's corrective retry schedules the
repair pass on the interactive path. Both call :func:`schedule_queries` and
render the result with :func:`format_schedule_lines`, so the benchmark path
and the chat path can never drift on what "directed work" means.

Deterministic, no LLM: the query text is assembled from the requirement's
own fields (text/entities/window), and the tool hints reuse the Stage-2
``profile_tools_for_evidence_class`` resolution (profile-allowlisted,
fail-closed on unknown profiles -- same posture as the gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dr_core.connectors.registry import Connector
from dr_core.models.ledger import Requirement
from dr_core.plan.evidence_class import profile_tools_for_evidence_class
from dr_core.profiles import ProfileError, load_profile


@dataclass
class ScheduledQuery:
    requirement_id: str
    text: str
    evidence_class: str
    freshness: str
    must_cover: bool
    query: str
    tool_names: list[str] = field(default_factory=list)


def _build_query(requirement: Requirement) -> str:
    parts = [requirement.text]
    if requirement.entities:
        parts.append(" / ".join(requirement.entities))
    window = requirement.window or {}
    window_from, window_to = window.get("from"), window.get("to")
    if window_from or window_to:
        parts.append(f"({window_from or '...'} to {window_to or '...'})")
    return " ".join(parts)


def _bound_display_name(name: str) -> str:
    """Directives must name tools by their ACTUAL bound names (D11 / the
    root-cause track): a name the ToolBinding registry resolves is already a
    bound name and passes through; otherwise it is a connector id and the
    registry describes its bound tool name(s)."""
    from dr_core.connectors.binding import describe_connector_tools, get_default_registry

    if get_default_registry().resolve(name) is not None:
        return name
    return describe_connector_tools(name) or name


def _profile_allowlist(profile: str) -> list[str]:
    try:
        return load_profile(profile or "general")["tool_allowlist"]
    except (ValueError, ProfileError):
        return []


def schedule_queries(
    requirements: list[Requirement],
    profile: str,
    connectors_by_name: dict[str, Connector],
) -> list[ScheduledQuery]:
    """Deterministic per-requirement work schedule, must-cover first.

    Requirements whose ``evidence_class`` is "any" get no tool hint (generic
    web search suffices); class-tagged requirements get the profile-allowlisted
    connector tools able to supply that class, exactly as the gate's Stage-2
    hints resolve them.
    """
    allowlist = _profile_allowlist(profile)
    scheduled: list[ScheduledQuery] = []
    ordered = sorted(requirements, key=lambda r: (not r.must_cover, r.id))
    for requirement in ordered:
        tool_names: list[str] = []
        if requirement.evidence_class != "any":
            tool_names = [_bound_display_name(name) for name in profile_tools_for_evidence_class(requirement.evidence_class, allowlist, connectors_by_name)]
        scheduled.append(
            ScheduledQuery(
                requirement_id=requirement.id,
                text=requirement.text,
                evidence_class=requirement.evidence_class,
                freshness=requirement.freshness,
                must_cover=requirement.must_cover,
                query=_build_query(requirement),
                tool_names=tool_names,
            )
        )
    return scheduled


def format_schedule_lines(scheduled: list[ScheduledQuery]) -> list[str]:
    """Render one directive line per scheduled item (shared by the entry
    planning directive and the gate's corrective message)."""
    lines: list[str] = []
    for item in scheduled:
        marker = "must-cover" if item.must_cover else "supporting"
        tool_suffix = f" (use one of: {', '.join(item.tool_names)})" if item.tool_names else ""
        lines.append(f"- {item.requirement_id} [{marker}, {item.evidence_class}] query: {item.query}{tool_suffix}")
    return lines

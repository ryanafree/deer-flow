"""DrResearchDirectiveMiddleware -- injects the dr-specific research directive
(SPEC_evidence_routing_2026-07-10.md, Stage 3) as a single tagged system message,
without editing the upstream ``lead_agent/prompt.py`` template (FROZEN constraint
-- ``apply_prompt_template`` takes no addendum parameter, so forking that file
would mean FORK_DELTA churn and rebase pain; see settled question 8).

Implementation note (deviation from the spec's stated mechanism, documented per
stage-contract convention): the Interfaces section describes relying on
``SystemMessageCoalescingMiddleware`` to merge an injected ``SystemMessage``
into the leading one. Direct inspection of ``lead_agent/agent.py::build_middlewares``
(coalescing appended before ``custom_middlewares``, i.e. ``extra_middlewares``)
and ``langchain.agents.factory._chain_model_call_handlers`` ("first in list
becomes outermost layer") shows coalescing is OUTER relative to this middleware,
so its merge pass runs BEFORE this middleware's ``wrap_model_call`` executes --
not after. Relying on it as described would leave this middleware's injected
text uncoalesced against strict backends. This middleware sidesteps the
ordering question entirely by merging the directive directly into
``request.system_message`` itself (the one field LangChain actually sends),
producing exactly one well-formed, marker-tagged system message regardless of
where in the middleware chain it runs. No upstream file is touched either way.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from dr_core.connectors.binding import describe_connector_tools
from dr_core.connectors.registry import Connector, load_connectors
from dr_core.plan.evidence_class import profile_tools_for_evidence_class
from dr_core.profiles import ProfileError, load_profile

DIRECTIVE_MARKER = "dr-research-directive/v1"

# Source hierarchy order for conceptual/causal claims (Interfaces section (b)).
_EVIDENCE_CLASSES_IN_HIERARCHY_ORDER = ("academic", "primary_data", "practitioner", "news")


class DrResearchDirectiveMiddleware(AgentMiddleware):
    """Injects the dr research-assurance directive into every model request."""

    def __init__(self, connectors: list[Connector] | None = None):
        super().__init__()
        connectors = connectors if connectors is not None else load_connectors()
        self._connectors_by_name = {c.name: c for c in connectors}
        self._profile_cache: dict[str, dict | None] = {}
        self._directive_cache: dict[str, str] = {}

    def _load_profile_cached(self, name: str) -> dict | None:
        if name not in self._profile_cache:
            try:
                self._profile_cache[name] = load_profile(name)
            except (ValueError, ProfileError):
                self._profile_cache[name] = None
        return self._profile_cache[name]

    def _profile_name(self, state: dict | None) -> str:
        dr_run = (state or {}).get("dr_run") or {}
        return dr_run.get("profile") or "general"

    def _preferred_tools_section(self, profile_name: str) -> str:
        profile = self._load_profile_cached(profile_name)
        if profile is None:
            return ""
        allowlist = profile["tool_allowlist"]
        lines = []
        for cls in _EVIDENCE_CLASSES_IN_HIERARCHY_ORDER:
            names = profile_tools_for_evidence_class(cls, allowlist, self._connectors_by_name)
            if names:
                # profile_tools_for_evidence_class returns CONNECTOR names
                # (profile allowlist vocabulary); describe_connector_tools
                # translates each to the ACTUAL bound tool name(s) -- literal
                # where dr_core knows them (Stage C typed tools, the two
                # community-wired web tools), else the MCP tool_name_prefix
                # pattern -- per D11 P0 (MYTHOS_REVIEW_2026-07-11.md finding
                # 3: the old directive named connector identifiers the model
                # can't actually call).
                described = [f"{name} ({describe_connector_tools(name)})" for name in names]
                lines.append(f"- {cls}: {', '.join(described)}")
        if not lines:
            return ""
        return "Preferred tools for this profile, by evidence class:\n" + "\n".join(lines)

    def _directive_text(self, profile_name: str) -> str:
        if profile_name in self._directive_cache:
            return self._directive_cache[profile_name]
        preferred = self._preferred_tools_section(profile_name)
        lines = [
            f"<!-- {DIRECTIVE_MARKER} -->",
            "Research directive (dr-assurance runs):",
            "1. Research mandate: this is a dr research question. Every substantive claim must "
            "be answered from tool-gathered sources (web search/fetch, connectors, structured "
            "data) surfaced during this run, never from parametric memory alone. If you have not "
            "called a tool to check something, do not assert it as fact.",
            "2. Source hierarchy: for conceptual or causal claims, prefer sources in this order: "
            "academic > primary_data > practitioner > news. For quantitative or current-value "
            "claims (a number, a rate, a current price or level), prefer primary_data sources "
            "first. Never assert a causal claim without citing a mechanism from the source.",
            "3. Baseline before frontier: state the classical or canonical model or finding before presenting a frontier or contested claim, so the reader has the baseline first.",
            "4. Disclose gaps: when a must-cover requirement cannot be answered from the sources gathered, say so explicitly rather than omitting it silently.",
        ]
        if preferred:
            lines.append(preferred)
        text = "\n".join(lines)
        self._directive_cache[profile_name] = text
        return text

    def _inject(self, request: ModelRequest) -> ModelRequest:
        profile_name = self._profile_name(request.state)
        directive = self._directive_text(profile_name)
        base = request.system_message
        base_text = base.text if base is not None else ""
        merged_content = f"{base_text}\n\n{directive}" if base_text else directive
        merged_kwargs = dict(base.additional_kwargs) if base is not None else {}
        merged_kwargs["dr_research_directive"] = DIRECTIVE_MARKER
        new_system_message = SystemMessage(
            content=merged_content,
            id=base.id if base is not None else None,
            additional_kwargs=merged_kwargs,
        )
        return request.override(system_message=new_system_message)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._inject(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._inject(request))

"""DrLedgerMiddleware — state-schema-only middleware plus the C2 source hook
and the C1 record_claim tool.

Declares the 5 dr_* channels via ``state_schema`` so ``create_agent`` merges
them into the nested lead-agent graph's state (see ``DrAgentState``). The
``before_model``/``abefore_model`` hooks add the C2 deterministic
source-extraction seam (D5 ruling C): every ``web_search``/``web_fetch``
``ToolMessage`` is turned mechanically into a ``Source`` record, no LLM, no
network. Per D7, any scan that processes at least one new source-bearing
ToolMessage also sets ``dr_run["deliverable"]=True`` -- attempting a search
counts as research intent for turn-scoped gate routing, even when zero
sources parse out of it. Claims (C1) ride ``tools = [record_claim]`` (see
``claim_tool.py``): a model-facing tool whose handler is likewise
deterministic validation, not content generation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime

from dr_core.connectors.binding import ResultShape, ToolBinding, get_default_registry, mcp_generic_url_and_title, parse_mcp_generic_records
from dr_core.graph.claim_tool import record_claim
from dr_core.graph.state import DrAgentState
from dr_core.models import Source

# Which tool names are source-bearing, and how to parse each one's payload,
# is resolved entirely through dr_core.connectors.binding.ToolBindingRegistry
# (D11 P0 "normalize the connector boundary" -- MYTHOS_REVIEW_2026-07-11.md).
# The registry resolves a runtime ToolMessage.name -- whether or not
# DeerFlow's MCP loader has prefixed it (harness/deerflow/mcp/tools.py,
# tool_name_prefix=True) -- to a ToolBinding carrying the connector,
# source_system, authority_tier, and ResultShape needed below. An unresolved
# name is NOT a source-bearing tool call and is skipped, not guessed at.

_DEFAULT_AUTHORITY_TIER = 3
_SOURCED_MSG_IDS_KEY = "_sourced_msg_ids"
_COUNTED_TOOL_CALL_IDS_KEY = "_counted_tool_call_ids"


def _tool_call_args_by_id(messages: list[AnyMessage]) -> dict[str, dict[str, Any]]:
    """Map tool_call_id -> args from every AIMessage tool call (mirrors
    ``delegation_ledger.extract_delegations``'s call/result pairing)."""
    args_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls or []:
            tool_call_id = tool_call.get("id")
            if tool_call_id:
                args_by_id[str(tool_call_id)] = tool_call.get("args") or {}
    return args_by_id


def _source_id(url: str) -> str:
    """Deterministic short id: the same URL always yields the same source id."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _build_source(url: str, title: str | None, retrieved_at: str) -> dict:
    source = Source(
        id=_source_id(url),
        url_or_id=url,
        source_system="web",
        title=title,
        authority_tier=_DEFAULT_AUTHORITY_TIER,
        retrieved_at=retrieved_at,
    )
    return source.model_dump(mode="json")


def _sources_from_web_search(content: str, retrieved_at: str) -> dict[str, dict]:
    try:
        results = json.loads(content)
    except (TypeError, ValueError):
        return {}
    # The live ddg_search web_search tool returns an OBJECT:
    # {"query", "total_results", "results": [{"title","url","content"}, ...]}
    # (community/ddg_search/tools.py:175-182; caught by the S6 live E2E run,
    # which extracted zero sources from a real search). Tavily-style tools
    # return a bare list. Accept both; anything else (e.g. the ddg "error"
    # object) extracts nothing.
    if isinstance(results, dict):
        results = results.get("results")
    if not isinstance(results, list):
        return {}

    sources: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        if not isinstance(url, str) or not url:
            continue
        record = _build_source(url, result.get("title"), retrieved_at)
        sources[record["id"]] = record
    return sources


def _source_from_web_fetch(content: str, url: str | None, retrieved_at: str) -> dict[str, dict]:
    if not url or not isinstance(content, str) or content.startswith("Error:"):
        return {}
    title = None
    if content.startswith("# "):
        title = content[2:].split("\n", 1)[0].strip() or None
    record = _build_source(url, title, retrieved_at)
    return {record["id"]: record}


def _source_from_structured_tool(content: str, retrieved_at: str, *, source_system: str, authority_tier: int) -> dict[str, dict]:
    """Generic parser for Stage-C structured connector tools: reads a
    `url_or_id` (+ optional `title`) straight from the tool's own JSON
    payload -- no tool-call-args lookup needed, the payload is
    self-describing (connectors/tools.py's contract). Anything that isn't
    `ok: true` JSON with a string `url_or_id` yields no source, mirroring
    _source_from_web_fetch's error-string skip. Every Stage-C payload also
    self-reports its own `source_system`; that wins over the binding's
    default when present (the default only covers tools -- like
    academic_search -- whose actual connector varies per call)."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {}
    url_or_id = payload.get("url_or_id")
    if not isinstance(url_or_id, str) or not url_or_id:
        return {}
    payload_source_system = payload.get("source_system")
    source = Source(
        id=_source_id(url_or_id),
        url_or_id=url_or_id,
        source_system=payload_source_system if isinstance(payload_source_system, str) and payload_source_system else source_system,
        title=payload.get("title"),
        authority_tier=authority_tier,
        retrieved_at=retrieved_at,
    )
    record = source.model_dump(mode="json")
    return {record["id"]: record}


def _sources_from_structured_search_tool(content: str, retrieved_at: str, *, source_system: str, authority_tier: int) -> dict[str, dict]:
    """Generic parser for Stage-C structured connector SEARCH tools (multiple
    results per call, e.g. courtlistener_search, academic_search): reads a
    `results` list of `{url_or_id, title, ...}` dicts from the tool's own
    JSON payload and mints ONE Source per result -- mirrors
    `_sources_from_web_search`'s per-result minting, but keyed on `url_or_id`
    (the structured-tool convention) instead of `url` (the web-tool
    convention). The payload's own top-level `source_system` wins over the
    binding's default when present, same as the single-fact parser above --
    academic_search's actual connector (semantic_scholar/openalex/arxiv)
    varies per call and is only known from the payload."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict) or not payload.get("ok"):
        return {}
    results = payload.get("results")
    if not isinstance(results, list):
        return {}
    payload_source_system = payload.get("source_system")
    resolved_source_system = payload_source_system if isinstance(payload_source_system, str) and payload_source_system else source_system

    sources: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        url_or_id = result.get("url_or_id")
        if not isinstance(url_or_id, str) or not url_or_id:
            continue
        source = Source(
            id=_source_id(url_or_id),
            url_or_id=url_or_id,
            source_system=resolved_source_system,
            title=result.get("title"),
            authority_tier=authority_tier,
            retrieved_at=retrieved_at,
        )
        record = source.model_dump(mode="json")
        sources[record["id"]] = record
    return sources


def _sources_from_mcp_generic(content: str, retrieved_at: str, *, source_system: str, authority_tier: int) -> dict[str, dict]:
    """Parser for ResultShape.MCP_GENERIC: the binding registry resolved the
    CONNECTOR behind a prefixed MCP tool call by name, but not the
    third-party MCP server's own JSON payload shape (dr_core doesn't control
    it). ``connectors.binding.parse_mcp_generic_records`` does the permissive
    list-of-dicts extraction; this mints one Source per record that carries a
    url-like field, same fail-closed-on-no-match behavior as the other
    parsers here."""
    sources: dict[str, dict] = {}
    for record in parse_mcp_generic_records(content):
        url, title = mcp_generic_url_and_title(record)
        if not url:
            continue
        source = Source(
            id=_source_id(url),
            url_or_id=url,
            source_system=source_system,
            title=title,
            authority_tier=authority_tier,
            retrieved_at=retrieved_at,
        )
        record_out = source.model_dump(mode="json")
        sources[record_out["id"]] = record_out
    return sources


def _extract_for_binding(binding: ToolBinding, content: str, retrieved_at: str, call_args_by_id: dict[str, dict[str, Any]], tool_call_id: str) -> dict[str, dict]:
    if binding.result_shape is ResultShape.SEARCH_URL:
        return _sources_from_web_search(content, retrieved_at)
    if binding.result_shape is ResultShape.FETCH_URL:
        url = call_args_by_id.get(tool_call_id, {}).get("url")
        return _source_from_web_fetch(content, url, retrieved_at)
    if binding.result_shape is ResultShape.STRUCTURED_SINGLE:
        return _source_from_structured_tool(content, retrieved_at, source_system=binding.source_system, authority_tier=binding.authority_tier)
    if binding.result_shape is ResultShape.STRUCTURED_SEARCH:
        return _sources_from_structured_search_tool(content, retrieved_at, source_system=binding.source_system, authority_tier=binding.authority_tier)
    return _sources_from_mcp_generic(content, retrieved_at, source_system=binding.source_system, authority_tier=binding.authority_tier)


class DrLedgerMiddleware(AgentMiddleware):
    """Contributes the dr_* ledger channels, the C2 source-extraction hook, and
    the C1 record_claim tool."""

    state_schema = DrAgentState
    tools = [record_claim]

    def before_model(self, state: dict, runtime: Runtime) -> dict | None:
        return self._extract_sources(state)

    async def abefore_model(self, state: dict, runtime: Runtime) -> dict | None:
        return self._extract_sources(state)

    def _extract_sources(self, state: dict) -> dict | None:
        """Scan the message tail for unprocessed source-bearing ToolMessages.

        A watermark (``dr_run["_sourced_msg_ids"]``, keyed on tool_call_id)
        guards against re-processing the SAME ToolMessage. That alone is not
        enough: the same URL routinely resurfaces across DIFFERENT
        ToolMessages (a web_search result later web_fetched, or two searches
        with overlapping results), and each sighting would otherwise mint a
        fresh ``retrieved_at`` for the SAME deterministic source id --
        ``merge_ledger`` treats that as a divergent same-field write and
        raises. So a source id is only ever emitted for its FIRST sighting
        (first retrieval time wins): ids already present in ``dr_sources``
        are skipped, and ids extracted earlier in this same scan are skipped
        too. The watermark still advances for every scanned ToolMessage so
        re-scans stay inert regardless of whether it produced a new source.
        """
        messages = state["messages"]
        dr_run = state.get("dr_run") or {}
        processed_ids: set[str] = set(dr_run.get(_SOURCED_MSG_IDS_KEY) or [])
        counted_ids: set[str] = set(dr_run.get(_COUNTED_TOOL_CALL_IDS_KEY) or [])
        tool_call_counts = dict(dr_run.get("tool_call_counts") or {})
        known_source_ids: set[str] = set((state.get("dr_sources") or {}).keys())
        call_args_by_id = _tool_call_args_by_id(messages)
        registry = get_default_registry()

        new_sources: dict[str, dict] = {}
        newly_processed: list[str] = []
        newly_counted: list[str] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_call_id = str(message.tool_call_id)
            if tool_call_id not in counted_ids:
                tool_name = message.name or "unknown"
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                newly_counted.append(tool_call_id)
            if not message.name:
                continue
            binding = registry.resolve(message.name)
            if binding is None:
                continue
            if tool_call_id in processed_ids:
                continue

            content = message.content if isinstance(message.content, str) else str(message.content)
            retrieved_at = datetime.now(UTC).isoformat()
            extracted = _extract_for_binding(binding, content, retrieved_at, call_args_by_id, tool_call_id)

            for source_id, record in extracted.items():
                if source_id in known_source_ids or source_id in new_sources:
                    continue
                new_sources[source_id] = record
            newly_processed.append(tool_call_id)

        if not newly_processed and not newly_counted:
            return None

        # D7: a scan that processed any source-bearing ToolMessage this turn
        # is research intent -- mark the turn deliverable even if zero
        # sources parsed, so route_after_research gates it.
        run_update: dict[str, Any] = {
            _COUNTED_TOOL_CALL_IDS_KEY: sorted(counted_ids | set(newly_counted)),
            "tool_call_count": int(dr_run.get("tool_call_count") or 0) + len(newly_counted),
            "tool_call_counts": dict(sorted(tool_call_counts.items())),
        }
        if newly_processed:
            run_update[_SOURCED_MSG_IDS_KEY] = sorted(processed_ids | set(newly_processed))
            run_update["deliverable"] = True
        update: dict[str, Any] = {"dr_run": run_update}
        if new_sources:
            update["dr_sources"] = new_sources
        return update

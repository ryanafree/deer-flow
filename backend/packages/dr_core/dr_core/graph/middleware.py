"""DrLedgerMiddleware — state-schema-only middleware plus the C2 source hook
and the C1 record_claim tool.

Declares the 5 dr_* channels via ``state_schema`` so ``create_agent`` merges
them into the nested lead-agent graph's state (see ``DrAgentState``). The
``before_model``/``abefore_model`` hooks add the C2 deterministic
source-extraction seam (D5 ruling C): every ``web_search``/``web_fetch``
``ToolMessage`` is turned mechanically into a ``Source`` record, no LLM, no
network. Claims (C1) ride ``tools = [record_claim]`` (see ``claim_tool.py``):
a model-facing tool whose handler is likewise deterministic validation, not
content generation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.runtime import Runtime

from dr_core.graph.claim_tool import record_claim
from dr_core.graph.state import DrAgentState
from dr_core.models import Source

# Single extension point for future source-bearing tool providers (exa/serper/
# brave/ddg, ...); out of scope for the walking skeleton.
SOURCE_TOOL_NAMES = frozenset({"web_search", "web_fetch"})

_DEFAULT_AUTHORITY_TIER = 3
_SOURCED_MSG_IDS_KEY = "_sourced_msg_ids"


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
        known_source_ids: set[str] = set((state.get("dr_sources") or {}).keys())
        call_args_by_id = _tool_call_args_by_id(messages)

        new_sources: dict[str, dict] = {}
        newly_processed: list[str] = []
        for message in messages:
            if not isinstance(message, ToolMessage) or message.name not in SOURCE_TOOL_NAMES:
                continue
            tool_call_id = str(message.tool_call_id)
            if tool_call_id in processed_ids:
                continue

            content = message.content if isinstance(message.content, str) else str(message.content)
            retrieved_at = datetime.now(UTC).isoformat()
            if message.name == "web_search":
                extracted = _sources_from_web_search(content, retrieved_at)
            else:
                url = call_args_by_id.get(tool_call_id, {}).get("url")
                extracted = _source_from_web_fetch(content, url, retrieved_at)

            for source_id, record in extracted.items():
                if source_id in known_source_ids or source_id in new_sources:
                    continue
                new_sources[source_id] = record
            newly_processed.append(tool_call_id)

        if not newly_processed:
            return None

        update: dict[str, Any] = {"dr_run": {_SOURCED_MSG_IDS_KEY: sorted(processed_ids | set(newly_processed))}}
        if new_sources:
            update["dr_sources"] = new_sources
        return update

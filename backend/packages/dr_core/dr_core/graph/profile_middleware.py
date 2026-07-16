"""DrProfileToolMiddleware -- profile-scoped tool allowlisting + per-connector
runtime policy (S9-B).

Two independent jobs, both riding the ``extra_middlewares`` seam
``make_dr_agent`` already uses for ``DrLedgerMiddleware`` (zero FORK_DELTA):

1. **Allowlist filtering.** ``wrap_model_call``/``awrap_model_call`` strip any
   bound tool whose name is CONNECTOR-BACKED (present in
   ``dr_core.connectors.tool_map``) but whose connector(s) are not in
   ``load_profile(dr_run["profile"]).tool_allowlist`` from the schema the LLM
   sees. ``wrap_tool_call``/``awrap_tool_call`` back this with an actual
   execution block (a filtered-out tool could still be invoked if the model
   had already committed to a stale tool_call before this turn's filtering,
   or via a directly-injected tool_call) -- this is the enforcement
   boundary, not just a prompt hint. A tool name NOT in the tool_map is not a
   connector tool (bash, record_claim, task, ...) and is always passed
   through untouched; this middleware only ever restricts connector-backed
   tools. An unknown/unresolvable profile name fails CLOSED (empty
   allowlist), never falls back to a more permissive profile.

2. **Runtime policy.** Every connector-backed tool call is wrapped with its
   connector's durability policy (timeout/retries/breaker_threshold, from
   ``connectors/generator.py``'s ``DURABILITY_POLICY_DEFAULTS`` by way of
   ``build_policy_map``). A community tool mapped to MULTIPLE connectors
   (web_search/web_fetch -- see tool_map.py's docstring) uses the most
   conservative (min) policy across its mapped connectors, tracked under a
   breaker key that is the sorted, pipe-joined connector-name set; a typed
   Stage-C tool bound 1:1 to a connector degenerates to that connector's own
   policy and its own breaker key. Breaker state lives in
   ``dr_run["connector_breakers"][breaker_key] = {"strikes": int, "open":
   bool}`` -- an ordinary ``dr_run`` sub-key, so it round-trips through the
   existing ``merge_run`` reducer and is checkpoint-visible / survives
   resume like any other dr_run field. Once a breaker opens it STAYS open
   for the rest of the run (no auto-reset); a successful call resets
   strikes to 0 but never un-opens an already-open breaker.

   Only actual call failures (timeouts, exceptions from the wrapped
   handler) count as strikes. A tool that runs successfully but returns an
   application-level error ``ToolMessage`` (e.g. bad arguments) is NOT a
   connector-reliability failure and does not strike the breaker -- that
   distinction is deliberate, not an oversight.

Scope note -- subagent bypass gap CLOSED (P-09, 2026-07-10): this middleware
only reaches tool calls made by the top-level ``research`` node's own agent
loop, because ``extra_middlewares`` is a parameter of ``_make_lead_agent`` and
is NOT threaded into subagent construction (``SubagentExecutor`` builds
subagents via ``build_subagent_runtime_middlewares``, a separate assembly
path with no injection seam reachable from ``dr_core`` alone -- closing it
properly would require editing core deer-flow files, out of scope). Rather
than leave a dr run with ``subagent_enabled=True`` able to reach connector
tools through an unenforced subagent loop, this middleware now blocks the
dispatch mechanism itself: ``task`` (the subagent-dispatch tool bound on the
research agent's own toolset -- see ``tools/builtins/task_tool.py``) is
stripped from the bound schema in ``wrap_model_call``/``awrap_model_call``
and, defense-in-depth, any call to it is denied with a clear error in
``wrap_tool_call``/``awrap_tool_call`` (see ``SUBAGENT_DISPATCH_TOOL_NAME``
below). Since ``task`` is the *only* path into ``SubagentExecutor``, and that
call is itself a tool call on the node this middleware already wraps, no
subagent can ever be spawned on a dr run -- the enforcement gap in the
subagent's own middleware assembly is now moot because the subagent path is
never reached at all. This gate is unconditional (it does not key off
``dr_run`` contents) because this middleware is only ever wired into
``make_dr_agent``'s research agent (never a plain, non-dr lead agent), so its
mere presence on a graph is itself the "this is a dr run" signal.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from dr_core.connectors.generator import build_policy_map
from dr_core.connectors.registry import Connector, load_connectors
from dr_core.connectors.tool_map import TOOL_TO_CONNECTORS
from dr_core.profiles import ProfileError, load_profile

_MISSING_TOOL_CALL_ID = "missing_tool_call_id"

# The subagent-dispatch tool (`@tool("task", ...)` in
# tools/builtins/task_tool.py). It is the sole entry point into
# SubagentExecutor; blocking it here closes the subagent-bypass gap (see the
# module docstring's "Scope note" above) without touching any core
# deer-flow file.
SUBAGENT_DISPATCH_TOOL_NAME = "task"


class DrProfileToolMiddleware(AgentMiddleware):
    """Filters bound tools to the active dr_run profile's connector
    allowlist and wraps connector-backed tool execution with durability
    policy (timeout/retries/breaker)."""

    def __init__(
        self,
        connectors: list[Connector] | None = None,
        tool_connector_map: dict[str, frozenset[str]] | None = None,
    ):
        super().__init__()
        connectors = connectors if connectors is not None else load_connectors()
        self._connector_names = {c.name for c in connectors}
        self._policy_map = build_policy_map(connectors)
        self._tool_map: dict[str, frozenset[str]] = dict(tool_connector_map) if tool_connector_map is not None else dict(TOOL_TO_CONNECTORS)
        unknown = {name for names in self._tool_map.values() for name in names} - self._connector_names
        if unknown:
            raise ValueError(f"DrProfileToolMiddleware tool_connector_map references unknown connector(s): {sorted(unknown)}")
        self._profile_cache: dict[str, dict] = {}
        self._executor: ThreadPoolExecutor | None = None

    # -- profile / allowlist ------------------------------------------------

    def _profile_name(self, state: dict | None) -> str:
        dr_run = (state or {}).get("dr_run") or {}
        return dr_run.get("profile") or "general"

    def _load_profile_cached(self, name: str) -> dict | None:
        if name not in self._profile_cache:
            try:
                self._profile_cache[name] = load_profile(name)
            except (ValueError, ProfileError):
                self._profile_cache[name] = None
        return self._profile_cache[name]

    def _allowlist(self, state: dict | None) -> set[str]:
        profile = self._load_profile_cached(self._profile_name(state))
        if profile is None:
            # Unknown/invalid profile name -- fail CLOSED, never fall back
            # to a more permissive default.
            return set()
        return set(profile["tool_allowlist"])

    def _connectors_for(self, tool_name: str) -> frozenset[str]:
        return self._tool_map.get(tool_name, frozenset())

    def _is_allowed(self, tool_name: str, allowlist: set[str]) -> bool:
        connectors = self._connectors_for(tool_name)
        if not connectors:
            # Not a connector-backed tool at all -- out of scope, always pass.
            return True
        return bool(connectors & allowlist)

    # -- wrap_model_call: schema filtering -----------------------------------

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        allowlist = self._allowlist(request.state)
        kept = [t for t in request.tools if (getattr(t, "name", None) or "") != SUBAGENT_DISPATCH_TOOL_NAME and self._is_allowed(getattr(t, "name", None) or "", allowlist)]
        if len(kept) == len(request.tools):
            return request
        return request.override(tools=kept)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._filter_tools(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._filter_tools(request))

    # -- wrap_tool_call: execution enforcement + policy ----------------------

    def _blocked_message(self, request: ToolCallRequest, name: str) -> ToolMessage | None:
        if name == SUBAGENT_DISPATCH_TOOL_NAME:
            tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
            return ToolMessage(
                content=(
                    "Error: subagent dispatch via 'task' is disabled on dr-assurance runs. "
                    "Profile/connector enforcement (DrProfileToolMiddleware, DrConnectorToolsMiddleware) "
                    "only reaches this agent's own tool-call loop, not a subagent's separate middleware "
                    "assembly, so dispatching a subagent here would bypass that enforcement. This call is "
                    "blocked outright rather than allowed to run unenforced."
                ),
                tool_call_id=tool_call_id,
                name=name,
                status="error",
            )
        if not name or self._is_allowed(name, self._allowlist(request.state)):
            return None
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
        profile_name = self._profile_name(request.state)
        return ToolMessage(
            content=(f"Error: tool '{name}' is not in the '{profile_name}' profile's connector allowlist and cannot be called on this run."),
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        )

    def _policy_for(self, tool_name: str) -> tuple[dict[str, int] | None, str | None]:
        connectors = self._connectors_for(tool_name)
        if not connectors:
            return None, None
        policies = [self._policy_map[c] for c in connectors if c in self._policy_map]
        if not policies:
            return None, None
        policy = {
            "timeout_s": min(p["timeout_s"] for p in policies),
            "retries": min(p["retries"] for p in policies),
            "breaker_threshold": min(p["breaker_threshold"] for p in policies),
        }
        breaker_key = "|".join(sorted(connectors))
        return policy, breaker_key

    def _breaker_open(self, state: dict | None, breaker_key: str) -> bool:
        dr_run = (state or {}).get("dr_run") or {}
        breakers = dr_run.get("connector_breakers") or {}
        return bool((breakers.get(breaker_key) or {}).get("open"))

    def _record_result(self, state: dict | None, breaker_key: str, policy: dict[str, int], success: bool) -> dict[str, dict]:
        dr_run = (state or {}).get("dr_run") or {}
        breakers = dict(dr_run.get("connector_breakers") or {})
        entry = dict(breakers.get(breaker_key) or {"strikes": 0, "open": False})
        if entry.get("open"):
            # Terminal for this run -- a stray success/failure after open
            # never reopens or un-opens it.
            breakers[breaker_key] = entry
            return breakers
        if success:
            entry["strikes"] = 0
        else:
            entry["strikes"] = int(entry.get("strikes", 0)) + 1
            if entry["strikes"] >= policy["breaker_threshold"]:
                entry["open"] = True
        breakers[breaker_key] = entry
        return breakers

    def _breaker_open_message(self, request: ToolCallRequest, name: str, breaker_key: str) -> ToolMessage:
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
        return ToolMessage(
            content=(f"Error: connector '{breaker_key}' circuit breaker is open after repeated failures; tool '{name}' is unavailable for the remainder of this run."),
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        )

    def _attach_breaker_update(self, result: ToolMessage | Command, breakers: dict[str, dict]) -> ToolMessage | Command:
        if isinstance(result, Command):
            update = dict(result.update or {})
            update["dr_run"] = {**(update.get("dr_run") or {}), "connector_breakers": breakers}
            return dataclasses.replace(result, update=update)
        return Command(update={"messages": [result], "dr_run": {"connector_breakers": breakers}})

    def _executor_pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dr-connector-policy")
        return self._executor

    def _run_with_timeout(self, handler: Callable[[ToolCallRequest], Any], request: ToolCallRequest, timeout_s: float) -> Any:
        future = self._executor_pool().submit(handler, request)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError as exc:
            raise TimeoutError(f"connector call timed out after {timeout_s}s") from exc

    async def _arun_with_timeout(self, handler: Callable[[ToolCallRequest], Awaitable[Any]], request: ToolCallRequest, timeout_s: float) -> Any:
        try:
            return await asyncio.wait_for(handler(request), timeout=timeout_s)
        except TimeoutError as exc:
            raise TimeoutError(f"connector call timed out after {timeout_s}s") from exc

    def _error_result(self, request: ToolCallRequest, name: str, attempts: int, last_exc: Exception | None, breakers: dict[str, dict]) -> Command:
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
        error_tm = ToolMessage(
            content=f"Error: connector call for tool '{name}' failed after {attempts} attempt(s): {last_exc}",
            tool_call_id=tool_call_id,
            name=name,
            status="error",
        )
        return Command(update={"messages": [error_tm], "dr_run": {"connector_breakers": breakers}})

    def _call_with_policy(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command], name: str) -> ToolMessage | Command:
        policy, breaker_key = self._policy_for(name)
        if policy is None:
            return handler(request)
        if self._breaker_open(request.state, breaker_key):
            return self._breaker_open_message(request, name, breaker_key)

        attempts = 1 + policy["retries"]
        last_exc: Exception | None = None
        result = None
        success = False
        for _ in range(attempts):
            try:
                result = self._run_with_timeout(handler, request, policy["timeout_s"])
                success = True
                break
            except Exception as exc:  # noqa: BLE001 -- connector failures must never crash the graph
                last_exc = exc

        breakers = self._record_result(request.state, breaker_key, policy, success)
        if not success:
            return self._error_result(request, name, attempts, last_exc, breakers)
        return self._attach_breaker_update(result, breakers)

    async def _acall_with_policy(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]], name: str) -> ToolMessage | Command:
        policy, breaker_key = self._policy_for(name)
        if policy is None:
            return await handler(request)
        if self._breaker_open(request.state, breaker_key):
            return self._breaker_open_message(request, name, breaker_key)

        attempts = 1 + policy["retries"]
        last_exc: Exception | None = None
        result = None
        success = False
        for _ in range(attempts):
            try:
                result = await self._arun_with_timeout(handler, request, policy["timeout_s"])
                success = True
                break
            except Exception as exc:  # noqa: BLE001 -- connector failures must never crash the graph
                last_exc = exc

        breakers = self._record_result(request.state, breaker_key, policy, success)
        if not success:
            return self._error_result(request, name, attempts, last_exc, breakers)
        return self._attach_breaker_update(result, breakers)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = str(request.tool_call.get("name") or "")
        blocked = self._blocked_message(request, name)
        if blocked is not None:
            return blocked
        return self._call_with_policy(request, handler, name)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        name = str(request.tool_call.get("name") or "")
        blocked = self._blocked_message(request, name)
        if blocked is not None:
            return blocked
        return await self._acall_with_policy(request, handler, name)

"""Tests for DrProfileToolMiddleware (S9-B): profile-scoped tool allowlisting
and per-connector runtime policy (timeout/retries/breaker).

HERMETIC: no live model, no network. Uses the real connector registry and
real profile YAMLs (financial/legal/general/tech) so the allowlist-filtering
tests exercise the actual S9-B acceptance property ("a legal-profile run
cannot call a financial-only tool") against real data. Stage C's not-yet-built
typed connector tools (e.g. a real WRDS tool) are stood in for with a
synthetic tool name bound through a test-local ``tool_connector_map``
override -- Stage C's ``tools.py`` will register real tool names into the
same production map at tool-creation time (see ``connectors/tool_map.py``'s
docstring).
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from dr_core.connectors.registry import load_connectors
from dr_core.graph.profile_middleware import DrProfileToolMiddleware
from langchain.agents import AgentState
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph
from langgraph.types import Command


def _wrds_query_map() -> dict[str, frozenset[str]]:
    # Stand-in for Stage C's real WRDS tool -- bound to the real "wrds"
    # connector row so the allowlist tests exercise real profile data.
    return {"wrds_query": frozenset({"wrds"})}


def _middleware(connectors=None, tool_connector_map=None) -> DrProfileToolMiddleware:
    return DrProfileToolMiddleware(
        connectors=connectors if connectors is not None else load_connectors(),
        tool_connector_map=tool_connector_map if tool_connector_map is not None else _wrds_query_map(),
    )


def _model_request(tools, profile: str | None, dr_run_extra: dict | None = None) -> ModelRequest:
    dr_run = {} if profile is None else {"profile": profile}
    if dr_run_extra:
        dr_run.update(dr_run_extra)
    state = {} if profile is None and not dr_run_extra else {"dr_run": dr_run}
    return ModelRequest(model=None, messages=[], tools=tools, state=state)


def _tool_call_request(tool_name: str, profile: str, tool_call_id: str = "tc1", dr_run_extra: dict | None = None) -> SimpleNamespace:
    dr_run = {"profile": profile}
    if dr_run_extra:
        dr_run.update(dr_run_extra)
    return SimpleNamespace(tool_call={"name": tool_name, "id": tool_call_id}, state={"dr_run": dr_run})


class TestAllowlistFiltering:
    def test_financial_profile_keeps_wrds_query(self):
        mw = _middleware()
        tools = [SimpleNamespace(name="wrds_query"), SimpleNamespace(name="record_claim")]
        filtered = mw._filter_tools(_model_request(tools, "financial"))
        assert {t.name for t in filtered.tools} == {"wrds_query", "record_claim"}

    def test_legal_profile_drops_wrds_query(self):
        mw = _middleware()
        tools = [SimpleNamespace(name="wrds_query"), SimpleNamespace(name="record_claim")]
        filtered = mw._filter_tools(_model_request(tools, "legal"))
        # record_claim is not a connector-backed tool -- always passes through.
        assert {t.name for t in filtered.tools} == {"record_claim"}

    def test_wrap_model_call_applies_filtering_to_the_bound_schema(self):
        mw = _middleware()
        tools = [SimpleNamespace(name="wrds_query")]
        captured = {}

        def handler(req):
            captured["tools"] = req.tools
            return "model-response"

        out = mw.wrap_model_call(_model_request(tools, "legal"), handler)
        assert out == "model-response"
        assert captured["tools"] == []

    def test_web_search_survives_every_profile(self):
        # The retrieval backbone (tavily/searxng/crawl4ai/jina_reader) is in
        # every profile's allowlist, so web_search/web_fetch must never be
        # filtered out by this middleware.
        mw = _middleware()
        tools = [SimpleNamespace(name="web_search"), SimpleNamespace(name="web_fetch")]
        for profile in ("general", "financial", "legal", "tech", "health"):
            filtered = mw._filter_tools(_model_request(tools, profile))
            assert {t.name for t in filtered.tools} == {"web_search", "web_fetch"}, profile

    def test_missing_dr_run_defaults_to_general_profile(self):
        mw = _middleware()
        tools = [SimpleNamespace(name="wrds_query")]
        request = ModelRequest(model=None, messages=[], tools=tools, state={})
        filtered = mw._filter_tools(request)
        # general profile's allowlist has no wrds.
        assert filtered.tools == []


class TestUnknownToolDefaultDeny:
    def test_connector_tool_absent_from_every_relevant_allowlist_is_denied(self):
        mw = _middleware(tool_connector_map={"ghost_tool": frozenset({"wrds"})})
        allowlist = mw._allowlist({"dr_run": {"profile": "tech"}})  # tech never lists wrds
        assert mw._is_allowed("ghost_tool", allowlist) is False

    def test_unknown_profile_name_fails_closed_not_open(self):
        mw = _middleware()
        allowlist = mw._allowlist({"dr_run": {"profile": "not-a-real-profile"}})
        assert allowlist == set()
        assert mw._is_allowed("wrds_query", allowlist) is False

    def test_non_connector_tool_is_always_out_of_scope(self):
        mw = _middleware()
        # Even under a nonsense profile, a non-connector tool (bash, record_claim,
        # task, ...) is never this middleware's concern.
        allowlist = mw._allowlist({"dr_run": {"profile": "not-a-real-profile"}})
        assert mw._is_allowed("bash", allowlist) is True
        assert mw._is_allowed("record_claim", allowlist) is True

    def test_wrap_tool_call_blocks_a_disallowed_tool_before_execution(self):
        mw = _middleware()
        request = _tool_call_request("wrds_query", "legal")

        def handler(req):
            raise AssertionError("handler must not run for a blocked tool")

        result = mw.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "wrds_query" in result.content
        assert "legal" in result.content


class TestPolicyTimeout:
    def test_timeout_fires_and_returns_an_error_without_opening_the_breaker(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 0.05, "retries": 0, "breaker_threshold": 5}
        request = _tool_call_request("wrds_query", "financial")

        def slow_handler(req):
            time.sleep(0.3)
            return ToolMessage(content="too slow", tool_call_id=req.tool_call["id"], name="wrds_query")

        result = mw.wrap_tool_call(request, slow_handler)
        assert isinstance(result, Command)
        messages = result.update["messages"]
        assert messages[0].status == "error"
        assert "timed out" in messages[0].content
        breakers = result.update["dr_run"]["connector_breakers"]
        assert breakers["wrds"] == {"strikes": 1, "open": False}

    def test_retries_are_attempted_before_failing(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 1, "retries": 2, "breaker_threshold": 5}
        request = _tool_call_request("wrds_query", "financial")
        call_count = {"n": 0}

        def always_fails(req):
            call_count["n"] += 1
            raise RuntimeError("connector down")

        result = mw.wrap_tool_call(request, always_fails)
        assert call_count["n"] == 3  # 1 + retries
        assert "failed after 3 attempt(s)" in result.update["messages"][0].content

    def test_success_does_not_touch_the_breaker(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 1, "retries": 0, "breaker_threshold": 3}
        request = _tool_call_request("wrds_query", "financial")

        def ok_handler(req):
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="wrds_query")

        result = mw.wrap_tool_call(request, ok_handler)
        assert isinstance(result, Command)
        assert result.update["messages"][0].content == "ok"
        assert result.update["dr_run"]["connector_breakers"]["wrds"] == {"strikes": 0, "open": False}


class TestBreaker:
    def test_breaker_opens_after_threshold_strikes_and_stays_open_across_a_simulated_resume(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 1, "retries": 0, "breaker_threshold": 2}

        def failing_handler(req):
            raise RuntimeError("connector down")

        # Strike 1.
        result1 = mw.wrap_tool_call(_tool_call_request("wrds_query", "financial", "tc1"), failing_handler)
        breakers1 = result1.update["dr_run"]["connector_breakers"]
        assert breakers1["wrds"] == {"strikes": 1, "open": False}

        # Strike 2 -- trips the breaker.
        req2 = _tool_call_request("wrds_query", "financial", "tc2", dr_run_extra={"connector_breakers": breakers1})
        result2 = mw.wrap_tool_call(req2, failing_handler)
        breakers2 = result2.update["dr_run"]["connector_breakers"]
        assert breakers2["wrds"] == {"strikes": 2, "open": True}

        def poisoned_handler(req):
            raise AssertionError("handler must not run once the breaker is open")

        # Same instance, breaker open: handler is skipped entirely.
        req3 = _tool_call_request("wrds_query", "financial", "tc3", dr_run_extra={"connector_breakers": breakers2})
        result3 = mw.wrap_tool_call(req3, poisoned_handler)
        assert isinstance(result3, ToolMessage)
        assert result3.status == "error"
        assert "circuit breaker is open" in result3.content

        # Simulated resume: a BRAND NEW middleware instance (no shared
        # in-process attributes, standing in for a fresh process after a
        # checkpoint resume) sees only the persisted dr_run state and still
        # refuses to call the handler -- proves the breaker lives in dr_run
        # state, not on the middleware instance.
        mw_resumed = _middleware()
        mw_resumed._policy_map["wrds"] = {"timeout_s": 1, "retries": 0, "breaker_threshold": 2}
        req4 = _tool_call_request("wrds_query", "financial", "tc4", dr_run_extra={"connector_breakers": breakers2})
        result4 = mw_resumed.wrap_tool_call(req4, poisoned_handler)
        assert isinstance(result4, ToolMessage)
        assert result4.status == "error"
        assert "circuit breaker is open" in result4.content

    def test_success_resets_strikes_but_never_un_opens_an_open_breaker(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 1, "retries": 0, "breaker_threshold": 5}

        def failing_handler(req):
            raise RuntimeError("boom")

        result1 = mw.wrap_tool_call(_tool_call_request("wrds_query", "financial", "tc1"), failing_handler)
        breakers1 = result1.update["dr_run"]["connector_breakers"]
        assert breakers1["wrds"]["strikes"] == 1

        def ok_handler(req):
            return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name="wrds_query")

        req2 = _tool_call_request("wrds_query", "financial", "tc2", dr_run_extra={"connector_breakers": breakers1})
        result2 = mw.wrap_tool_call(req2, ok_handler)
        breakers2 = result2.update["dr_run"]["connector_breakers"]
        assert breakers2["wrds"] == {"strikes": 0, "open": False}


class TestAsyncPaths:
    async def test_awrap_model_call_applies_filtering(self):
        mw = _middleware()
        tools = [SimpleNamespace(name="wrds_query")]
        captured = {}

        async def handler(req):
            captured["tools"] = req.tools
            return "model-response"

        out = await mw.awrap_model_call(_model_request(tools, "legal"), handler)
        assert out == "model-response"
        assert captured["tools"] == []

    async def test_awrap_tool_call_blocks_disallowed_tool(self):
        mw = _middleware()
        request = _tool_call_request("wrds_query", "legal")

        async def handler(req):
            raise AssertionError("handler must not run for a blocked tool")

        result = await mw.awrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"

    async def test_async_timeout_fires_without_opening_the_breaker(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 0.05, "retries": 0, "breaker_threshold": 5}
        request = _tool_call_request("wrds_query", "financial")

        async def slow_handler(req):
            await asyncio.sleep(0.3)
            return ToolMessage(content="too slow", tool_call_id=req.tool_call["id"], name="wrds_query")

        result = await mw.awrap_tool_call(request, slow_handler)
        assert isinstance(result, Command)
        assert "timed out" in result.update["messages"][0].content
        assert result.update["dr_run"]["connector_breakers"]["wrds"] == {"strikes": 1, "open": False}

    async def test_async_breaker_opens_and_stays_open(self):
        mw = _middleware()
        mw._policy_map["wrds"] = {"timeout_s": 1, "retries": 0, "breaker_threshold": 1}

        async def failing_handler(req):
            raise RuntimeError("connector down")

        result1 = await mw.awrap_tool_call(_tool_call_request("wrds_query", "financial", "tc1"), failing_handler)
        breakers1 = result1.update["dr_run"]["connector_breakers"]
        assert breakers1["wrds"] == {"strikes": 1, "open": True}

        async def poisoned_handler(req):
            raise AssertionError("handler must not run once the breaker is open")

        req2 = _tool_call_request("wrds_query", "financial", "tc2", dr_run_extra={"connector_breakers": breakers1})
        result2 = await mw.awrap_tool_call(req2, poisoned_handler)
        assert isinstance(result2, ToolMessage)
        assert "circuit breaker is open" in result2.content


class TestNonDrAgentsUnaffected:
    def test_harness_lead_agent_module_never_references_profile_middleware(self):
        """The middleware only reaches a run through make_dr_agent's
        extra_middlewares; the harness's own lead-agent assembly must have
        zero knowledge of it, so a plain (non-dr) lead agent or subagent is
        unaffected by construction, not by a runtime flag."""
        import inspect

        from deerflow.agents.lead_agent import agent as lead_agent_module

        source = inspect.getsource(lead_agent_module)
        assert "DrProfileToolMiddleware" not in source

    def test_make_dr_agent_wires_profile_middleware_into_extra_middlewares(self, monkeypatch):
        import dr_core.graph.agent as dr_agent_module
        from dr_core.graph.middleware import DrLedgerMiddleware

        class _Stub(AgentState):
            pass

        def _passthrough(state):
            return {}

        def _fake_make_lead_agent(config, *, app_config, extra_middlewares=None):
            _fake_make_lead_agent.captured = extra_middlewares
            inner = StateGraph(_Stub)
            inner.add_node("noop", _passthrough)
            inner.set_entry_point("noop")
            inner.set_finish_point("noop")
            return inner.compile()

        monkeypatch.setattr(dr_agent_module, "_make_lead_agent", _fake_make_lead_agent)

        dr_agent_module.make_dr_agent(config={}, app_config=object())

        mws = _fake_make_lead_agent.captured
        assert any(isinstance(m, DrProfileToolMiddleware) for m in mws)
        assert any(isinstance(m, DrLedgerMiddleware) for m in mws)


class TestConstructionValidation:
    def test_rejects_a_tool_map_referencing_an_unregistered_connector(self):
        import pytest

        with pytest.raises(ValueError, match="unknown connector"):
            DrProfileToolMiddleware(connectors=load_connectors(), tool_connector_map={"bogus_tool": frozenset({"not-a-real-connector"})})

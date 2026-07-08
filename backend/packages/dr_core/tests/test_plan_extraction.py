"""Tests for dr_core.plan.extraction (D9 requirement extraction). Hermetic: the
plan model is a fake injected via monkeypatching `_get_plan_model`; no network,
no real LLM.
"""

import json
from types import SimpleNamespace

from dr_core.plan import extraction as extraction_mod
from dr_core.plan.extraction import extract_requirements, latest_real_user_question
from langchain_core.messages import AIMessage, HumanMessage


def _response(payload, **usage_overrides) -> SimpleNamespace:
    usage = {"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cache_read": 1, "cache_creation": 0}}
    usage.update(usage_overrides)
    return SimpleNamespace(content=json.dumps(payload), usage_metadata=usage)


class _FakeModel:
    def __init__(self, response):
        self._response = response

    async def ainvoke(self, messages):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class TestDeterministicIds:
    async def test_same_text_and_kind_yields_same_id(self, monkeypatch):
        payload = [{"kind": "entity", "text": "Who is the CEO?", "must_cover": True}]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))

        reqs1, _ = await extract_requirements("Who is the CEO?")
        reqs2, _ = await extract_requirements("Who is the CEO?")
        assert reqs1[0].id == reqs2[0].id

    async def test_different_kind_yields_different_id(self, monkeypatch):
        payload_entity = [{"kind": "entity", "text": "revenue growth", "must_cover": False}]
        payload_metric = [{"kind": "metric", "text": "revenue growth", "must_cover": False}]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload_entity)))
        reqs_entity, _ = await extract_requirements("q")
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload_metric)))
        reqs_metric, _ = await extract_requirements("q")
        assert reqs_entity[0].id != reqs_metric[0].id


class TestIdempotentReMerge:
    async def test_requirement_payload_is_stable_across_calls(self, monkeypatch):
        payload = [{"kind": "comparison", "text": "compare A and B", "entities": ["A", "B"], "must_cover": True}]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))

        reqs1, _ = await extract_requirements("compare A and B")
        reqs2, _ = await extract_requirements("compare A and B")
        assert reqs1[0].model_dump(mode="json") == reqs2[0].model_dump(mode="json")


class TestHiddenMessageSkipping:
    def test_skips_hide_from_ui_messages_and_returns_last_real_question(self):
        messages = [
            HumanMessage(content="first real question"),
            AIMessage(content="answer"),
            HumanMessage(content="<dr_corrective>gap</dr_corrective>", additional_kwargs={"hide_from_ui": True, "deerflow_dr_corrective": True}),
        ]
        assert latest_real_user_question(messages) == "first real question"

    def test_returns_none_when_only_hidden_messages_present(self):
        messages = [HumanMessage(content="hidden", additional_kwargs={"hide_from_ui": True})]
        assert latest_real_user_question(messages) is None

    def test_returns_none_for_empty_or_blank_only_messages(self):
        assert latest_real_user_question([]) is None
        assert latest_real_user_question([HumanMessage(content="   ")]) is None

    def test_picks_the_most_recent_real_question(self):
        messages = [
            HumanMessage(content="older question"),
            AIMessage(content="answer"),
            HumanMessage(content="newer question"),
        ]
        assert latest_real_user_question(messages) == "newer question"


class TestInvalidOutputDegradesToEmpty:
    async def test_unparseable_model_output_returns_empty(self, monkeypatch):
        response = SimpleNamespace(content="not json at all", usage_metadata={})
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(response))
        reqs, usage = await extract_requirements("some question")
        assert reqs == []
        assert usage["calls"] == 1

    async def test_model_call_failure_returns_empty(self, monkeypatch):
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(RuntimeError("boom")))
        reqs, usage = await extract_requirements("some question")
        assert reqs == []
        assert usage["calls"] == 0

    async def test_individual_malformed_item_is_dropped_not_the_whole_batch(self, monkeypatch):
        payload = [
            {"kind": "not-a-real-kind", "text": "bad item", "must_cover": False},
            {"kind": "entity", "text": "good item", "must_cover": False},
        ]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        reqs, _ = await extract_requirements("q")
        assert len(reqs) == 1
        assert reqs[0].text == "good item"


class TestCapAtEight:
    async def test_more_than_eight_requirements_are_capped(self, monkeypatch):
        payload = [{"kind": "entity", "text": f"requirement {i}", "must_cover": False} for i in range(12)]
        monkeypatch.setattr(extraction_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        reqs, _ = await extract_requirements("q")
        assert len(reqs) == 8

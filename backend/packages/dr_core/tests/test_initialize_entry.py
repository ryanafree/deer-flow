"""D11 item 4 tests: initialize/scope entry node, entry router, shared query
scheduler, entry planning node, and the planned corrective retry."""

from __future__ import annotations

import dr_core.plan.schedule as schedule_mod
import pytest
from dr_core.graph.gate import eligibility_gate
from dr_core.graph.initialize import initialize_node, route_after_initialize
from dr_core.graph.plan import plan_coverage_node, plan_requirements_node
from dr_core.models.ledger import Requirement
from dr_core.plan.schedule import format_schedule_lines, schedule_queries
from langchain_core.messages import HumanMessage


def _req(req_id: str, *, must_cover: bool = True, evidence_class: str = "any", **kwargs) -> Requirement:
    return Requirement(id=req_id, kind="subtopic", text=f"requirement {req_id}", must_cover=must_cover, evidence_class=evidence_class, **kwargs)


class TestInitializeNode:
    def test_clears_turn_scoped_state_and_snapshots_baseline(self):
        state = {
            "messages": [HumanMessage(content="What drives X?")],
            "dr_claims": {"c1": {}, "c2": {}},
            "dr_sources": {"s1": {}},
            "dr_run": {
                "gate_decision": "render",
                "gate_retries": 2,
                "gate_ledger_sig": "stale",
                "active_requirement_ids": ["r-old"],
                "requirements_planned": True,
                "stop_reason": "completed_with_open_requirements",
                "citation_ordinals": {"c1": 1},
                "profile": "financial",
                "depth": "full",
            },
        }
        update = initialize_node(state)["dr_run"]
        assert update["gate_decision"] is None
        assert update["gate_retries"] == 0
        assert update["gate_ledger_sig"] is None
        assert update["active_requirement_ids"] == []
        assert update["requirements_planned"] is False
        assert update["stop_reason"] is None
        assert update["citation_ordinals"] is None
        assert update["baseline_claim_ids"] == ["c1", "c2"]
        assert update["baseline_source_ids"] == ["s1"]
        assert update["question"] == "What drives X?"
        assert update["research_run_id"]
        # run inputs the caller owns are not clobbered by the reset dict
        assert "profile" not in update
        assert "depth" not in update

    def test_deliverable_preset_from_input_state(self):
        update = initialize_node({"dr_run": {"deliverable": True}, "messages": []})["dr_run"]
        assert update["deliverable"] is True

    def test_deliverable_preset_from_configurable(self):
        update = initialize_node({"dr_run": {}, "messages": []}, config={"configurable": {"dr_deliverable": True}})["dr_run"]
        assert update["deliverable"] is True

    def test_no_preset_means_interactive(self):
        update = initialize_node({"dr_run": {}, "messages": []})["dr_run"]
        assert update["deliverable"] is False

    def test_route_after_initialize(self):
        assert route_after_initialize({"dr_run": {"deliverable": True}}) == "plan"
        assert route_after_initialize({"dr_run": {}}) == "research"
        assert route_after_initialize({}) == "research"


class TestScheduleQueries:
    def test_must_cover_first_and_deterministic(self):
        reqs = [_req("r2", must_cover=False), _req("r1"), _req("r3")]
        scheduled = schedule_queries(reqs, "general", {})
        assert [s.requirement_id for s in scheduled] == ["r1", "r3", "r2"]

    def test_query_includes_entities_and_window(self):
        req = _req("r1", entities=["ACME"], window={"from": "2024-01-01", "to": None})
        (item,) = schedule_queries([req], "general", {})
        assert "requirement r1" in item.query
        assert "ACME" in item.query
        assert "2024-01-01" in item.query

    def test_class_tagged_requirement_gets_tool_hints(self, monkeypatch):
        monkeypatch.setattr(schedule_mod, "profile_tools_for_evidence_class", lambda cls, allowlist, connectors: ["fred_series"])
        monkeypatch.setattr(schedule_mod, "load_profile", lambda name: {"tool_allowlist": ["fred_series"]})
        (item,) = schedule_queries([_req("r1", evidence_class="primary_data")], "financial", {})
        assert item.tool_names == ["fred_series"]
        (line,) = format_schedule_lines([item])
        assert "use one of: fred_series" in line
        assert "query:" in line

    def test_unknown_profile_fails_closed(self, monkeypatch):
        def _boom(name):
            raise ValueError(name)

        monkeypatch.setattr(schedule_mod, "load_profile", _boom)
        (item,) = schedule_queries([_req("r1", evidence_class="academic")], "nope", {})
        assert item.tool_names == []


class TestPlanRequirementsNode:
    @pytest.mark.asyncio
    async def test_extracts_schedules_and_injects_directive(self, monkeypatch):
        async def _fake_extract(question, *, current_date=None):
            assert question == "Q?"
            return [_req("r1", evidence_class="any")], {"input_tokens": 1, "output_tokens": 1, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 1}

        monkeypatch.setattr("dr_core.graph.plan.extract_requirements", _fake_extract)
        state = {"messages": [HumanMessage(content="Q?")], "dr_run": {"deliverable": True, "question": "Q?"}}
        result = await plan_requirements_node(state)
        assert result["dr_run"]["requirements_planned"] is True
        assert result["dr_run"]["active_requirement_ids"] == ["r1"]
        assert "r1" in result["dr_requirements"]
        (directive,) = result["messages"]
        assert directive.additional_kwargs.get("hide_from_ui") is True
        assert "<dr_research_plan>" in directive.content
        assert "r1" in directive.content

    @pytest.mark.asyncio
    async def test_no_question_still_marks_planned(self, monkeypatch):
        called = False

        async def _fake_extract(question):
            nonlocal called
            called = True
            return [], {}

        monkeypatch.setattr("dr_core.graph.plan.extract_requirements", _fake_extract)
        result = await plan_requirements_node({"messages": [], "dr_run": {"deliverable": True}})
        assert result["dr_run"]["requirements_planned"] is True
        assert result["dr_run"]["active_requirement_ids"] == []
        assert not called
        assert "messages" not in result


class TestPlanCoverageSkipsWhenPlanned:
    @pytest.mark.asyncio
    async def test_no_reextraction_after_entry_planning(self, monkeypatch):
        async def _fail_extract(question, *, current_date=None):
            raise AssertionError("must not re-extract after entry planning")

        async def _fake_map(active_requirements, claims, coverage, *, sources=None):
            return {}, {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0, "calls": 0}

        monkeypatch.setattr("dr_core.graph.plan.extract_requirements", _fail_extract)
        monkeypatch.setattr("dr_core.graph.plan.map_coverage", _fake_map)
        req = _req("r1")
        state = {
            "messages": [HumanMessage(content="Q?")],
            "dr_run": {"requirements_planned": True, "active_requirement_ids": ["r1"]},
            "dr_requirements": {"r1": req.model_dump(mode="json")},
            "dr_claims": {},
            "dr_coverage": {},
        }
        result = await plan_coverage_node(state)
        # extraction skipped: active ids come from dr_run, never overwritten
        assert "active_requirement_ids" not in result["dr_run"]


class TestPlannedCorrectiveRetry:
    def test_corrective_message_carries_scheduled_queries(self):
        req = _req("r1", evidence_class="any")
        state = {
            "dr_claims": {},
            "dr_sources": {},
            "dr_requirements": {"r1": req.model_dump(mode="json")},
            "dr_coverage": {},
            "dr_run": {"deliverable": True, "active_requirement_ids": ["r1"], "gate_retries": 0},
        }
        result = eligibility_gate(state)
        assert result["dr_run"]["gate_decision"] == "research"
        (corrective,) = result["messages"]
        assert "<dr_corrective>" in corrective.content
        assert "query: requirement r1" in corrective.content


class TestBaselineRunScoping:
    def test_prior_run_claims_never_eligible_or_cited(self):
        claim = {
            "claim_id": "c-old",
            "text": "old claim",
            "source_id": "s1",
            "verification": {"status": "supported", "complete": True},
        }
        state = {
            "dr_claims": {"c-old": claim},
            "dr_sources": {"s1": {"source_id": "s1", "url_or_id": "https://x", "source_system": "web"}},
            "dr_requirements": {},
            "dr_coverage": {},
            "dr_run": {"deliverable": True, "baseline_claim_ids": ["c-old"], "active_requirement_ids": [], "gate_retries": 2},
        }
        result = eligibility_gate(state)
        run_update = result["dr_run"]
        # retries exhausted forces proceed; the baseline claim must not be cited
        assert run_update["gate_decision"] == "render"
        assert run_update["citation_ordinals"] == {}

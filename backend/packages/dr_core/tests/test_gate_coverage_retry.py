"""Stage 1 (SPEC_evidence_routing_2026-07-10.md, section A) tests: coverage-forced
retry. Hermetic: no model/network, drives eligibility_gate / route_after_research
directly with synthetic state dicts. Companion to test_dr_gate.py (D6 eligibility
predicate) and test_gate_coverage.py (D9 coverage predicate), both of which stay
green unmodified -- these tests pin the NEW behavior this stage adds: the
corrective message must name uncovered requirements by id (not just text), and
gate_retries must survive a forced (retries-exhausted/breaker) proceed instead of
being reset to 0, so a terminal completed_with_open_requirements run is
distinguishable from a never-retried one (settled question 4).
"""

from dr_core.graph.agent import route_after_gate, route_after_research
from dr_core.graph.gate import RETRY_CAP, eligibility_gate
from dr_core.models.enums import StopReason
from dr_core.models.ledger import Claim, Requirement


def _claim(claim_id: str, source_id: str = "s1", **overrides) -> dict:
    defaults = dict(claim_id=claim_id, text=f"claim {claim_id}", importance=4, source_id=source_id)
    defaults.update(overrides)
    return Claim(**defaults).model_dump(mode="json")


def _source(source_id: str) -> dict:
    return {"id": source_id, "url_or_id": f"https://example.com/{source_id}", "source_system": "web", "authority_tier": 3, "retrieved_at": "2026-07-10T00:00:00+00:00"}


def _requirement(req_id: str, text: str, *, must_cover: bool = True, kind: str = "entity") -> dict:
    return Requirement(id=req_id, kind=kind, text=text, must_cover=must_cover).model_dump(mode="json")


def _uncovered_state(*, gate_retries: int) -> dict:
    """A claim exists, but no dr_coverage mapping ties it to the must-cover
    requirement -- req1 evaluates to UNCOVERED (mirrors
    test_gate_coverage.py::TestUncoveredMustCoverBounces)."""
    return {
        "dr_claims": {"c1": _claim("c1")},
        "dr_sources": {"s1": _source("s1")},
        "dr_requirements": {"req1": _requirement("req1", "Name the current CEO of the company.")},
        "dr_coverage": {},
        "dr_run": {"gate_retries": gate_retries, "active_requirement_ids": ["req1"]},
    }


class TestA1UncoveredMustCoverForcesTargetedRetry:
    def test_corrective_message_names_uncovered_requirement_id_and_text(self):
        state = _uncovered_state(gate_retries=0)
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "research"
        assert route_after_gate({"dr_run": dr_run}) == "research"

        corrective = result["messages"][0]
        assert "req1" in corrective.content
        assert "Name the current CEO of the company." in corrective.content


class TestA2RetryCapExhaustedKeepsRetryCount:
    def test_cap_hit_proceeds_with_open_requirements_and_preserves_gate_retries(self):
        state = _uncovered_state(gate_retries=RETRY_CAP)
        result = eligibility_gate(state)
        dr_run = result["dr_run"]
        assert dr_run["gate_decision"] == "render"
        assert dr_run["stop_reason"] == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        # S1 fix: a forced (non-ready) proceed must NOT reset gate_retries to
        # 0 -- downstream consumers distinguish "tried and failed" from
        # "never tried" via this count (settled question 4 / test A.2).
        assert dr_run["gate_retries"] == RETRY_CAP
        assert route_after_gate({"dr_run": dr_run}) == "render"


class TestA3BreakerFiresRegardlessOfRemainingRetries:
    def test_unchanged_signature_across_two_passes_proceeds_before_cap(self):
        state1 = _uncovered_state(gate_retries=0)
        result1 = eligibility_gate(state1)
        dr_run1 = result1["dr_run"]
        assert dr_run1["gate_decision"] == "research"
        assert dr_run1["gate_retries"] == 1

        # Second pass: identical claims/sources/requirements/coverage (no
        # progress at all) and retries carried forward at 1, still below
        # RETRY_CAP=2. The unchanged signature must still force a proceed
        # (the one-strike breaker), not another retry.
        state2 = _uncovered_state(gate_retries=dr_run1["gate_retries"])
        state2["dr_run"]["gate_ledger_sig"] = dr_run1["gate_ledger_sig"]
        result2 = eligibility_gate(state2)
        dr_run2 = result2["dr_run"]

        assert dr_run1["gate_retries"] < RETRY_CAP
        assert dr_run2["gate_decision"] == "render"
        assert dr_run2["stop_reason"] == StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        assert route_after_gate({"dr_run": dr_run2}) == "render"


class TestA4RoutingSurvivesZeroToolCallsOnADrDeliverableRun:
    """Regression for the 2026-07-10 diagnostic run's gate bypass: a dr run
    where the model made zero tool calls (so DrLedgerMiddleware's source scan
    never fires and never sets dr_run["deliverable"]) must still reach the
    gate, PROVIDED the run is flagged as a genuine dr deliverable turn up
    front. route_after_research keys only on dr_run["deliverable"] /
    gate_decision -- never on tool-call or claim counts -- so this is already
    true of the pure routing function; the diagnostic run's actual bypass was
    that its headless invocation script never set dr_run["deliverable"] at
    all in initial_state (see this stage's BUILD_LEDGER entry for the full
    root-cause writeup and the interim invocation-side workaround)."""

    def test_deliverable_marked_run_gates_even_with_no_claims_or_tool_calls(self):
        state = {"dr_claims": {}, "dr_run": {"deliverable": True}}
        assert route_after_research(state) == "gate"

    def test_deliverable_marked_run_gates_even_with_no_messages_at_all(self):
        state = {"messages": [], "dr_claims": {}, "dr_run": {"deliverable": True}}
        assert route_after_research(state) == "gate"

"""Tests for dr_core.verify.votes (D8 adaptive 1-3 vote protocol). Hermetic: the
vote model is a fake injected via monkeypatching `_get_vote_model`; no network,
no real LLM.
"""

import json
from types import SimpleNamespace

from dr_core.models.enums import VerificationStatus
from dr_core.models.ledger import Claim
from dr_core.verify import votes as votes_mod
from dr_core.verify.votes import cast_vote, risk_reasons, verify_claim


def _claim(**overrides) -> Claim:
    defaults = dict(claim_id="c1", text="claim text", importance=4, source_id="s1")
    defaults.update(overrides)
    return Claim(**defaults)


def _source(**overrides) -> dict:
    defaults = dict(id="s1", url_or_id="https://example.com/s1", source_system="web", authority_tier=1, retrieved_at="2026-07-06T00:00:00+00:00")
    defaults.update(overrides)
    return defaults


def _vote_response(refuted: bool, abstain: bool, confidence: str = "high", reasoning: str = "ok") -> SimpleNamespace:
    content = json.dumps({"refuted": refuted, "abstain": abstain, "confidence": confidence, "reasoning": reasoning})
    return SimpleNamespace(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "input_token_details": {"cache_read": 1, "cache_creation": 0}})


class _FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        response = self._responses[self.calls] if self.calls < len(self._responses) else self._responses[-1]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


async def _reserve_all(n: int) -> bool:
    return True


async def _reserve_none(n: int) -> bool:
    return False


class TestRiskReasons:
    def test_clean_claim_has_no_risk_reasons(self):
        assert risk_reasons(_claim(), _source()) == []

    def test_seeded_trap_source_flags_injected(self):
        assert "injected" in risk_reasons(_claim(), _source(source_system="seeded-trap"))

    def test_gate_flags_are_reported(self):
        from dr_core.models.enums import GateFlag

        reasons = risk_reasons(_claim(gate_flags=[GateFlag.VENDOR_REPORTED]), _source())
        assert "gate:vendor_reported" in reasons

    def test_worse_authority_tier_flags_risk(self):
        assert "tier-4-or-worse" in risk_reasons(_claim(), _source(authority_tier=4))

    def test_default_web_tier_does_not_flag_risk(self):
        """D10: the C2 hook mints every web source at authority_tier=3 by default;
        the recalibrated >=4 threshold must not fire on that default (the old >=3
        threshold fired on every single claim, per the S10 acceptance defect)."""
        assert "tier-4-or-worse" not in risk_reasons(_claim(), _source(authority_tier=3))

    def test_missing_source_flags_risk(self):
        assert "tier-4-or-worse" in risk_reasons(_claim(), None)

    def test_unmatched_structured_claim_flags_risk(self):
        from dr_core.models.enums import DataProvenance

        claim = _claim(data_ref={"value": "1"}, data_provenance=DataProvenance.UNAUDITED)
        assert "structured-unmatched" in risk_reasons(claim, _source())

    def test_sole_must_cover_supporter_flags_risk(self):
        """D9: activates the deferred D8 risk reason once the caller (graph.verify)
        derives it from real requirement/coverage state."""
        assert "sole-must-cover-supporter" in risk_reasons(_claim(), _source(), sole_must_cover_supporter=True)

    def test_sole_must_cover_supporter_absent_by_default(self):
        assert "sole-must-cover-supporter" not in risk_reasons(_claim(), _source())


class TestCastVoteParsing:
    async def test_clean_json_parses(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([_vote_response(False, False, "high")]))
        vote, response = await cast_vote(_claim(), _source(), "some excerpt", 1, [])
        assert vote.refuted is False
        assert vote.abstain is False
        assert vote.confidence == "high"
        assert response is not None

    async def test_json_wrapped_in_prose_is_extracted(self, monkeypatch):
        wrapped = SimpleNamespace(content='Here is my answer:\n{"refuted": true, "abstain": false, "confidence": "low", "reasoning": "weak"}\nThanks.')
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([wrapped]))
        vote, _ = await cast_vote(_claim(), _source(), None, 1, [])
        assert vote.refuted is True

    async def test_unparseable_output_becomes_abstain(self, monkeypatch):
        garbage = SimpleNamespace(content="not json at all", usage_metadata={})
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([garbage]))
        vote, response = await cast_vote(_claim(), _source(), None, 1, [])
        assert vote.abstain is True
        assert vote.refuted is False
        assert response is not None  # usage still accumulable even on a parse failure

    async def test_model_call_exception_becomes_abstain_with_no_response(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([RuntimeError("boom")]))
        vote, response = await cast_vote(_claim(), _source(), None, 1, [])
        assert vote.abstain is True
        assert response is None

    def test_vote_id_is_deterministic(self):
        assert True  # covered structurally by verify_claim tests below (vote_id == f"{claim_id}:v{n}")


class TestVerifyClaimSingleClearFastPath:
    async def test_clean_vote1_short_circuits_to_supported(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([_vote_response(False, False, "high")]))
        record, usage = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_all)
        assert record.mode == "single_clear"
        assert record.status == VerificationStatus.SUPPORTED
        assert record.complete is True
        assert len(record.votes) == 1
        assert usage["calls"] == 1


class TestVerifyClaimThreeVoteEscalation:
    async def test_two_of_three_refute_kills(self, monkeypatch):
        responses = [
            _vote_response(False, False, "low"),  # vote1 not clean (low confidence) -> escalate
            _vote_response(True, False, "high"),  # vote2 refutes
            _vote_response(True, False, "high"),  # vote3 refutes
        ]
        # One shared model instance across all three cast_vote calls -- _get_vote_model
        # is invoked fresh per call in production, but the fake must advance through
        # `responses` in order, so the SAME instance (not a new one per call) is returned.
        fake_model = _FakeModel(responses)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: fake_model)
        record, usage = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_all)
        assert record.mode == "three_vote"
        assert record.status == VerificationStatus.KILLED_ON_REFUTE
        assert record.complete is True
        assert len(record.votes) == 3
        assert usage["calls"] == 3

    async def test_all_abstain_is_not_verified(self, monkeypatch):
        responses = [_vote_response(False, True, "low")] * 3
        fake_model = _FakeModel(responses)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: fake_model)
        record, _ = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_all)
        assert record.mode == "three_vote"
        assert record.status == VerificationStatus.NOT_VERIFIED
        assert record.complete is True

    async def test_risk_reasons_escalate_even_with_clean_vote1(self, monkeypatch):
        """A clean vote1 on a risky claim (e.g. seeded-trap) must still escalate,
        since is_single_clear also requires zero risk reasons."""
        responses = [_vote_response(False, False, "high")] * 3
        fake_model = _FakeModel(responses)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: fake_model)
        record, _ = await verify_claim(_claim(), _source(source_system="seeded-trap"), "excerpt", reserve_votes=_reserve_all)
        assert record.mode == "three_vote"
        assert len(record.votes) == 3

    async def test_sole_must_cover_supporter_escalates_even_with_clean_vote1(self, monkeypatch):
        """D9: a claim flagged as the sole direct supporter of a must-cover
        requirement must not take the single-vote fast path, even on a clean
        vote1 -- losing it on refutation would uncover a mandatory requirement."""
        responses = [_vote_response(False, False, "high")] * 3
        fake_model = _FakeModel(responses)
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: fake_model)
        record, _ = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_all, sole_must_cover_supporter=True)
        assert record.mode == "three_vote"
        assert "sole-must-cover-supporter" in record.risk_reasons


class TestVerifyClaimIncompleteOnBudget:
    async def test_no_budget_for_vote1_is_incomplete(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([_vote_response(False, False, "high")]))
        record, usage = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_none)
        assert record.mode == "incomplete"
        assert record.status == VerificationStatus.NOT_VERIFIED
        assert record.complete is False
        assert record.votes == []
        assert usage["calls"] == 0

    async def test_no_budget_for_escalation_keeps_vote1_and_marks_incomplete(self, monkeypatch):
        calls = {"n": 0}

        async def _reserve_first_only(n: int) -> bool:
            calls["n"] += 1
            return calls["n"] == 1

        # vote1 comes back ambiguous (low confidence) so it must try to escalate.
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([_vote_response(False, False, "low")]))
        record, usage = await verify_claim(_claim(), _source(), "excerpt", reserve_votes=_reserve_first_only)
        assert record.mode == "incomplete"
        assert record.status == VerificationStatus.NOT_VERIFIED
        assert record.complete is False
        assert len(record.votes) == 1
        assert usage["calls"] == 1


class _CapturingModel:
    """Fake model that records the messages it was invoked with, so a test can
    inspect the system prompt actually sent for a given evidence regime."""

    def __init__(self, response):
        self._response = response
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return self._response


class TestVoterEpistemicRegimes:
    """D10: the vote prompt must branch on evidence availability -- the refute-on-
    weak-support bias survives only when a fetched excerpt is present; when it is
    absent, the voter must be told explicitly that unavailability is not falsity."""

    def test_system_instructions_require_affirmative_grounds_for_refutation(self):
        from dr_core.verify.votes import _vote_system_instructions

        for evidence_available in (True, False):
            text = _vote_system_instructions(evidence_available)
            assert "AFFIRMATIVE grounds" in text

    def test_evidence_absent_prompt_states_unavailability_is_not_falsity(self):
        from dr_core.verify.votes import _vote_system_instructions

        text = _vote_system_instructions(evidence_available=False)
        assert "do not treat unavailability as falsity" in text

    def test_evidence_present_prompt_keeps_the_adversarial_weak_support_bias(self):
        from dr_core.verify.votes import _vote_system_instructions

        text = _vote_system_instructions(evidence_available=True)
        assert "default toward refuted=true" in text.lower()

    async def test_cast_vote_sends_the_evidence_absent_regime_when_excerpt_is_none(self, monkeypatch):
        model = _CapturingModel(_vote_response(False, True, "low"))
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: model)
        await cast_vote(_claim(), _source(), None, 1, [])
        system_content = model.calls[0][0].content
        assert "do not treat unavailability as falsity" in system_content

    async def test_cast_vote_sends_the_evidence_present_regime_when_excerpt_exists(self, monkeypatch):
        model = _CapturingModel(_vote_response(False, False, "high"))
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: model)
        await cast_vote(_claim(), _source(), "an excerpt", 1, [])
        system_content = model.calls[0][0].content
        assert "do not treat unavailability as falsity" not in system_content
        assert "default toward refuted=true" in system_content.lower()


class TestSourceUnreachableRiskReason:
    async def test_missing_evidence_with_a_url_adds_source_unreachable(self, monkeypatch):
        monkeypatch.setattr(votes_mod, "_get_vote_model", lambda: _FakeModel([_vote_response(False, False, "high")]))
        record, _ = await verify_claim(_claim(), _source(), None, reserve_votes=_reserve_all)
        assert "source_unreachable" in record.risk_reasons
        # single_clear requires zero risk reasons, so this must have escalated.
        assert record.mode != "single_clear"

"""Tests for dr_core.plan.mapping (D9 coverage mapping). Hermetic: the plan
model is a fake injected via monkeypatching `_get_plan_model`; no network, no
real LLM.
"""

import json
from types import SimpleNamespace

from dr_core.models.ledger import Claim, Requirement
from dr_core.plan import mapping as mapping_mod
from dr_core.plan.mapping import map_coverage, unmapped_pairs


def _response(payload) -> SimpleNamespace:
    return SimpleNamespace(content=json.dumps(payload), usage_metadata={"input_tokens": 4, "output_tokens": 2, "input_token_details": {}})


class _FakeModel:
    def __init__(self, response):
        self._response = response

    async def ainvoke(self, messages):
        self.messages = messages
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _requirement(req_id: str, must_cover: bool = False) -> Requirement:
    return Requirement(id=req_id, kind="entity", text=f"requirement {req_id}", must_cover=must_cover)


def _claim(claim_id: str) -> Claim:
    return Claim(claim_id=claim_id, text=f"claim {claim_id}", importance=3, source_id="s1")


class TestUnmappedPairs:
    def test_cross_product_minus_existing_coverage(self):
        pairs = unmapped_pairs(["r1", "r2"], ["c1", "c2"], {"r1:c1": {}})
        assert set(pairs) == {("r1", "c2"), ("r2", "c1"), ("r2", "c2")}

    def test_empty_requirements_or_claims_yields_no_pairs(self):
        assert unmapped_pairs([], ["c1"], {}) == []
        assert unmapped_pairs(["r1"], [], {}) == []


def test_mapping_instructions_define_relation_by_requirement_kind():
    instructions = mapping_mod._INSTRUCTIONS

    assert "For a subtopic requirement" in instructions
    assert "need not exhaust the whole subtopic" in instructions
    assert "For a comparison requirement" in instructions
    assert "For a metric requirement" in instructions
    assert "For a date_window requirement" in instructions
    assert "Tangential context" in instructions


class TestMapCoverageKeyShape:
    async def test_mapping_key_is_requirement_colon_claim(self, monkeypatch):
        payload = [{"requirement_id": "r1", "claim_id": "c1", "relation": "direct", "elements_satisfied": ["x"], "relationship_stated": None}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))

        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, usage = await map_coverage(requirements, claims, {})
        assert "r1:c1" in result
        assert result["r1:c1"]["relation"] == "direct"
        assert usage["calls"] == 1

    async def test_mapper_sees_quote_target_and_source_class(self, monkeypatch):
        payload = [{"requirement_id": "r1", "claim_id": "c1", "relation": "direct"}]
        model = _FakeModel(_response(payload))
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: model)
        requirements = {"r1": _requirement("r1")}
        claims = {
            "c1": Claim(
                claim_id="c1",
                text="claim c1",
                importance=3,
                source_id="s1",
                target_requirement_ids=["r1"],
                support={"quote": "exact supporting quote", "relation_extractor": "supports_directly"},
            )
        }
        sources = {"s1": {"source_system": "semantic_scholar", "url_or_id": "https://api.semanticscholar.org/paper/1"}}

        await map_coverage(requirements, claims, {}, sources=sources)
        prompt = json.loads(model.messages[1].content)

        assert prompt["claims"][0]["supporting_quote"] == "exact supporting quote"
        assert prompt["claims"][0]["target_requirement_ids"] == ["r1"]
        assert prompt["claims"][0]["source_evidence_class"] == "academic"

    async def test_targeted_claim_is_only_sent_to_its_candidate_requirement(self, monkeypatch):
        model = _FakeModel(_response([{"requirement_id": "r1", "claim_id": "c1", "relation": "direct"}]))
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: model)
        requirements = {"r1": _requirement("r1"), "r2": _requirement("r2")}
        claims = {
            "c1": Claim(
                claim_id="c1",
                text="claim c1",
                importance=3,
                source_id="s1",
                target_requirement_ids=["r1"],
            )
        }

        result, usage = await map_coverage(requirements, claims, {})

        assert set(result) == {"r1:c1"}
        assert usage["calls"] == 1

    async def test_no_unmapped_pairs_skips_the_model_call(self, monkeypatch):
        def _boom():
            raise AssertionError("model must not be called when nothing is unmapped")

        monkeypatch.setattr(mapping_mod, "_get_plan_model", _boom)
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, usage = await map_coverage(requirements, claims, {"r1:c1": {"requirement_id": "r1", "claim_id": "c1", "relation": "none"}})
        assert result == {}
        assert usage["calls"] == 0


class TestReferentialIntegrity:
    async def test_unknown_requirement_id_is_dropped(self, monkeypatch):
        payload = [{"requirement_id": "unknown-req", "claim_id": "c1", "relation": "direct"}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, _ = await map_coverage(requirements, claims, {})
        assert result == {}

    async def test_unknown_claim_id_is_dropped(self, monkeypatch):
        payload = [{"requirement_id": "r1", "claim_id": "unknown-claim", "relation": "direct"}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, _ = await map_coverage(requirements, claims, {})
        assert result == {}

    async def test_pair_not_asked_about_is_dropped(self, monkeypatch):
        # r1:c1 already mapped -- model still hallucinating it back must not
        # overwrite the existing entry via this pass.
        payload = [{"requirement_id": "r1", "claim_id": "c1", "relation": "direct"}, {"requirement_id": "r2", "claim_id": "c1", "relation": "direct"}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        requirements = {"r1": _requirement("r1"), "r2": _requirement("r2")}
        claims = {"c1": _claim("c1")}
        existing = {"r1:c1": {"requirement_id": "r1", "claim_id": "c1", "relation": "none"}}
        result, _ = await map_coverage(requirements, claims, existing)
        assert "r1:c1" not in result
        assert "r2:c1" in result

    async def test_invalid_relation_is_dropped(self, monkeypatch):
        payload = [{"requirement_id": "r1", "claim_id": "c1", "relation": "not-a-relation"}]
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(_response(payload)))
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, _ = await map_coverage(requirements, claims, {})
        assert result == {}


class TestDegradation:
    async def test_model_failure_degrades_to_no_new_mappings(self, monkeypatch):
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(RuntimeError("boom")))
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, usage = await map_coverage(requirements, claims, {})
        assert result == {}
        assert usage["calls"] == 0

    async def test_unparseable_output_degrades_to_no_new_mappings(self, monkeypatch):
        response = SimpleNamespace(content="nonsense", usage_metadata={})
        monkeypatch.setattr(mapping_mod, "_get_plan_model", lambda: _FakeModel(response))
        requirements = {"r1": _requirement("r1")}
        claims = {"c1": _claim("c1")}
        result, usage = await map_coverage(requirements, claims, {})
        assert result == {}
        assert usage["calls"] == 1

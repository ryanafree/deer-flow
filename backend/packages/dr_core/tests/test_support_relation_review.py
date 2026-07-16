import json
from types import SimpleNamespace

from dr_core.models.enums import SupportRelation
from dr_core.models.ledger import Claim, SupportRecord
from dr_core.verify import relation as relation_mod


class _Model:
    async def ainvoke(self, messages):
        return SimpleNamespace(
            content=json.dumps({"relation": "supports_directly", "reasoning": "The quote states the proposition."}),
            usage_metadata={"input_tokens": 8, "output_tokens": 4},
        )


async def test_relation_review_parses_and_accounts(monkeypatch):
    monkeypatch.setattr(relation_mod, "_get_relation_model", lambda: _Model())
    claim = Claim(
        claim_id="c1",
        text="The study reports lower volatility.",
        importance=4,
        source_id="s1",
        support=SupportRecord(
            quote="The study reports significantly lower volatility.",
            relation_extractor=SupportRelation.SUPPORTS_DIRECTLY,
        ),
    )

    reviewed, note, usage = await relation_mod.review_support_relation(claim, {"title": "Study"}, "full excerpt")

    assert reviewed == SupportRelation.SUPPORTS_DIRECTLY
    assert note == "The quote states the proposition."
    assert usage["calls"] == 1


async def test_relation_review_failure_fails_closed(monkeypatch):
    class _Broken:
        async def ainvoke(self, messages):
            raise RuntimeError("unavailable")

    monkeypatch.setattr(relation_mod, "_get_relation_model", lambda: _Broken())
    claim = Claim(
        claim_id="c1",
        text="Claim",
        importance=4,
        source_id="s1",
        support=SupportRecord(quote="supporting words", relation_extractor=SupportRelation.SUPPORTS_DIRECTLY),
    )

    reviewed, note, usage = await relation_mod.review_support_relation(claim, None, None)

    assert reviewed is None
    assert note is None
    assert usage["calls"] == 0

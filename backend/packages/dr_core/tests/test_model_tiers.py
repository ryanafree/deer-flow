"""Tests for dr_core.models.tiers (Item 3): the single owner of the three
model-tier env knobs. No network, no live model calls -- create_chat_model is
mocked at the factory boundary everywhere a model object would be constructed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import dr_core.plan.extraction as extraction_mod
import dr_core.plan.mapping as mapping_mod
import dr_core.verify.votes as votes_mod
from dr_core import accounting
from dr_core.models import tiers


def _mock_create_chat_model(monkeypatch):
    """Install a fake deerflow.models module and return the captured-args dict."""
    captured = {}

    def _fake(name):
        captured["name"] = name
        return object()

    monkeypatch.setitem(sys.modules, "deerflow.models", types.SimpleNamespace(create_chat_model=_fake))
    return captured


class TestResolverDefaults:
    def test_plan_model_name_default(self, monkeypatch):
        monkeypatch.delenv("DR_PLAN_MODEL", raising=False)
        assert tiers.plan_model_name() == "or-sonnet"

    def test_plan_model_name_env_override(self, monkeypatch):
        monkeypatch.setenv("DR_PLAN_MODEL", "some-other-model")
        assert tiers.plan_model_name() == "some-other-model"

    def test_verify_model_name_default(self, monkeypatch):
        monkeypatch.delenv("DR_VERIFY_MODEL", raising=False)
        assert tiers.verify_model_name() == "claude-verify"

    def test_verify_model_name_env_override(self, monkeypatch):
        monkeypatch.setenv("DR_VERIFY_MODEL", "or-sonnet")
        assert tiers.verify_model_name() == "or-sonnet"

    def test_synth_model_name_default(self, monkeypatch):
        monkeypatch.delenv("DR_SYNTH_MODEL", raising=False)
        assert tiers.synth_model_name() == "claude-top"

    def test_synth_model_name_env_override(self, monkeypatch):
        monkeypatch.setenv("DR_SYNTH_MODEL", "some-synth-model")
        assert tiers.synth_model_name() == "some-synth-model"


class TestCallSitesUseTiersModule:
    def test_extraction_builds_model_through_tiers(self, monkeypatch):
        captured = _mock_create_chat_model(monkeypatch)
        monkeypatch.setattr(tiers, "plan_model_name", lambda: "sentinel-plan-model")
        extraction_mod._get_plan_model()
        assert captured["name"] == "sentinel-plan-model"

    def test_mapping_builds_model_through_tiers(self, monkeypatch):
        captured = _mock_create_chat_model(monkeypatch)
        monkeypatch.setattr(tiers, "plan_model_name", lambda: "sentinel-plan-model")
        mapping_mod._get_plan_model()
        assert captured["name"] == "sentinel-plan-model"

    def test_votes_builds_model_through_tiers(self, monkeypatch):
        captured = _mock_create_chat_model(monkeypatch)
        monkeypatch.setattr(tiers, "verify_model_name", lambda: "sentinel-verify-model")
        votes_mod._get_vote_model()
        assert captured["name"] == "sentinel-verify-model"


class TestAccountingCoversEveryTierAlias:
    def test_default_tier_names_are_priced(self, monkeypatch):
        monkeypatch.delenv("DR_PLAN_MODEL", raising=False)
        monkeypatch.delenv("DR_VERIFY_MODEL", raising=False)
        monkeypatch.delenv("DR_SYNTH_MODEL", raising=False)
        names = {tiers.plan_model_name(), tiers.verify_model_name(), tiers.synth_model_name()}
        assert names == {"or-sonnet", "claude-verify", "claude-top"}
        for name in names:
            assert name in accounting.MODEL_TIER_IDS, f"{name} missing from MODEL_TIER_IDS"
            resolved = accounting.MODEL_TIER_IDS[name]
            assert resolved in accounting.PRICING, f"{name} -> {resolved} missing from PRICING"


class TestGrepGuardNoScatteredEnvReads:
    def test_no_module_outside_tiers_reads_the_three_env_vars(self):
        package_root = Path(tiers.__file__).resolve().parent.parent  # dr_core/
        tiers_path = Path(tiers.__file__).resolve()
        needles = ("DR_PLAN_MODEL", "DR_VERIFY_MODEL", "DR_SYNTH_MODEL")
        offenders = []
        for path in package_root.rglob("*.py"):
            if path.resolve() == tiers_path:
                continue
            if "tests" in path.parts:
                continue
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                if needle in text:
                    offenders.append(f"{path}: {needle}")
        assert not offenders, "env knobs read outside tiers.py:\n" + "\n".join(offenders)

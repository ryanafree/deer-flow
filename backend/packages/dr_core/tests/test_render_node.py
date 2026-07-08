"""Tests for the real render node (S3: REVIEW_FINISH_PLAN_2026-07-06.md Part II/III).

Hermetic: no model/network, drives render_node directly with synthetic state dicts,
then runs the real report_lint over the generated run folder -- the ledger's own
acceptance spec for this step (zero hard findings on every fixture).
"""

import datetime
import importlib
import json
import re
import sys
from pathlib import Path

from dr_core.graph.render import render_node
from dr_core.lint import report_lint
from dr_core.models.enums import CitationStatus, GateFlag, StopReason, VerificationStatus
from dr_core.models.ledger import Claim, SupportRecord, VerificationRecord


def _source(source_id: str, title: str, **overrides) -> dict:
    defaults = dict(
        id=source_id,
        url_or_id=f"https://example.test/{source_id}",
        source_system="web",
        title=title,
        authority_tier=2,
        retrieved_at="2026-07-06T00:00:00-05:00",
    )
    defaults.update(overrides)
    return defaults


def _supported_claim(claim_id: str, text: str, source_id: str) -> dict:
    return Claim(
        claim_id=claim_id,
        text=text,
        importance=3,
        source_id=source_id,
        support=SupportRecord(quote="the primary source states this directly", relation_extractor="supports_directly"),
        verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
    ).model_dump(mode="json")


def _contested_claim(claim_id: str, text: str, source_id: str) -> dict:
    return Claim(
        claim_id=claim_id,
        text=text,
        importance=3,
        source_id=source_id,
        support=SupportRecord(quote="a vendor benchmark reports this", relation_extractor="supports_directly"),
        verification=VerificationRecord(status=VerificationStatus.SUPPORTED, complete=True),
        gate_flags=[GateFlag.VENDOR_REPORTED],
    ).model_dump(mode="json")


def _not_verified_claim(claim_id: str, text: str, source_id: str) -> dict:
    return Claim(claim_id=claim_id, text=text, importance=3, source_id=source_id).model_dump(mode="json")


def _excluded_claim(claim_id: str, text: str, source_id: str) -> dict:
    return Claim(
        claim_id=claim_id,
        text=text,
        importance=3,
        source_id=source_id,
        citation_status=CitationStatus.NOT_FOUND,
    ).model_dump(mode="json")


def _happy_path_state(runs_dir: str) -> dict:
    return {
        "dr_claims": {
            "c1": _supported_claim("c1", "The observatory recorded a twelve percent increase in nightly visitors during 2025.", "s1"),
            "c2": _contested_claim("c2", "The vendor reported a doubling of throughput after the upgrade.", "s2"),
            "c3": _not_verified_claim("c3", "Local officials say the new policy will take effect next year.", "s1"),
        },
        "dr_sources": {
            "s1": _source("s1", "Example Observatory Annual Report"),
            "s2": _source("s2", "Vendor Benchmark Disclosure"),
        },
        "dr_run": {
            "citation_ordinals": {"c1": 1, "c2": 2, "c3": 3},
            "question": "What changed at the observatory in 2025?",
            "profile": "general",
            "runs_dir": runs_dir,
        },
    }


def _run_lint(folder: str) -> int:
    old_argv = sys.argv
    sys.argv = ["report_lint.py", folder]
    try:
        return report_lint.main()
    finally:
        sys.argv = old_argv


class TestHappyPath:
    def test_happy_path_writes_full_run_folder(self, tmp_path):
        state = _happy_path_state(str(tmp_path))
        result = render_node(state)
        run_dir = Path(result["dr_run"]["run_dir"])
        assert result["dr_run"]["render_completed"] is True
        for name in ("report.md", "claims.jsonl", "sources.jsonl", "manifest.json", "report.html"):
            assert (run_dir / name).is_file(), name

        lines = (run_dir / "claims.jsonl").read_text().splitlines()
        assert [json.loads(line)["claim_id"] for line in lines] == ["c1", "c2", "c3"]

    def test_happy_path_lints_clean(self, tmp_path, capsys):
        state = _happy_path_state(str(tmp_path))
        result = render_node(state)
        run_dir = result["dr_run"]["run_dir"]
        code = _run_lint(run_dir)
        out = capsys.readouterr().out
        assert code == 0, out


class TestCapHitPath:
    def _state(self, runs_dir):
        return {
            "dr_claims": {
                "c1": _excluded_claim("c1", "A fabricated statistic about market share was recorded.", "s1"),
            },
            "dr_sources": {"s1": _source("s1", "Example Source")},
            "dr_run": {
                "citation_ordinals": {},
                "stop_reason": StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value,
                "question": "What is the current market share?",
                "profile": "general",
                "runs_dir": runs_dir,
            },
        }

    def test_cap_hit_renders_unsubstantiated_section_and_lints_clean(self, tmp_path, capsys):
        state = self._state(str(tmp_path))
        result = render_node(state)
        run_dir = result["dr_run"]["run_dir"]
        report_md = Path(run_dir, "report.md").read_text()
        assert "## Unsubstantiated" in report_md
        assert "c1" in report_md
        assert "excluded" in report_md

        code = _run_lint(run_dir)
        out = capsys.readouterr().out
        assert code == 0, out


class TestZeroEligiblePath:
    def _state(self, runs_dir):
        return {
            "dr_claims": {},
            "dr_sources": {},
            "dr_run": {
                "citation_ordinals": {},
                "stop_reason": StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value,
                "question": "Did anything happen?",
                "profile": "general",
                "runs_dir": runs_dir,
            },
        }

    def test_zero_eligible_renders_no_findings_report_and_lints_clean(self, tmp_path, capsys):
        state = self._state(str(tmp_path))
        result = render_node(state)
        run_dir = result["dr_run"]["run_dir"]
        report_md = Path(run_dir, "report.md").read_text()
        assert re.search(r"^#\s+\S", report_md, re.M)
        assert "## Executive summary" in report_md
        assert "## Conclusion" in report_md
        assert "No claim" in report_md or "No recorded claim" in report_md

        code = _run_lint(run_dir)
        out = capsys.readouterr().out
        assert code == 0, out


class TestDeterminism:
    def test_two_runs_produce_byte_identical_report_md(self, tmp_path, monkeypatch):
        # dr_core.run.__init__ does `from dr_core.run.write_run import write_run`,
        # which shadows the package's `write_run` submodule attribute with the
        # function of the same name -- `import dr_core.run.write_run as x` would
        # resolve to that function via attribute lookup. importlib.import_module
        # looks up sys.modules by dotted name instead, sidestepping the shadow.
        write_run_mod = importlib.import_module("dr_core.run.write_run")

        class _FixedDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 7, 6, 12, 0, 0, tzinfo=datetime.UTC)

        monkeypatch.setattr(write_run_mod, "datetime", _FixedDatetime)

        state1 = _happy_path_state(str(tmp_path / "run1"))
        state2 = _happy_path_state(str(tmp_path / "run2"))
        result1 = render_node(state1)
        result2 = render_node(state2)

        report1 = Path(result1["dr_run"]["run_dir"], "report.md").read_text()
        report2 = Path(result2["dr_run"]["run_dir"], "report.md").read_text()
        assert report1 == report2


class TestManifestAccounting:
    """S9: render_node must pass accounting/depth/profile/verify_mode through into
    the written manifest.json (REVIEW_FINISH_PLAN_2026-07-06.md Part III gaps)."""

    def test_manifest_carries_accounting_depth_profile(self, tmp_path):
        state = _happy_path_state(str(tmp_path))
        state["messages"] = []
        result = render_node(state)
        manifest = json.loads(Path(result["dr_run"]["run_dir"], "manifest.json").read_text())

        assert "accounting" in manifest
        assert set(manifest["accounting"].keys()) == {"phases", "totals", "dollar_cost"}
        assert set(manifest["accounting"]["phases"].keys()) == {"research", "verify"}
        assert manifest["depth"] == "quick"
        assert manifest["profile"] == "general"
        assert manifest["verify_mode"] == "off"

    def test_manifest_verify_mode_reflects_dr_run(self, tmp_path):
        state = _happy_path_state(str(tmp_path))
        state["dr_run"]["verify_mode"] = "adaptive_1_3"
        result = render_node(state)
        manifest = json.loads(Path(result["dr_run"]["run_dir"], "manifest.json").read_text())
        assert manifest["verify_mode"] == "adaptive_1_3"

    def test_manifest_accounting_reflects_message_usage(self, tmp_path):
        from langchain_core.messages import AIMessage

        state = _happy_path_state(str(tmp_path))
        state["messages"] = [AIMessage(content="x", usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})]
        result = render_node(state)
        manifest = json.loads(Path(result["dr_run"]["run_dir"], "manifest.json").read_text())
        assert manifest["accounting"]["phases"]["research"]["input_tokens"] == 7
        assert manifest["accounting"]["phases"]["research"]["output_tokens"] == 3

    def test_unknown_profile_degrades_without_crashing(self, tmp_path):
        state = _happy_path_state(str(tmp_path))
        state["dr_run"]["profile"] = "not-a-real-profile"
        result = render_node(state)
        manifest = json.loads(Path(result["dr_run"]["run_dir"], "manifest.json").read_text())
        assert manifest["accounting"]["dollar_cost"] is None


class TestOpenMustCoverUnsubstantiated:
    """D9: render's Unsubstantiated section additionally enumerates ACTIVE
    must-cover requirements the gate froze as not fully COVERED
    (dr_run["must_cover_states"]), by id + text + evidence state."""

    def _state(self, runs_dir: str) -> dict:
        from dr_core.models.ledger import Requirement

        state = _happy_path_state(runs_dir)
        state["dr_run"]["stop_reason"] = StopReason.COMPLETED_WITH_OPEN_REQUIREMENTS.value
        state["dr_run"]["must_cover_states"] = {"req1": "uncovered", "req2": "covered"}
        state["dr_requirements"] = {
            "req1": Requirement(id="req1", kind="entity", text="Name the acting director.", must_cover=True).model_dump(mode="json"),
            "req2": Requirement(id="req2", kind="entity", text="State the founding year.", must_cover=True).model_dump(mode="json"),
        }
        return state

    def test_open_must_cover_requirement_named_in_unsubstantiated(self, tmp_path, capsys):
        state = self._state(str(tmp_path))
        result = render_node(state)
        run_dir = result["dr_run"]["run_dir"]
        report_md = Path(run_dir, "report.md").read_text()
        unsubstantiated = report_md.split("## Unsubstantiated")[1].split("## Conclusion")[0]

        assert "req1" in unsubstantiated
        assert "Name the acting director." in unsubstantiated
        assert "uncovered" in unsubstantiated
        # req2 is COVERED -- must NOT be listed as an open gap.
        assert "req2" not in unsubstantiated

        code = _run_lint(run_dir)
        out = capsys.readouterr().out
        assert code == 0, out


class TestOrdinalInvariant:
    def test_markers_resolve_to_the_nth_claims_jsonl_line(self, tmp_path):
        state = _happy_path_state(str(tmp_path))
        result = render_node(state)
        run_dir = result["dr_run"]["run_dir"]
        claims_lines = Path(run_dir, "claims.jsonl").read_text().splitlines()
        report_md = Path(run_dir, "report.md").read_text()
        body = report_md.split("## Appendix")[0]

        markers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
        assert markers == list(range(1, len(claims_lines) + 1))

        for i, line in enumerate(claims_lines, start=1):
            claim = json.loads(line)
            marker = f"[{i}]"
            idx = body.index(marker)
            line_start = body.rfind("\n", 0, idx) + 1
            paragraph = body[line_start:idx].lower()
            first_word = claim["text"].split()[0].lower()
            assert first_word in paragraph or "according to" in paragraph

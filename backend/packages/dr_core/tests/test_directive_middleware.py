"""Tests for DrResearchDirectiveMiddleware and the source-class-mismatch lint
warning (SPEC_evidence_routing_2026-07-10.md, Stage 3 acceptance tests C.9-C.10).

HERMETIC: no live model, no network. Test 9 exercises the middleware directly
(pattern per test_profile_tool_middleware.py: build a ModelRequest, capture what
a stub handler receives). Test 10 exercises dr_core.lint.report_lint.main() over
a synthetic run folder, mirroring test_report_lint_eligibility.py's fixture style.
"""

from __future__ import annotations

import json

from dr_core.connectors.registry import load_connectors
from dr_core.graph.directive_middleware import DIRECTIVE_MARKER, DrResearchDirectiveMiddleware
from dr_core.lint.report_lint import main
from dr_core.models import Claim, CoverageMapping, Requirement, Source
from dr_core.models.enums import CoverageRelation, RequirementKind
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage


def _middleware() -> DrResearchDirectiveMiddleware:
    return DrResearchDirectiveMiddleware(connectors=load_connectors())


def _model_request(profile: str | None, system_message: SystemMessage | None = None) -> ModelRequest:
    state = {} if profile is None else {"dr_run": {"profile": profile}}
    return ModelRequest(model=None, system_message=system_message, messages=[], tools=[], state=state)


class TestDrResearchDirectiveMiddleware:
    def test_marker_and_mandate_present_in_model_request(self):
        mw = _middleware()
        captured = {}

        def handler(req):
            captured["request"] = req
            return "model-response"

        out = mw.wrap_model_call(_model_request("financial", SystemMessage(content="base prompt")), handler)
        assert out == "model-response"
        sm = captured["request"].system_message
        assert sm is not None
        content = sm.content
        assert DIRECTIVE_MARKER in content
        assert "tool-gathered" in content or "parametric memory" in content

    def test_financial_profile_names_tools_from_two_evidence_classes(self):
        mw = _middleware()
        captured = {}

        def handler(req):
            captured["request"] = req
            return "model-response"

        mw.wrap_model_call(_model_request("financial", SystemMessage(content="base prompt")), handler)
        content = captured["request"].system_message.content
        # openalex (academic) and fred (primary_data) are both in the financial
        # profile's tool_allowlist and both carry an evidence_class in connectors.yaml.
        assert "openalex" in content
        assert "fred" in content

    def test_preferred_tools_section_names_actual_bound_tool_names(self):
        """D11 P0 (MYTHOS_REVIEW_2026-07-11.md finding 3): the directive used
        to name bare connector identifiers (e.g. "openalex") that are not
        themselves callable tool names. It must now also surface the ACTUAL
        bound tool name (or, for MCP connectors whose live tool name isn't
        knowable ahead of time, the tool_name_prefix pattern) -- resolved via
        connectors.binding.describe_connector_tools."""
        mw = _middleware()
        captured = {}

        def handler(req):
            captured["request"] = req
            return "model-response"

        mw.wrap_model_call(_model_request("financial", SystemMessage(content="base prompt")), handler)
        content = captured["request"].system_message.content
        # fred backs the Stage-C fred_series tool -- dr_core knows the exact
        # literal bound name.
        assert "fred_series" in content
        # openalex rides the Stage-C academic_search fallback chain
        # (connectors/tools.py) -- dr_core knows this exact literal bound
        # name too, unlike a connector reached only through the generic MCP
        # loader (whose live prefixed name isn't knowable ahead of time).
        assert "academic_search" in content

    def test_works_with_no_base_system_message(self):
        mw = _middleware()
        captured = {}

        def handler(req):
            captured["request"] = req
            return "model-response"

        mw.wrap_model_call(_model_request("general", system_message=None), handler)
        assert DIRECTIVE_MARKER in captured["request"].system_message.content

    async def test_async_variant_also_injects(self):
        mw = _middleware()
        captured = {}

        async def handler(req):
            captured["request"] = req
            return "model-response"

        out = await mw.awrap_model_call(_model_request("legal", SystemMessage(content="base")), handler)
        assert out == "model-response"
        assert DIRECTIVE_MARKER in captured["request"].system_message.content


# ---------------------------------------------------------------------------
# Test 10: source-class-mismatch lint warning
# ---------------------------------------------------------------------------

CLEAN_REPORT = """# Mismatch test report

## Executive summary

Volatility clusters because of persistent shocks to the conditional variance process [1].

## Conclusion

The evidence supports the clustering mechanism.
"""


def _write_mismatch_run_folder(folder, *, source_system, url_or_id):
    (folder / "report.md").write_text(CLEAN_REPORT)
    claim = Claim(
        claim_id="c1",
        text="Volatility clusters because of persistent shocks to the conditional variance process.",
        importance=4,
        source_id="s1",
    )
    source = Source(
        id="s1",
        url_or_id=url_or_id,
        source_system=source_system,
        authority_tier=2,
        retrieved_at="2026-07-10T00:00:00-05:00",
    )
    requirement = Requirement(
        id="r1",
        kind=RequirementKind.SUBTOPIC,
        text="Explain the academic mechanism behind volatility clustering.",
        must_cover=True,
        evidence_class="academic",
    )
    mapping = CoverageMapping(requirement_id="r1", claim_id="c1", relation=CoverageRelation.DIRECT)

    with open(folder / "claims.jsonl", "w") as fh:
        fh.write(json.dumps(claim.model_dump(mode="json")) + "\n")
    with open(folder / "sources.jsonl", "w") as fh:
        fh.write(json.dumps(source.model_dump(mode="json")) + "\n")
    with open(folder / "requirements.jsonl", "w") as fh:
        fh.write(json.dumps(requirement.model_dump(mode="json")) + "\n")
    with open(folder / "coverage.jsonl", "w") as fh:
        fh.write(json.dumps(mapping.model_dump(mode="json")) + "\n")


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["report_lint.py", str(tmp_path)])
    return main()


def test_academic_requirement_citing_news_source_warns_not_fails(tmp_path, monkeypatch, capsys):
    _write_mismatch_run_folder(tmp_path, source_system="web", url_or_id="https://example-news-blog.com/article")
    exit_code = _run(tmp_path, monkeypatch)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "LINT-WARN" in out
    assert "source-class-mismatch" in out


def test_academic_requirement_citing_academic_source_has_no_mismatch_warning(tmp_path, monkeypatch, capsys):
    _write_mismatch_run_folder(tmp_path, source_system="arxiv", url_or_id="arxiv:2101.00001")
    exit_code = _run(tmp_path, monkeypatch)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "source-class-mismatch" not in out

import sys
from datetime import date
from pathlib import Path

import pytest
from dr_core.graph.gate import eligibility_gate
from dr_core.graph.render import render_node
from dr_core.lint import report_lint
from dr_core.models.ledger import Requirement
from dr_core.structured import vix_fomc

B2 = (
    "Using primary data sources, characterize how the VIX term structure "
    "(VIX vs the CBOE 3-month volatility index) behaved around FOMC announcement dates "
    "from January 2023 to the present, and how the current spread compares to that history."
)


def test_compile_spec_binds_both_series_and_current_date():
    spec = vix_fomc.compile_spec(B2, as_of=date(2026, 7, 15))

    assert spec["series_ids"] == ["VIXCLS", "VXVCLS"]
    assert spec["end"] == "2026-07-15"
    assert spec["spread_formula"] == "VIXCLS - VXVCLS"


def test_align_spreads_uses_event_date_then_nearest_prior_common_date():
    vix = [
        {"date": "2023-01-31", "value": "20"},
        {"date": "2023-02-01", "value": "21"},
        {"date": "2023-02-02", "value": "22"},
    ]
    vxv = [
        {"date": "2023-01-31", "value": "19"},
        {"date": "2023-02-02", "value": "20"},
    ]

    rows, current = vix_fomc.align_spreads(
        vix,
        vxv,
        announcement_dates=(date(2023, 2, 1),),
        as_of=date(2023, 2, 2),
    )

    assert rows[0].observation_date == date(2023, 1, 31)
    assert rows[0].spread == 1
    assert current.observation_date == date(2023, 2, 2)
    assert current.spread == 2


def _requirements():
    items = [
        Requirement(id="metric", kind="metric", text="State dated VIX and VIX3M values.", must_cover=True, evidence_class="primary_data"),
        Requirement(id="compare", kind="comparison", text="Compare the current spread with FOMC history.", entities=["VIXCLS", "VXVCLS"], must_cover=True, evidence_class="primary_data"),
    ]
    return {item.id: item.model_dump(mode="json") for item in items}


def _payload(series: str):
    values = {
        "VIXCLS": [("2023-01-31", "20"), ("2023-02-01", "21"), ("2023-02-02", "22")],
        "VXVCLS": [("2023-01-31", "19"), ("2023-02-01", "19.5"), ("2023-02-02", "20")],
    }
    return {"ok": True, "observations": [{"date": d, "value": v} for d, v in values[series]]}


async def test_prepare_node_mints_matched_claims_and_deterministic_mappings(monkeypatch):
    monkeypatch.setattr(vix_fomc, "_call_fred_series", lambda series, year: _payload(series))
    state = {
        "dr_run": {
            "question": B2,
            "current_date": "2023-02-02",
            "active_requirement_ids": ["metric", "compare"],
            "tool_call_count": 0,
            "tool_call_counts": {},
        },
        "dr_requirements": _requirements(),
    }

    result = await vix_fomc.prepare_vix_fomc_node(state)

    assert result["dr_run"]["b2_data_status"] == "complete"
    assert result["dr_run"]["tool_call_counts"] == {"fred_series": 2}
    assert {source["source_system"] for source in result["dr_sources"].values()} == {"fred"}
    assert all(claim["data_provenance"] == "matched" for claim in result["dr_claims"].values())
    assert all(set(claim["data_ref"]["series_ids"]) == {"VIXCLS", "VXVCLS"} for claim in result["dr_claims"].values())
    assert len(result["dr_coverage"]) == len(result["dr_claims"]) * 2


async def test_prepare_gate_render_roundtrip_contains_dated_numeric_primary_claim(monkeypatch, tmp_path):
    monkeypatch.setattr(vix_fomc, "_call_fred_series", lambda series, year: _payload(series))
    base_run = {
        "question": B2,
        "current_date": "2023-02-02",
        "active_requirement_ids": ["metric", "compare"],
        "gate_retries": 0,
        "runs_dir": str(tmp_path),
    }
    prepared = await vix_fomc.prepare_vix_fomc_node({"dr_run": base_run, "dr_requirements": _requirements()})
    state = {
        "dr_run": {**base_run, **prepared["dr_run"]},
        "dr_requirements": _requirements(),
        "dr_sources": prepared["dr_sources"],
        "dr_claims": prepared["dr_claims"],
        "dr_coverage": prepared["dr_coverage"],
        "messages": [],
    }
    gate = eligibility_gate(state)
    state["dr_run"].update(gate["dr_run"])

    rendered = render_node(state)
    run_dir = Path(rendered["dr_run"]["run_dir"])
    report = (run_dir / "report.md").read_text()

    assert gate["dr_run"]["requirements_covered"] == 2
    assert "VIXCLS" in report and "VXVCLS" in report
    assert "2023-02-01" in report and "1.50 index points" in report
    assert "[1]" in report
    old_argv = sys.argv
    sys.argv = ["report_lint.py", str(run_dir)]
    try:
        assert report_lint.main() == 0
    finally:
        sys.argv = old_argv


async def test_missing_series_is_explicitly_unavailable_and_renderable(monkeypatch, tmp_path):
    def _fetch(series, year):
        return _payload(series) if series == "VIXCLS" else {"ok": False, "error": "series unavailable"}

    monkeypatch.setattr(vix_fomc, "_call_fred_series", _fetch)
    prepared = await vix_fomc.prepare_vix_fomc_node({"dr_run": {"question": B2, "current_date": "2023-02-02"}, "dr_requirements": {}})
    state = {
        "dr_run": {**prepared["dr_run"], "question": B2, "runs_dir": str(tmp_path), "citation_ordinals": {}},
        "dr_sources": {},
        "dr_claims": {},
        "dr_requirements": {},
        "dr_coverage": {},
        "messages": [],
    }

    rendered = render_node(state)
    report = Path(rendered["dr_run"]["run_dir"], "report.md").read_text()

    assert prepared["dr_run"]["b2_data_status"] == "unavailable"
    assert "Primary data unavailable" in report
    assert "series unavailable" in report


def test_render_blocks_nonterminal_b2_contract(tmp_path):
    with pytest.raises(ValueError, match="no terminal status"):
        render_node(
            {
                "dr_run": {"b2_contract_required": True, "question": B2, "runs_dir": str(tmp_path), "citation_ordinals": {}},
                "dr_sources": {},
                "dr_claims": {},
                "dr_requirements": {},
                "dr_coverage": {},
            }
        )

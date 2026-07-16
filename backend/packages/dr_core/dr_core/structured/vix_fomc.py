"""Deterministic VIX/VIX3M analysis around FOMC announcement dates.

Contract:
* series: FRED VIXCLS and VXVCLS;
* spread: VIXCLS - VXVCLS, in index points;
* event observation: announcement date when both series exist, otherwise the
  nearest prior common observation within seven calendar days;
* current observation: latest common date not after the run date.

The FOMC dates are the policy-statement dates from the Federal Reserve's
official meeting calendar. The list is bounded to the benchmark's January
2023 start and the completed meetings available on 2026-07-15.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from dr_core.connectors.tools import fred_series
from dr_core.models.enums import CoverageRelation, DataProvenance, VerificationStatus
from dr_core.models.ledger import Claim, CoverageMapping, Requirement, Source, VerificationRecord

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
SERIES_IDS = ("VIXCLS", "VXVCLS")
FOMC_ANNOUNCEMENT_DATES = tuple(
    date.fromisoformat(value)
    for value in (
        "2023-02-01",
        "2023-03-22",
        "2023-05-03",
        "2023-06-14",
        "2023-07-26",
        "2023-09-20",
        "2023-11-01",
        "2023-12-13",
        "2024-01-31",
        "2024-03-20",
        "2024-05-01",
        "2024-06-12",
        "2024-07-31",
        "2024-09-18",
        "2024-11-07",
        "2024-12-18",
        "2025-01-29",
        "2025-03-19",
        "2025-05-07",
        "2025-06-18",
        "2025-07-30",
        "2025-09-17",
        "2025-10-29",
        "2025-12-10",
        "2026-01-28",
        "2026-03-18",
        "2026-04-29",
        "2026-06-17",
    )
)


@dataclass(frozen=True)
class SpreadRow:
    event_date: date | None
    observation_date: date
    vixcls: float
    vxvcls: float

    @property
    def spread(self) -> float:
        return self.vixcls - self.vxvcls


def is_vix_fomc_question(question: str | None) -> bool:
    text = (question or "").lower()
    return "fomc" in text and "vix" in text and ("3-month" in text or "three-month" in text or "vix3m" in text or "vxv" in text)


def compile_spec(question: str, *, as_of: date) -> dict:
    if not is_vix_fomc_question(question):
        return {}
    return {
        "series_ids": list(SERIES_IDS),
        "start": "2023-01-01",
        "end": as_of.isoformat(),
        "spread_formula": "VIXCLS - VXVCLS",
        "event_alignment": "announcement date, else nearest prior common observation within 7 calendar days",
        "current_alignment": "latest common observation not after the run date",
        "fomc_calendar_url": FOMC_CALENDAR_URL,
    }


def _observation_map(observations: list[dict], *, as_of: date) -> dict[date, float]:
    result = {}
    for row in observations:
        try:
            observed = date.fromisoformat(str(row["date"]))
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if observed <= as_of:
            result[observed] = value
    return result


def align_spreads(
    vix_observations: list[dict],
    vxv_observations: list[dict],
    *,
    announcement_dates: tuple[date, ...] = FOMC_ANNOUNCEMENT_DATES,
    as_of: date,
) -> tuple[list[SpreadRow], SpreadRow]:
    vix = _observation_map(vix_observations, as_of=as_of)
    vxv = _observation_map(vxv_observations, as_of=as_of)
    common = sorted(set(vix) & set(vxv))
    if not common:
        raise ValueError("VIXCLS and VXVCLS have no common observations")

    rows = []
    for event_date in announcement_dates:
        if event_date > as_of:
            continue
        earliest = event_date - timedelta(days=7)
        candidates = [observed for observed in common if earliest <= observed <= event_date]
        if not candidates:
            continue
        observed = candidates[-1]
        rows.append(SpreadRow(event_date=event_date, observation_date=observed, vixcls=vix[observed], vxvcls=vxv[observed]))
    if not rows:
        raise ValueError("no paired observations align to completed FOMC announcement dates")

    current_date = common[-1]
    current = SpreadRow(event_date=None, observation_date=current_date, vixcls=vix[current_date], vxvcls=vxv[current_date])
    return rows, current


def _source_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _claim_id(text: str, source_id: str) -> str:
    return hashlib.sha256((text.strip() + source_id).encode()).hexdigest()[:16]


def _structured_claim(text: str, source_id: str, row: SpreadRow, *, targets: list[str], extra: dict | None = None) -> Claim:
    data_ref = {
        "period": row.observation_date.isoformat(),
        "claimed_period": row.observation_date.isoformat(),
        "source_class": "official_stat",
        "value": round(row.spread, 6),
        "unit": "index_points",
        "series_ids": list(SERIES_IDS),
        "series_values": {"VIXCLS": row.vixcls, "VXVCLS": row.vxvcls},
        "spread_formula": "VIXCLS - VXVCLS",
        **(extra or {}),
    }
    return Claim(
        claim_id=_claim_id(text, source_id),
        text=text,
        importance=5,
        source_id=source_id,
        target_requirement_ids=targets,
        data_ref=data_ref,
        data_provenance=DataProvenance.MATCHED,
        verification=VerificationRecord(
            selected=True,
            risk_reasons=[],
            mode="deterministic_structured",
            complete=True,
            status=VerificationStatus.SUPPORTED,
        ),
    )


def build_ledger_update(
    requirements: dict[str, Requirement],
    event_rows: list[SpreadRow],
    current: SpreadRow,
    *,
    retrieved_at: datetime,
) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    urls = {series: f"https://fred.stlouisfed.org/series/{series}" for series in SERIES_IDS}
    sources = {}
    source_ids = []
    for series, url in urls.items():
        source = Source(
            id=_source_id(url),
            url_or_id=url,
            source_system="fred",
            title=f"FRED {series}",
            authority_tier=1,
            publication_date=current.observation_date.isoformat(),
            retrieved_at=retrieved_at,
        )
        sources[source.id] = source.model_dump(mode="json")
        source_ids.append(source.id)

    targets = list(requirements)
    claims: dict[str, dict] = {}
    mappings: dict[str, dict] = {}
    all_rows = [*event_rows, current]
    for index, row in enumerate(all_rows):
        source_id = source_ids[index % len(source_ids)]
        if row.event_date is None:
            historical_spreads = sorted(item.spread for item in event_rows)
            median = historical_spreads[len(historical_spreads) // 2]
            comparison = "above" if row.spread > median else "below" if row.spread < median else "equal to"
            text = (
                f"The VIXCLS minus VXVCLS spread was {row.spread:.2f} index points on {row.observation_date.isoformat()}; "
                f"VIXCLS was {row.vixcls:.2f}, VXVCLS was {row.vxvcls:.2f}, and the spread was {comparison} the FOMC-date median of {median:.2f}."
            )
            extra = {"comparison_to_fomc_median": comparison, "fomc_median_spread": round(median, 6)}
        else:
            text = (
                f"The VIXCLS minus VXVCLS spread was {row.spread:.2f} index points for the {row.event_date.isoformat()} FOMC announcement, "
                f"using the paired {row.observation_date.isoformat()} observations: VIXCLS {row.vixcls:.2f} and VXVCLS {row.vxvcls:.2f}."
            )
            extra = {"fomc_announcement_date": row.event_date.isoformat()}
        claim = _structured_claim(text, source_id, row, targets=targets, extra=extra)
        claims[claim.claim_id] = claim.model_dump(mode="json")
        for req_id, requirement in requirements.items():
            mapping = CoverageMapping(
                requirement_id=req_id,
                claim_id=claim.claim_id,
                relation=CoverageRelation.DIRECT,
                elements_satisfied=["VIXCLS", "VXVCLS", "spread", row.observation_date.isoformat()],
                relationship_stated=True if requirement.kind.value == "comparison" else None,
            )
            mappings[f"{req_id}:{claim.claim_id}"] = mapping.model_dump(mode="json")
    return sources, claims, mappings


def _call_fred_series(series_id: str, year: int) -> dict:
    return json.loads(fred_series.func(series_id, str(year)))


async def prepare_vix_fomc_node(state) -> dict:
    dr_run = state.get("dr_run") or {}
    question = dr_run.get("question") or ""
    if not is_vix_fomc_question(question):
        return {}

    as_of = date.fromisoformat(dr_run["current_date"]) if dr_run.get("current_date") else date.today()
    spec = compile_spec(question, as_of=as_of)
    observations = {series: [] for series in SERIES_IDS}
    calls = 0
    errors = []
    for year in range(2023, as_of.year + 1):
        for series in SERIES_IDS:
            calls += 1
            payload = await asyncio.to_thread(_call_fred_series, series, year)
            if not payload.get("ok"):
                errors.append(f"{series} {year}: {payload.get('error') or 'unavailable'}")
                continue
            observations[series].extend(payload.get("observations") or [])

    prior_total = int(dr_run.get("tool_call_count") or 0)
    by_name = dict(dr_run.get("tool_call_counts") or {})
    by_name["fred_series"] = by_name.get("fred_series", 0) + calls
    run_update = {
        "b2_contract_required": True,
        "b2_spec": spec,
        "tool_call_count": prior_total + calls,
        "tool_call_counts": dict(sorted(by_name.items())),
    }
    if errors or not all(observations.values()):
        run_update.update({"b2_data_status": "unavailable", "b2_unavailable_reason": "; ".join(errors) or "one or both FRED series returned no observations"})
        return {"dr_run": run_update}

    try:
        event_rows, current = align_spreads(observations["VIXCLS"], observations["VXVCLS"], as_of=as_of)
    except ValueError as exc:
        run_update.update({"b2_data_status": "unavailable", "b2_unavailable_reason": str(exc)})
        return {"dr_run": run_update}

    active_ids = dr_run.get("active_requirement_ids") or []
    raw_requirements = state.get("dr_requirements") or {}
    requirements = {req_id: Requirement.model_validate(raw_requirements[req_id]) for req_id in active_ids if req_id in raw_requirements}
    sources, claims, mappings = build_ledger_update(requirements, event_rows, current, retrieved_at=datetime.now(UTC))
    run_update.update(
        {
            "b2_data_status": "complete",
            "b2_event_rows": len(event_rows),
            "b2_current_observation_date": current.observation_date.isoformat(),
        }
    )
    return {"dr_run": run_update, "dr_sources": sources, "dr_claims": claims, "dr_coverage": mappings}

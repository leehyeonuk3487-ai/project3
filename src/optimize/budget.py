"""예산 ↔ 보장 최적화.

지자체의 피보험 모집단(현역 청년)과 급부 스케줄, 시도별 20대 남성 발생률을
결합해 **연간 기대 청구액**을 산출하고, 예산 안에서 보장 mix를 최적화한다.
모집단 N은 병무청·인구추계 기반으로 자동 산출(population.py)하되 명시적
입력도 허용한다.

기대 청구액(항목별):
  lump(사망)     : N × rate_py × payout
  lump(후유장해) : N × rate_py × payout × 지급률보정계수   ← 과대추정 수정
  per_event      : N × rate_py × payout
  per_day        : N × rate_py × payout × min(평균재원일수, 한도)

  rate_py = (20대 남성 × 해당 시도) 1,000명당 발생률 / 1000

⚠️ 발생률은 반드시 '20대 남성 × 해당 지역' 셀을 쓴다(전체 평균 금지).
   정신질환·수술비 등 미매칭 항목 제외 → 핵심 항목 기준의 하한 근사.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..data import benefits
from ..models import population, rates

DEFAULT_LOADING = 1.25
# 예산 제약 시 보장 우선순위(사망 > 후유장해 > 질병사망 > 질병후유장해 > 골절 > 입원)
DEFAULT_PRIORITY = ["death_injury", "disability", "death_disease",
                    "disease_disability", "fracture", "hospitalization"]


def resolve_population(schedule_name: str, year: int = 2024,
                       population_override: int | None = None) -> tuple[int, str]:
    """모집단 N 결정. 명시 입력 우선, 광역이면 인구추계로 자동 산출."""
    if population_override is not None:
        return int(population_override), "입력"
    sido = benefits.SCHEDULE_SIDO[schedule_name]
    if benefits.IS_PROVINCE_WIDE.get(schedule_name):
        return int(round(population.conscript_stock(sido, year))), "추계(시도)"
    # 시군구: 시도 추정은 과대 → 명시 입력 권장. 없으면 시도값을 반환하되 표기.
    return int(round(population.conscript_stock(sido, year))), "추계(시도·과대주의)"


def _avg_los(sex: str = "남자") -> float:
    """평균 재원일수(15-24/25-34 평균, 한도 적용)."""
    from ..data.reshape import read_wide_pivot
    los = read_wide_pivot(config.DISCHARGE_DIR / "average_length_of_stay.csv")
    y = int(los["year"].max())
    sub = los[(los["dim"] == sex) & (los["metric"].str.contains("재원", na=False))
              & (los["year"] == y) & los["level"].isin(["15-24세", "25-34세"])]
    return float(min(sub["value"].mean(), benefits.HOSPITAL_DAY_CAP))


@dataclass
class ClaimEstimate:
    schedule: str
    sido: str
    year: int
    population: int
    population_source: str
    by_item: pd.DataFrame
    total_claims: float
    per_capita_claims: float
    premium_per_capita: float


def expected_claims(
    schedule_name: str,
    population_override: int | None = None,
    year: int = 2024,
    coverage_scale: dict[str, float] | None = None,
    loading: float = DEFAULT_LOADING,
) -> ClaimEstimate:
    """급부 스케줄·모집단 → 연간 기대 청구액."""
    sched = benefits.SCHEDULES[schedule_name]
    sido = benefits.SCHEDULE_SIDO[schedule_name]
    pop, pop_src = resolve_population(schedule_name, year, population_override)
    scale = coverage_scale or {}
    los = _avg_los()

    # 후유장해 지급률 보정계수(외상/질병 각각)
    payout_ratio = {
        "disability": rates.severe_disability_payout_ratio("trauma"),
        "disease_disability": rates.severe_disability_payout_ratio("nontrauma"),
    }

    rows = []
    for item, spec in sched.items():
        rate_py = rates.conscript_item_rate(item, sido) / 1000.0
        if not np.isfinite(rate_py):
            continue
        amount = spec["amount"] * scale.get(item, 1.0)
        if spec["payout_type"] == "per_day":
            payout = amount * los
        elif item in payout_ratio:           # 후유장해 지급률 가중
            payout = amount * payout_ratio[item]
        else:
            payout = amount
        exp_events = pop * rate_py
        rows.append({
            "coverage_item": item,
            "label": config.COVERAGE_ITEMS[item]["label"],
            "track": config.COVERAGE_ITEMS[item]["track"],
            "rate_per_1000py": round(rate_py * 1000, 4),
            "expected_events": round(exp_events, 2),
            "payout_per_event": round(payout, 0),
            "expected_claims": round(exp_events * payout, 0),
        })
    by_item = pd.DataFrame(rows)
    total = float(by_item["expected_claims"].sum())
    return ClaimEstimate(
        schedule=schedule_name, sido=sido, year=year, population=pop,
        population_source=pop_src, by_item=by_item, total_claims=total,
        per_capita_claims=total / pop if pop else float("nan"),
        premium_per_capita=(total / pop) * loading if pop else float("nan"),
    )


def optimize_under_budget(
    schedule_name: str,
    annual_budget: float,
    population_override: int | None = None,
    year: int = 2024,
    loading: float = DEFAULT_LOADING,
    priority: list[str] | None = None,
) -> dict:
    """연간 예산 한도 내 보장 mix 배분(우선순위 greedy)."""
    priority = priority or DEFAULT_PRIORITY
    full = expected_claims(schedule_name, population_override, year, loading=loading)
    cost = dict(zip(full.by_item["coverage_item"], full.by_item["expected_claims"]))

    budget_claims = annual_budget / loading
    scale, remaining = {}, budget_claims
    for item in priority:
        c = cost.get(item, 0.0)
        if c <= 0:
            scale[item] = 0.0
            continue
        if remaining >= c:
            scale[item] = 1.0
            remaining -= c
        else:
            scale[item] = max(0.0, remaining / c)
            remaining = 0.0
    est = expected_claims(schedule_name, population_override, year,
                          coverage_scale=scale, loading=loading)
    return {
        "coverage_scale": scale,
        "estimate": est,
        "budget": annual_budget,
        "feasible_full": full.total_claims * loading <= annual_budget,
        "full_premium_total": full.total_claims * loading,
    }


def project_budget(
    schedule_name: str,
    years: list[int],
    loading: float = DEFAULT_LOADING,
) -> pd.DataFrame:
    """연도별 모집단 변화에 따른 예산 투영(인구절벽 시뮬)."""
    rows = []
    for y in years:
        est = expected_claims(schedule_name, year=y, loading=loading)
        rows.append({
            "year": y,
            "population": est.population,
            "per_capita_premium": round(est.premium_per_capita),
            "total_premium": round(est.total_claims * loading),
        })
    return pd.DataFrame(rows)

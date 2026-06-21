"""예산 ↔ 보장 최적화.

지자체의 피보험 모집단(현역 청년)과 급부 스케줄, 시도별 기대 발생률을
결합해 **연간 기대 청구액**을 산출하고, 주어진 예산 안에서 보장 mix를
최적화한다.

기대 청구액(항목별):
  lump/per_event : N × rate_py × payout
  per_day        : N × rate_py × payout × min(평균재원일수, 한도)

  rate_py = 인구 1,000명당 연간 발생률 / 1000  (1인 1년 발생확률 근사)

⚠️ 후유장해는 지급률(3~100%) 가중 없이 정액을 적용하므로 상한 추정이다.
정신질환·수술비 등 매칭 안 되는 항목은 제외되어 총액은 핵심항목 근사다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..data import benefits
from ..models import rates

# 보험료 = 기대청구액 × 손해율 역수(사업비·이윤 부가). 단체보험 가정 loading.
DEFAULT_LOADING = 1.25


def _conscript_rate_py(item: str, sido: str) -> float:
    """해당 시도 20대 남성의 1인 1년 발생확률(근사)."""
    surf = rates.coverage_rate_surface(item)
    age_col = "discharge_age_band" if "discharge_age_band" in surf else "age_band"
    if age_col == "discharge_age_band":
        m = (surf["sido"] == sido) & (surf["sex"] == "남자") & \
            surf[age_col].isin(["15-24세", "25-34세"])
        rate_1000 = surf[m]["rate_per_1000py"].mean()
    else:
        m = (surf["sido"] == sido) & (surf["sex"] == "남자") & \
            (surf[age_col] == config.CONSCRIPT_KDCA_AGE_BAND)
        rate_1000 = surf[m]["rate_per_1000py"].mean()
    return float(rate_1000) / 1000.0


def _avg_los(sido_sex: str = "남자") -> float:
    """평균 재원일수(15-24/25-34 평균, 한도 적용)."""
    from ..data.reshape import read_wide_pivot
    los = read_wide_pivot(config.DISCHARGE_DIR / "average_length_of_stay.csv")
    y = int(los["year"].max())
    sub = los[(los["dim"] == sido_sex) & (los["metric"].str.contains("재원", na=False))
              & (los["year"] == y) & los["level"].isin(["15-24세", "25-34세"])]
    days = sub["value"].mean()
    return float(min(days, benefits.HOSPITAL_DAY_CAP))


@dataclass
class ClaimEstimate:
    schedule: str
    sido: str
    population: int
    by_item: pd.DataFrame      # 항목별 기대 발생건수·청구액
    total_claims: float        # 연간 기대 청구액(원)
    per_capita_claims: float   # 1인당 기대 청구액(원)
    premium_per_capita: float  # loading 반영 권장 보험료(원)


def expected_claims(
    schedule_name: str,
    population: int,
    coverage_scale: dict[str, float] | None = None,
    loading: float = DEFAULT_LOADING,
) -> ClaimEstimate:
    """급부 스케줄·모집단 → 연간 기대 청구액.

    coverage_scale: 항목별 보장금액 배수(최적화에서 0~1 등으로 조정). 기본 1.0.
    """
    sched = benefits.SCHEDULES[schedule_name]
    sido = benefits.SCHEDULE_SIDO[schedule_name]
    scale = coverage_scale or {}
    los = _avg_los()

    rows = []
    for item, spec in sched.items():
        rate_py = _conscript_rate_py(item, sido)
        amount = spec["amount"] * scale.get(item, 1.0)
        if spec["payout_type"] == "per_day":
            payout = amount * los
        else:
            payout = amount
        exp_events = population * rate_py
        exp_claim = exp_events * payout
        rows.append({
            "coverage_item": item,
            "label": config.COVERAGE_ITEMS[item]["label"],
            "rate_per_1000py": round(rate_py * 1000, 4),
            "expected_events": round(exp_events, 2),
            "payout_per_event": round(payout, 0),
            "expected_claims": round(exp_claim, 0),
        })
    by_item = pd.DataFrame(rows)
    total = float(by_item["expected_claims"].sum())
    return ClaimEstimate(
        schedule=schedule_name,
        sido=sido,
        population=population,
        by_item=by_item,
        total_claims=total,
        per_capita_claims=total / population if population else float("nan"),
        premium_per_capita=(total / population) * loading if population else float("nan"),
    )


def optimize_under_budget(
    schedule_name: str,
    population: int,
    annual_budget: float,
    loading: float = DEFAULT_LOADING,
    priority: list[str] | None = None,
) -> dict:
    """연간 예산 한도 내에서 보장 mix를 배분한다.

    전략: 우선순위가 높은 항목(기본: 사망>후유장해>골절>입원)부터 정액 보장을
    채우고, 예산 소진 시 마지막 항목을 비례 축소한다(greedy, 해석 가능).
    """
    priority = priority or ["death_injury", "disability", "fracture", "hospitalization"]
    full = expected_claims(schedule_name, population, loading=loading)
    cost = dict(zip(full.by_item["coverage_item"], full.by_item["expected_claims"]))

    budget_claims = annual_budget / loading  # 예산을 기대청구액 기준으로 환산
    scale = {}
    remaining = budget_claims
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
    est = expected_claims(schedule_name, population, coverage_scale=scale, loading=loading)
    return {
        "coverage_scale": scale,
        "estimate": est,
        "budget": annual_budget,
        "feasible_full": full.total_claims * loading <= annual_budget,
        "full_premium_total": full.total_claims * loading,
    }

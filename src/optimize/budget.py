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
# 고위험(생명·중대장해) 보장 — '고위험 배제 금지' 원칙상 LP에서 최소보장 floor 적용
CATASTROPHIC_ITEMS = ["death_injury", "disability", "death_disease",
                      "disease_disability"]

# 복지가중(중대도) 목적함수 가중치 — ★결과(LP 우위)에서 역산하지 않고 '독립 근거'로
# 사전 고정한다(조작 의심 차단). 근거 = 사건이 장병에게 끼치는 '건강·생애 손실 크기'
# (GBD 장애가중 류): 사망=생애상실 1.0, 영구 후유장해≈0.4, 경상(골절·입원)=경미·일시
# 0.08. ★골절과 입원은 동일 '경상 tier'로 둔다 — 둘의 상대 중대도는 데이터로 견고히
# 가릴 수 없어(지급액도 30.0만≈31.6만 거의 동일) 임의 구분으로 LP 우위를 만들지 않는다.
# 질병계열은 상해계열과 동일 tier(사망=사망, 후유장해=후유장해).
DEFAULT_WELFARE_WEIGHTS = {
    "death_injury": 1.0, "death_disease": 1.0,
    "disability": 0.4, "disease_disability": 0.4,
    "fracture": 0.08, "hospitalization": 0.08,
}


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


def optimize_benefit_lp(
    schedule_name: str,
    annual_budget: float,
    population_override: int | None = None,
    year: int = 2024,
    loading: float = DEFAULT_LOADING,
    weights: dict[str, float] | None = None,
    catastrophic_floor: float = 1.0,
    floor_items: list[str] | None = None,
) -> dict:
    """예산 제약하 **복지가중 혜택 최대화** (fractional-knapsack LP).

    그리디(optimize_under_budget)를 대체하지 않고 '추가'하는 진짜 최적화 레이어.
    미션("같은 예산으로 더 많은 장병이 더 많은 혜택")을 직접 목적함수로 옮긴다.

    의사결정변수  s_i ∈ [floor_i, 1]   (항목 i의 보장률)
    목적(최대화)  Σ  w_i · s_i · expected_events_i     (복지가중 혜택)
    제약          Σ  s_i · expected_claims_i  ≤  예산/로딩  (예산)
                  s_i ≥ floor               (고위험 항목 — 배제 금지)

    ★목적함수는 '머릿수'가 아니라 '복지가중 혜택'이다(weights=None → 중대도가중).
      머릿수 최대화(w_i=1)는 밀도=events/claims=1/payout 라 **싼 보장을 잔뜩 주는
      쪽으로 퇴화**(미션 '더 많은 혜택'과 불일치). 그래서 기본 목적을 사건 중대도
      (사망>후유장해>경상)로 가중한다(DEFAULT_WELFARE_WEIGHTS — 결과 아닌 독립 근거).
      머릿수 목적이 필요하면 weights={항목:1.0 …} 를 명시한다.
    ★고위험 floor는 목적함수와 무관하게 필수다. 사망은 희소·고액(payout 5천만)이라
      복지가중을 줘도 밀도(w/payout)가 골절의 1/100 수준 → 어떤 밀도최적화(LP·그리디)
      든 floor 없으면 사망을 버린다. floor는 꼼수가 아니라 '고위험 배제 금지'의 인코딩.
    scipy.linprog(method='highs')로 해를 구한다. 시드/입력 동일 시 재현성 보장.
    """
    from scipy.optimize import linprog

    wmap = DEFAULT_WELFARE_WEIGHTS if weights is None else weights
    floor_items = floor_items if floor_items is not None else CATASTROPHIC_ITEMS
    full = expected_claims(schedule_name, population_override, year, loading=loading)
    bi = full.by_item.reset_index(drop=True)
    items = list(bi["coverage_item"])
    events = bi["expected_events"].to_numpy(float)
    claims = bi["expected_claims"].to_numpy(float)
    w = np.array([wmap.get(it, 1.0) for it in items], float)

    budget_claims = annual_budget / loading
    lo = np.array([catastrophic_floor if it in floor_items else 0.0
                   for it in items], float)
    bounds = list(zip(lo, np.ones(len(items))))

    # 고위험 floor만으로 예산 초과 → 비현실적 floor. 해 없음을 정직히 보고.
    floor_cost = float(np.dot(lo, claims))
    if floor_cost > budget_claims + 1e-6:
        return {
            "method": "benefit_lp",
            "feasible": False,
            "reason": "catastrophic_floor가 예산을 초과 — floor 완화 필요",
            "floor_premium": floor_cost * loading,
            "budget": annual_budget,
            "coverage_scale": {it: float(lo[i]) for i, it in enumerate(items)},
        }

    # linprog는 최소화 → 목적 부호 반전(수혜자 최대화)
    res = linprog(c=-(w * events), A_ub=claims.reshape(1, -1),
                  b_ub=[budget_claims], bounds=bounds, method="highs")
    if not res.success:
        return {"method": "benefit_lp", "feasible": False,
                "reason": res.message, "budget": annual_budget}

    # 6자리로 '내림'(반올림 금지) — binding 항목이 위로 반올림되어 예산을
    # 미세 초과하는 것을 막는다(예산 제약은 절대 위반하지 않게 보수적 절사).
    scale = {it: float(np.floor(min(res.x[i], 1.0) * 1e6) / 1e6)
             for i, it in enumerate(items)}
    est = expected_claims(schedule_name, population_override, year,
                          coverage_scale=scale, loading=loading)
    beneficiaries = float(np.dot(res.x, events))
    return {
        "method": "benefit_lp",
        "feasible": True,
        "coverage_scale": scale,
        "estimate": est,
        "budget": annual_budget,
        "objective": "welfare_weighted" if weights is None else "custom_weights",
        "expected_beneficiaries": round(beneficiaries, 1),   # 머릿수(해석용)
        "weighted_benefit": round(float(np.dot(w * res.x, events)), 1),  # ★최대화 대상
        "full_premium_total": full.total_claims * loading,
        "feasible_full": full.total_claims * loading <= annual_budget,
        "catastrophic_floor": catastrophic_floor,
    }


def compare_allocation(
    schedule_name: str,
    annual_budget: float,
    population_override: int | None = None,
    year: int = 2024,
    loading: float = DEFAULT_LOADING,
    catastrophic_floor: float = 1.0,
) -> dict:
    """그리디(고정 우선순위) vs 복지가중 LP — 정직 비교.

    ★비교는 LP가 실제로 최대화하는 '복지가중 혜택'(weighted_benefit) 기준이다.
      LP는 정의상 그 목적에서 그리디 이상(>=)을 보장한다(동률 가능). 머릿수도 함께
      보고하되, LP가 머릿수를 줄이고 더 중대한 보장을 택할 수 있으므로(미션 '깊이')
      머릿수 기준 LP>=gre디는 일반적으로 성립하지 않는다 — 가중혜택으로 비교한다.
    """
    g = optimize_under_budget(schedule_name, annual_budget, population_override,
                              year, loading=loading)
    lp = optimize_benefit_lp(schedule_name, annual_budget, population_override,
                             year, loading=loading,
                             catastrophic_floor=catastrophic_floor)
    full = expected_claims(schedule_name, population_override, year, loading=loading)
    ev = dict(zip(full.by_item["coverage_item"], full.by_item["expected_events"]))
    wt = DEFAULT_WELFARE_WEIGHTS

    def _headcount(scale: dict) -> float:
        return float(sum(ev.get(it, 0.0) * s for it, s in scale.items()))

    def _welfare(scale: dict) -> float:
        return float(sum(wt.get(it, 1.0) * ev.get(it, 0.0) * s
                         for it, s in scale.items()))

    feasible = lp.get("feasible")
    g_head, g_welf = _headcount(g["coverage_scale"]), _welfare(g["coverage_scale"])
    lp_head = _headcount(lp["coverage_scale"]) if feasible else float("nan")
    lp_welf = _welfare(lp["coverage_scale"]) if feasible else float("nan")
    return {
        "budget": annual_budget,
        "objective": "welfare_weighted",
        "greedy": {"coverage_scale": g["coverage_scale"],
                   "beneficiaries": round(g_head, 1),
                   "welfare_benefit": round(g_welf, 1)},
        "lp": {"coverage_scale": lp.get("coverage_scale"),
               "beneficiaries": round(lp_head, 1),
               "welfare_benefit": round(lp_welf, 1),
               "feasible": feasible},
        # ★보장 기준: LP의 복지가중 혜택 ≥ 그리디 (+0.0로 부호있는 0 정규화).
        "lp_gain_welfare": (round(lp_welf - g_welf, 1) + 0.0) if feasible else None,
        "lp_gain_beneficiaries": (round(lp_head - g_head, 1) + 0.0)
        if feasible else None,
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

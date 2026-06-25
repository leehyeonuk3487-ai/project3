"""M6 — 시나리오·민감도 분석.

★대원칙: 현재 pricing(보험료·예산)은 M0 직접관측 백본 불변. 외인 추세 외삽은
  '미래 N년 투영' 시나리오에만 쓴다(projection.external_trend_band). 둘을 섞어
  현재가를 바꾸지 않는다 — MECE envelope 보존.

구성:
  1) sensitivity_table : 변수별 낙관/기준/보수 → 1인당 총보험료·총예산·LP 수혜자
  2) tornado           : 변수별 영향 크기(1인당 보험료 swing) → 무엇이 예산을 흔드나
  3) multiyear_band    : 인구절벽 × 외인추세 결합 → 미래 총예산 경로 밴드
  4) optimization_under_scenarios : 시나리오별 혜택최대화 LP 배분 변화

라벨 규약(예산 영향 기준): 낙관=예산부담↓, 보수=예산부담↑.
  단, 인구절벽의 '낙관(=빠른 감소)'은 장병 수 감소를 뜻하므로 미션상 바람직하지
  않음 — 라벨은 어디까지나 '예산 숫자' 방향이며 결과표에 함께 명시한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import budget, premium
from ..models import projection

DEFAULT_SCHEDULE = "경기도"
HORIZON = 2030
BASE_YEAR = 2024


def _ext_mult(year: int = HORIZON) -> dict[str, float]:
    """외인 추세 외삽 배수(낙관/기준/보수) at horizon."""
    band = projection.external_trend_band(BASE_YEAR, [year]).iloc[0]
    return {"opt": float(band["optimistic_mult"]),
            "base": float(band["base_mult"]),
            "cons": float(band["conservative_mult"])}


def _pop_levels(schedule: str, year: int = HORIZON) -> dict[str, int]:
    """인구절벽 모집단(현역 N) 낙관/기준/보수.

    기준 = 통계청 장래인구추계 중위(conscript_stock). 추계 불확실성 밴드는
    중위 감소폭의 ±배수(낙관=빠른감소 1.3×, 보수=완만 0.6×)로 둔다(고위/저위
    변형 미적재 → 명시적 가정, 0단계 보고·확정 범위).
    """
    from ..models import population
    base_n = population.conscript_stock(budget.benefits.SCHEDULE_SIDO[schedule], BASE_YEAR)
    mid_n = population.conscript_stock(budget.benefits.SCHEDULE_SIDO[schedule], year)
    drop = base_n - mid_n                       # 중위 감소량
    return {"opt": int(round(base_n - drop * 1.3)),   # 더 빠른 절벽(예산↓)
            "base": int(round(mid_n)),
            "cons": int(round(base_n - drop * 0.6))}   # 완만(예산↑)


def sensitivity_table(schedule: str = DEFAULT_SCHEDULE,
                      horizon: int = HORIZON) -> pd.DataFrame:
    """변수별 낙관/기준/보수 → 1인당 총보험료. 다른 변수는 기준 고정.

    pricing 자체는 M0 백본 — 외인추세는 death_injury 빈도배수(coverage_scale로
    근사)로만 미래 투영에 반영. 현재가(기준)는 불변.
    """
    em = _ext_mult(horizon)
    pl = _pop_levels(schedule, horizon)
    base = premium.actuarial_premium(schedule)          # 기준(현재 M0 백본)
    base_pc = base.gross_pc

    specs = {
        "보장 한도(급부)": {
            "rationale": "급부 한도 ±20%",
            "opt": lambda: premium.actuarial_premium(
                schedule, coverage_scale={k: 0.8 for k in budget.benefits.SCHEDULES[schedule]}),
            "cons": lambda: premium.actuarial_premium(
                schedule, coverage_scale={k: 1.2 for k in budget.benefits.SCHEDULES[schedule]})},
        "위험할증 α": {
            "rationale": "정규근사 84% ~ 보수 적정성",
            "opt": lambda: premium.actuarial_premium(schedule, alpha=0.84),
            "cons": lambda: premium.actuarial_premium(schedule, alpha=1.5)},
        "사업비 로딩률": {
            "rationale": "민영 손보 사업비 관행 범위",
            "opt": lambda: premium.actuarial_premium(schedule, expense_ratio=0.15),
            "cons": lambda: premium.actuarial_premium(schedule, expense_ratio=0.30)},
        "외인사망 추세(2030 투영)": {
            "rationale": f"로그선형 외삽 밴드 {BASE_YEAR}->{horizon} (점추정 아님)",
            "opt": lambda: premium.actuarial_premium(
                schedule, coverage_scale={"death_injury": em["opt"]}),
            "cons": lambda: premium.actuarial_premium(
                schedule, coverage_scale={"death_injury": em["cons"]})},
        "인구절벽(현역 N, 2030)": {
            "rationale": "통계청 장래인구추계 중위 ± 추계밴드",
            "opt": lambda: premium.actuarial_premium(schedule, population_override=pl["opt"]),
            "cons": lambda: premium.actuarial_premium(schedule, population_override=pl["cons"])},
    }
    rows = []
    for name, s in specs.items():
        lo, hi = s["opt"]().gross_pc, s["cons"]().gross_pc
        rows.append({"변수": name, "낙관(예산↓)": round(lo), "기준": round(base_pc),
                     "보수(예산↑)": round(hi), "근거": s["rationale"]})
    df = pd.DataFrame(rows)
    df.attrs["base_premium_pc"] = round(base_pc)
    df.attrs["note"] = ("기준=현재 M0 직접관측 백본(불변). 외인추세는 미래 투영에만 "
                        "반영. pricing에 외삽 주입 없음.")
    return df


def tornado(schedule: str = DEFAULT_SCHEDULE, horizon: int = HORIZON) -> pd.DataFrame:
    """변수별 1인당 보험료 swing(보수-낙관) — 예산을 가장 흔드는 변수 식별."""
    st = sensitivity_table(schedule, horizon)
    st = st.assign(swing=(st["보수(예산↑)"] - st["낙관(예산↓)"]).abs())
    st["swing%기준"] = (st["swing"] / st.attrs["base_premium_pc"] * 100).round(1)
    out = st.sort_values("swing", ascending=False).reset_index(drop=True)
    out.attrs.update(st.attrs)
    return out[["변수", "낙관(예산↓)", "기준", "보수(예산↑)", "swing", "swing%기준", "근거"]]


def multiyear_band(schedule: str = DEFAULT_SCHEDULE,
                   years: list[int] | None = None,
                   loading: float = budget.DEFAULT_LOADING) -> pd.DataFrame:
    """인구절벽 × 외인추세 결합 → 미래 총예산 경로 밴드(낙관/기준/보수).

    - 인구절벽: 중위 추계로 모든 항목 발생 모집단 스케일(예산 수준의 추세).
    - 외인추세: death_injury 청구에만 추세 배수(밴드 폭).
    - pricing 단가는 M0 고정 — 미래는 '모집단×추세'로만 움직인다.
    """
    years = years or list(range(BASE_YEAR, BASE_YEAR + 7))
    sido = budget.benefits.SCHEDULE_SIDO[schedule]
    em_band = projection.external_trend_band(BASE_YEAR, years).set_index("year")
    from ..models import population
    base_n = population.conscript_stock(sido, BASE_YEAR)

    # 기준연도 항목별 청구(M0 백본)
    est0 = budget.expected_claims(schedule, year=BASE_YEAR, loading=loading)
    bi0 = est0.by_item.set_index("coverage_item")["expected_claims"]
    di0 = float(bi0.get("death_injury", 0.0))
    other0 = float(bi0.sum() - di0)

    rows = []
    for y in years:
        pop_mult = population.conscript_stock(sido, y) / base_n
        row = {"year": y, "pop_mult": round(pop_mult, 4),
               "population": int(round(population.conscript_stock(sido, y)))}
        for sc in ("optimistic", "base", "conservative"):
            ext = float(em_band.loc[y, f"{sc}_mult"])
            total_claims = pop_mult * (other0 + di0 * ext)
            row[f"{sc}_총예산"] = round(total_claims * loading)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["note"] = projection.EXTRAPOLATION_NOTE
    df.attrs["base_total_premium"] = round(est0.total_claims * loading)
    return df


def optimization_under_scenarios(schedule: str = DEFAULT_SCHEDULE,
                                 budget_fracs=(0.6, 0.8, 1.0, 1.2)) -> pd.DataFrame:
    """예산 수준별 혜택최대화 LP 최적배분 변화 (vs 그리디 정직 비교).

    예산이 조여질수록 최적 보장 mix가 어떻게 변하고 기대 수혜자가 어떻게
    줄어드는지를 보인다 — '최적화 결과가 시나리오에 따라 어떻게 변하나'의 핵심.
    예산↑ → 수혜자↑(단조)여야 하며, LP 수혜자 ≥ 그리디(정의상 최적).
    """
    full = budget.expected_claims(schedule).total_claims * budget.DEFAULT_LOADING
    rows = []
    for frac in budget_fracs:
        B = full * frac
        cmp = budget.compare_allocation(schedule, B)
        lp = budget.optimize_benefit_lp(schedule, B)
        feasible = lp.get("feasible")
        scale = lp.get("coverage_scale", {})
        rows.append({
            "예산(full대비)": f"{frac:.0%}",
            "예산액": round(B),
            "LP_feasible": feasible,
            "LP_수혜자": cmp["lp"]["beneficiaries"] if feasible else None,
            "그리디_수혜자": cmp["greedy"]["beneficiaries"],
            "LP_추가수혜": cmp["lp_gain_beneficiaries"],
            "사망보장률": round(scale.get("death_injury", float("nan")), 2) if feasible else None,
            "입원보장률": round(scale.get("hospitalization", float("nan")), 2) if feasible else None,
        })
    df = pd.DataFrame(rows)
    df.attrs["note"] = ("예산 50%대 미만은 고위험 완전보장(floor) 비용 초과로 "
                        "LP infeasible(정직 보고). LP는 정의상 그리디 이상의 수혜자.")
    return df


# --- 안전장치: 가중치 민감도 (역산 의심 차단) -------------------------------
# ★독립 근거로 사전 고정한 '여러' 가중 셋. 결과에서 역산하지 않았음을 투명 공개.
#   W1 GBD·W2 순서형은 골절≥입원(중대도), W3는 임상 acuity(입원 admission이 더 중증)
#   관점. 어느 가정이 옳은지는 데이터로 단정 불가 → '특정 가중에서만 LP가 이긴다'를
#   숨기지 않고 전부 보여준다(조작 아님의 증거).
WEIGHT_SCHEMES = {
    "W0 머릿수(균등)": {},   # 빈 dict → 모두 1.0 (머릿수)
    "W1 GBD장애가중(골절>입원)": {"death_injury": 1.0, "death_disease": 1.0,
                              "disability": 0.6, "disease_disability": 0.6,
                              "fracture": 0.07, "hospitalization": 0.05},
    "W2 순서형중대도(골절>입원)": {"death_injury": 5, "death_disease": 5,
                             "disability": 4, "disease_disability": 4,
                             "fracture": 2, "hospitalization": 1},
    "W3 임상acuity(입원>골절)": {"death_injury": 5, "death_disease": 5,
                            "disability": 4, "disease_disability": 4,
                            "hospitalization": 2, "fracture": 1},
}


def _benefit(scale: dict, ev: dict, w: dict) -> float:
    return float(sum(w.get(it, 1.0) * ev.get(it, 0.0) * s for it, s in scale.items()))


def weight_sensitivity(schedule: str = DEFAULT_SCHEDULE,
                       budget_frac: float = 0.8) -> pd.DataFrame:
    """가중치 민감도: 여러 독립 가중 셋에서 그리디 vs LP(해당 가중) 가중혜택 비교.

    ★조작 의심 차단 장치. LP 우위가 '특정 가중치에서만' 나오는지 전부 공개한다.
      실제: W0/W1/W2(골절≥입원)에서는 동률(그리디가 마침 밀도최적), W3(입원>골절)
      에서만 LP 우위 → '복지가중이 무조건 LP를 이기게 한다'는 주장은 하지 않는다.
      (현 표는 floor 위 재량항목이 골절·입원 2개뿐·지급액 거의 동일한 knife-edge.)
    """
    B = budget.expected_claims(schedule).total_claims * budget.DEFAULT_LOADING * budget_frac
    full = budget.expected_claims(schedule)
    ev = dict(zip(full.by_item["coverage_item"], full.by_item["expected_events"]))
    g = budget.optimize_under_budget(schedule, B)        # 우선순위 그리디(가중 무관)
    rows = []
    for name, wd in WEIGHT_SCHEMES.items():
        w = wd or {}
        lp = budget.optimize_benefit_lp(schedule, B, weights=(w or {k: 1.0 for k in ev}))
        if not lp.get("feasible"):
            rows.append({"가중셋": name, "그리디_가중혜택": None,
                         "LP_가중혜택": None, "LP우위": None, "비고": "infeasible"})
            continue
        gb = _benefit(g["coverage_scale"], ev, w)
        lb = _benefit(lp["coverage_scale"], ev, w)
        rows.append({"가중셋": name, "그리디_가중혜택": round(gb, 1),
                     "LP_가중혜택": round(lb, 1), "LP우위": round(lb - gb, 1) + 0.0,
                     "비고": "LP우위" if lb - gb > 1 else "동률(역산 안 함)"})
    df = pd.DataFrame(rows)
    df.attrs["note"] = ("독립 근거 가중 셋들. LP 우위가 W3에서만 → 가중치를 LP가 이기게 "
                        "고른 게 아님을 투명 공개(현 표 knife-edge). 견고한 LP 필요성 "
                        "근거는 perturbation_sensitivity 참고.")
    return df


# --- Option 5: 섭동 민감도 (LP 필요성의 견고한 증거) ------------------------

def perturbation_sensitivity(schedule: str = DEFAULT_SCHEDULE,
                             budget_fracs=(0.7, 0.8, 0.9)) -> pd.DataFrame:
    """그리디 '우선순위 가정' 섭동에 LP가 견고함 — 표·예산·가중 불변, 가정만 흔듦.

    그리디는 손으로 정한 우선순위에 의존한다. 현 표에선 그 순서가 우연히 밀도최적이라
    LP와 동률이지만, **순서 가정 하나만 바꾸면(입원先)** 그리디는 즉시 suboptimal,
    LP는 목적함수만 보므로 불변 → 같은 예산에서 더 많은 (균등가중) 수혜자.
    → "지금은 우연히 같지만, 가정이 흔들리면 LP만 최적을 보장한다"의 수치 증거.
    """
    full = budget.expected_claims(schedule).total_claims * budget.DEFAULT_LOADING
    fullest = budget.expected_claims(schedule)
    ev = dict(zip(fullest.by_item["coverage_item"], fullest.by_item["expected_events"]))
    w1 = {k: 1.0 for k in ev}                 # 균등가중(머릿수) — 섭동 증거는 가중 무관
    # 섭동 우선순위: 고위험 먼저(동일) + 입원을 골절 앞으로(밀도 역행 가정)
    alt_priority = [i for i in budget.DEFAULT_PRIORITY if i in budget.CATASTROPHIC_ITEMS] \
        + ["hospitalization", "fracture"]
    rows = []
    for frac in budget_fracs:
        B = full * frac
        g0 = budget.optimize_under_budget(schedule, B)                 # 기본(우연 밀도최적)
        g1 = budget.optimize_under_budget(schedule, B, priority=alt_priority)  # 섭동
        lp = budget.optimize_benefit_lp(schedule, B, weights=w1)       # 균등 LP
        h0 = _benefit(g0["coverage_scale"], ev, {})
        h1 = _benefit(g1["coverage_scale"], ev, {})
        hl = _benefit(lp["coverage_scale"], ev, {}) if lp.get("feasible") else float("nan")
        rows.append({
            "예산(full대비)": f"{frac:.0%}",
            "그리디_기본_수혜자": round(h0, 1),
            "그리디_섭동_수혜자": round(h1, 1),
            "LP_수혜자(불변)": round(hl, 1),
            "LP_추가수혜_vs섭동": round(hl - h1, 1) + 0.0,
        })
    df = pd.DataFrame(rows)
    df.attrs["note"] = ("표·예산·가중 불변, 그리디 '우선순위 가정'만 입원先으로 섭동. "
                        "LP는 가정 불변이라 최적 유지 → 섭동된 그리디 대비 추가 수혜. "
                        "그리디 최적성이 '운 좋은 가정'에 의존함을 정량 입증.")
    return df

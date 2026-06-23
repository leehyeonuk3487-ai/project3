"""M5 — 계리 보험료 산정 (집단 단위).

budget.py의 단순 정액할증(×1.25)을 계리적으로 고도화한다:

  총보험료(gross) = (순보험료 + 위험할증) / (1 − 사업비율)

  · 순보험료(net/pure) = 1인당 기대청구액 E[S]/N (검증된 발생률 surface×급부).
  · 위험할증(risk margin) = α · SD[S]/N  (표준편차 원리).
      집합청구 S = Σ_i payout_i · N_i, N_i ~ Poisson(μ_i=N·rate_i), 고정심도 가정 →
      E[S]=Σ payout_i·μ_i,  Var[S]=Σ payout_i²·μ_i  (독립 포아송 빈도).
      변동계수 CV=SD/E는 N이 클수록 작아져 대형(시도) 풀은 할증↓, 소형(시군구) 풀은 할증↑
      — 신뢰도(pooling) 효과를 계리적으로 반영.
  · 사업비율(expense) = 모집·관리비 가정(데이터 출처 없음 → 파라미터·문서화).

비교: 보고된 실제 보험료(benefits.REPORTED_PREMIUM)·단순 ×1.25와 대조.
정직성: 집단 가격용. 사업비율·안전계수는 가정이며 민감도로 노출. 개인 차등 아님.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import benefits

DEFAULT_ALPHA = 1.0          # 안전계수(표준편차 배수) — 1.0 ≈ 단측 84% 적정성(정규근사)
DEFAULT_EXPENSE_RATIO = 0.20  # 사업비율 가정(모집·관리·이윤) — 데이터 없음, 파라미터화


@dataclass
class PremiumResult:
    schedule: str
    sido: str
    population: int
    net_pc: float            # 순보험료(1인당)
    risk_margin_pc: float    # 위험할증(1인당)
    expense_ratio: float
    gross_pc: float          # 총보험료(1인당)
    cv: float                # 집합청구 변동계수 SD/E
    alpha: float
    implied_loading: float   # gross/net (정액할증 환산)
    reported_pc: float | None
    by_item: pd.DataFrame
    note: str


def actuarial_premium(schedule_name: str, year: int = 2024,
                      alpha: float = DEFAULT_ALPHA,
                      expense_ratio: float = DEFAULT_EXPENSE_RATIO,
                      population_override: int | None = None,
                      coverage_scale: dict[str, float] | None = None) -> PremiumResult:
    """계리 보험료(순+위험할증+사업비). budget.expected_claims를 재사용."""
    from . import budget
    est = budget.expected_claims(schedule_name, population_override, year,
                                 coverage_scale=coverage_scale, loading=1.0)
    N = est.population
    bi = est.by_item.copy()
    # 집합청구 평균·분산 (복합포아송, 고정심도)
    ES = float((bi["payout_per_event"] * bi["expected_events"]).sum())
    VarS = float(((bi["payout_per_event"] ** 2) * bi["expected_events"]).sum())
    SD = float(np.sqrt(VarS))
    cv = SD / ES if ES > 0 else float("nan")

    net_pc = ES / N if N else float("nan")
    risk_margin_pc = alpha * SD / N if N else float("nan")
    gross_pc = (net_pc + risk_margin_pc) / (1.0 - expense_ratio)
    implied = gross_pc / net_pc if net_pc else float("nan")

    bi["claims_share%"] = (bi["expected_claims"] / ES * 100).round(1)
    bi["var_share%"] = ((bi["payout_per_event"] ** 2 * bi["expected_events"]) / VarS * 100).round(1)

    return PremiumResult(
        schedule=schedule_name, sido=est.sido, population=N,
        net_pc=round(net_pc, 1), risk_margin_pc=round(risk_margin_pc, 1),
        expense_ratio=expense_ratio, gross_pc=round(gross_pc, 1),
        cv=round(cv, 4), alpha=alpha, implied_loading=round(implied, 3),
        reported_pc=benefits.REPORTED_PREMIUM.get(schedule_name),
        by_item=bi[["label", "track", "expected_claims", "claims_share%", "var_share%"]],
        note=(f"순 {net_pc:,.0f} + 위험할증(α={alpha}) {risk_margin_pc:,.0f} → 사업비 "
              f"{expense_ratio:.0%} 반영 총 {gross_pc:,.0f}원/인. CV={cv:.3f} "
              f"(N={N:,} 풀링). 사업비율·α는 가정(민감도)."),
    )


def loading_sensitivity(schedule_name: str, year: int = 2024,
                        alphas=(0.0, 1.0, 1.645, 2.0),
                        expense_ratios=(0.15, 0.20, 0.30)) -> pd.DataFrame:
    """안전계수 α × 사업비율 민감도표(1인당 총보험료)."""
    rows = []
    for a in alphas:
        rec = {"alpha": a}
        for e in expense_ratios:
            pr = actuarial_premium(schedule_name, year, alpha=a, expense_ratio=e)
            rec[f"사업비{int(e*100)}%"] = round(pr.gross_pc, 0)
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_to_reported(schedule_name: str = "경기도", year: int = 2024) -> dict:
    """계리 총보험료 vs 보고된 실제 보험료 vs 단순 ×1.25 대조."""
    pr = actuarial_premium(schedule_name, year)
    flat = pr.net_pc * 1.25
    return {
        "schedule": schedule_name, "sido": pr.sido, "population": pr.population,
        "net_pure": pr.net_pc, "actuarial_gross": pr.gross_pc,
        "flat_1.25": round(flat, 1), "reported": pr.reported_pc,
        "actuarial_vs_reported%": (round((pr.gross_pc / pr.reported_pc - 1) * 100, 1)
                                   if pr.reported_pc else None),
        "cv": pr.cv, "implied_loading": pr.implied_loading,
        "note": ("계리 총보험료가 보고치보다 낮으면 보고 보험료에 사업비·이윤·미반영 보장이 "
                 "더 실려있음을 시사(우리 추정은 핵심항목 하한)."),
    }

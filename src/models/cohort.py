"""M4 — 코호트 리스크 통합·보정 (집단 단위, 개인 예측 아님).

M1 코호트 프로파일(병무청 BMI + CHS 20대남 행태)과 LORO로 검증된 생태회귀 상대위험
(IRR)을 결합해 **지자체별 코호트 리스크 지수**를 산출하고, 발생률 surface·급부와 통합한
**코호트 기대청구액**을 만든다(M5 보험료 산정의 입력).

핵심:
  · cohort_risk_index : 시도 20대남 행태유병률을 자체 지역분산으로 표준화 후 생태회귀
    β(SD당 log-IRR)를 적용 → exp(가중합). 전국 인구가중 평균=1.0로 정규화. 요인기여 병기.
    → "지역 코호트가 평균 대비 몇 % 높은 행태리스크인가"를 집단 단위로 해석.
  · 현역 BMI 선택 : 병무청 BMI≥25(과체중25–29.9 + 비만≥30) vs CHS 일반 20대남 obese(≥25).
    정의 호환 확인됨. 현역이 건강선택될 수 있으나 군 훈련손상이 이를 상쇄할 수 있어
    손상률에 자동 적용하지 않고 '특성·민감도'로만 보고(정직성).
  · cohort_adjusted_claims : 검증된 rate surface×급부(budget 재사용) 기대청구액에 코호트
    지수를 이중계산 없이 병기. apply_index=True면 지수 스케일을 민감도로 적용.

정직성: 생태회귀 계수는 지역 단위 연관(ecological fallacy 주의). 모두 집단 단위 해석이며
개인 장병 스코어링/차등에 쓰지 않는다. 시드/결정적 집계.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import aggregate, loaders
from . import ecological, population


def _young_male_prevalence(predictors: list[str]) -> pd.DataFrame:
    """시도별 20대 남성 행태 위험요인 가중 유병률."""
    chs = aggregate.add_risk_indicators(loaders.load_chs())
    m20 = chs[(chs["sex_name"] == "남자") & chs["age"].between(20, 29)]
    rows = []
    for sido, g in m20.groupby("sido"):
        rec = {"sido": sido, "n": int(len(g))}
        for f in predictors:
            rec[f] = aggregate._weighted_mean(g[f], g["wt_p"])
        rows.append(rec)
    return pd.DataFrame(rows).set_index("sido").sort_index()


def _conscript_weights() -> pd.Series:
    """시도별 현역 stock(정규화 가중)."""
    sidos = sorted(loaders.load_chs()["sido"].unique())
    w = {s: population.conscript_stock(s, 2024) for s in sidos}
    return pd.Series(w)


def cohort_risk_index(predictors: list[str] | None = None,
                      calibrate: bool = False) -> pd.DataFrame:
    """시도별 20대남 코호트 행태 상대위험 지수(전국 인구가중 평균=1.0) + 요인기여.

    생태회귀 β(SD당)를 20대남 유병률의 '자체 지역 표준화' 값에 적용한다(검증된 효과크기를
    코호트의 지역 분산에 적용 — 절대 IRR을 개인위험으로 주장하지 않음).
    """
    efit = ecological.fit(predictors, calibrate=calibrate)
    preds = efit.predictors
    prof = _young_male_prevalence(preds)
    z = (prof[preds] - prof[preds].mean()) / prof[preds].std(ddof=0).replace(0, 1.0)
    beta = efit.coef[preds]
    log_rr = z.mul(beta, axis=1)               # 요인별 log 기여
    rr = np.exp(log_rr.sum(axis=1))
    w = _conscript_weights().reindex(rr.index).fillna(0.0)
    rr_index = rr / np.average(rr, weights=w)  # 전국 인구가중 평균=1.0

    out = pd.DataFrame({"risk_index": rr_index.round(4)})
    for f in preds:                            # 요인별 승법 기여(=exp(z·β))
        out[f"기여_{f}"] = np.exp(log_rr[f]).round(4)
        out[f"유병_{f}"] = prof[f].round(4)
    return out.sort_values("risk_index", ascending=False)


def conscript_bmi_selection(predictors: list[str] | None = None) -> dict:
    """현역(병무청 BMI≥25) vs 일반 20대남(CHS obese≥25) 비교 + 1차 보정계수(민감도).

    병무청 비만 기준 BMI≥25 = 과체중(25–29.9)+비만(≥30). CHS obese(≥25)와 정의 일치.
    선택보정계수 = exp(β_obese × (현역−일반)/SD_region). 군 훈련손상이 건강선택을 상쇄할
    수 있어 손상률에 자동 적용하지 않고 민감도로만 제시.
    """
    bmi = population.bmi_distribution()
    conscript_obese = float(bmi.get("과체중", 0.0) + bmi.get("비만", 0.0))
    chs = aggregate.add_risk_indicators(loaders.load_chs())
    m20 = chs[(chs["sex_name"] == "남자") & chs["age"].between(20, 29)]
    general_obese = float(aggregate._weighted_mean(m20["obese"], m20["wt_p"]))

    efit = ecological.fit(predictors)
    factor = float("nan")
    if "obese" in efit.predictors:
        dz = (conscript_obese - general_obese) / float(efit.stds["obese"])
        factor = float(np.exp(efit.coef["obese"] * dz))
    return {
        "conscript_bmi_ge25": round(conscript_obese, 4),
        "general_bmi_ge25": round(general_obese, 4),
        "obese_irr_per_sd": round(float(efit.irr.get("obese", float("nan"))), 3),
        "selection_factor_sensitivity": round(factor, 4),
        "note": ("현역이 일반보다 BMI≥25 비율 "
                 + ("낮음(건강선택)" if conscript_obese < general_obese else "높음")
                 + " — 단, 군 훈련손상이 건강선택을 상쇄할 수 있어 손상률에 자동 적용 안 함(민감도). "
                 "생태회귀 β는 지역단위 연관(ecological fallacy 주의)."),
    }


def cohort_adjusted_claims(schedule_name: str, year: int = 2024,
                           apply_index: bool = False,
                           population_override: int | None = None) -> dict:
    """검증된 rate surface×급부 기대청구액(budget 재사용) + 코호트 리스크 지수 병기.

    apply_index=False(기본): 이중계산 방지 — surface 발생률이 이미 지역차를 반영하므로
      지수는 해석용으로 병기만 한다.
    apply_index=True: 코호트 지수를 청구액에 곱하는 민감도 시나리오(지역 행태 재가중).
    """
    from ..data import benefits
    from ..optimize import budget
    sido = benefits.SCHEDULE_SIDO[schedule_name]
    est = budget.expected_claims(schedule_name, population_override, year)
    idx_tbl = cohort_risk_index()
    risk_index = float(idx_tbl["risk_index"].get(sido, 1.0))

    per_capita = est.per_capita_claims
    scenario = per_capita * (risk_index if apply_index else 1.0)
    return {
        "schedule": schedule_name, "sido": sido, "year": year,
        "population": est.population, "population_source": est.population_source,
        "expected_claims_total": round(est.total_claims, 0),
        "per_capita_claims": round(per_capita, 1),
        "cohort_risk_index": round(risk_index, 4),
        "per_capita_index_scenario": round(scenario, 1),
        "apply_index": apply_index,
        "by_item": est.by_item,
        "note": "코호트 지수는 해석/민감도용 — 기본 청구액은 검증 surface 기준(이중계산 없음).",
    }


def summary(predictors: list[str] | None = None) -> dict:
    """M4 종합: 코호트 리스크 지수표 + 현역 BMI 선택 + 생태회귀 IRR 출처."""
    efit = ecological.fit(predictors)
    return {
        "risk_index_table": cohort_risk_index(predictors),
        "bmi_selection": conscript_bmi_selection(predictors),
        "irr_source": ecological.summary_table(efit),
        "predictors": efit.predictors,
        "n_regions": efit.n_obs,
    }

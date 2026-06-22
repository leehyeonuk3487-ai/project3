"""KNHANES 측정값으로 CHS 자가보고 보정.

CHS BMI는 자가보고 키·몸무게, 흡연은 자가응답이다. KNHANES는 동일 항목을
**검진 측정값(BMI)** 으로 보유하므로, 전국 가중 유병률을 비교해 보정계수
(factor = KNHANES/CHS)를 구하고 CHS 지역 유병률에 곱한다.

비교는 연령대를 맞추기 위해 19~39세(CHS 표본 범위)로 한정한다. 보정계수는
[0.5, 2.0]로 클리핑해 과보정을 막는다. 생태회귀·층화에서 선택적으로 적용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import aggregate, loaders

CALIB_FACTORS = ("smoker", "obese", "underweight")
_AGE_LO, _AGE_HI = 19, 39


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna()
    return float(np.average(values[m], weights=weights[m])) if m.any() else float("nan")


def _chs_prevalence() -> dict:
    df = aggregate.add_risk_indicators(loaders.load_chs())
    df = df[df["age"].between(_AGE_LO, _AGE_HI)]
    return {f: _wmean(df[f], df["wt_p"]) for f in CALIB_FACTORS}


def _knhanes_prevalence() -> dict:
    kn = loaders.load_knhanes()
    kn = kn[kn["age"].between(_AGE_LO, _AGE_HI)].copy()
    smk = kn["BS3_1"]
    kn["smoker"] = np.where(smk.isin([1, 2]), 1.0,
                            np.where(smk.isin([3, 8]), 0.0, np.nan))
    kn["obese"] = np.where(kn["bmi"].notna(), (kn["bmi"] >= 25).astype(float), np.nan)
    kn["underweight"] = np.where(kn["bmi"].notna(), (kn["bmi"] < 18.5).astype(float), np.nan)
    w = kn["wt_itvex"] if "wt_itvex" in kn else pd.Series(1.0, index=kn.index)
    return {f: _wmean(kn[f], w) for f in CALIB_FACTORS}


def calibration_factors() -> pd.DataFrame:
    """요인별 CHS·KNHANES 유병률과 보정계수(클리핑)."""
    chs, kn = _chs_prevalence(), _knhanes_prevalence()
    rows = []
    for f in CALIB_FACTORS:
        c, k = chs[f], kn[f]
        factor = np.clip(k / c, 0.5, 2.0) if c and np.isfinite(c) and c > 0 else 1.0
        rows.append({"factor": f, "chs": round(c, 4), "knhanes": round(k, 4),
                     "calibration": round(float(factor), 3)})
    return pd.DataFrame(rows).set_index("factor")


def apply_calibration(region_prev: pd.DataFrame,
                      factors: pd.DataFrame | None = None) -> pd.DataFrame:
    """시도 유병률 DataFrame에 보정계수를 곱한다(유병률은 [0,1] 클리핑)."""
    factors = factors if factors is not None else calibration_factors()
    out = region_prev.copy()
    for f in CALIB_FACTORS:
        if f in out.columns:
            out[f] = (out[f] * factors.loc[f, "calibration"]).clip(0, 1)
    return out

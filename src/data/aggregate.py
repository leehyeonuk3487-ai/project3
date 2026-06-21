"""CHS 개인 데이터 → 집계(지역/연령/성별) 위험요인 유병률.

생태학적 회귀와 층화에 쓰기 위해 개인 행태·신체 위험요인을 이진/연속
지표로 정의하고, 개인가중치(wt_p)로 가중 집계한다.

CHS 값코드(확인 결과):
  sma_03z2 현재흡연: 1=매일,2=가끔,3=과거,8=비해당(평생비흡연),9=모름
  drb_04z1 월간폭음(남): 1=전혀~5=거의매일, 8=비해당(여/비음주), 9=모름
  pha_04z1 격렬신체활동 일수: 0~7, 77/99=결측
  hya_04z1 고혈압진단 / dia_04z1 당뇨진단: 1=예,2=아니오,9=모름
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import loaders

# 위험요인 컬럼명(집계·회귀 공통)
RISK_FACTORS = ["smoker", "binge_drink", "phys_inactive", "obese", "underweight"]


def add_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """CHS 개인 DataFrame에 이진 위험요인 지표를 추가한다(결측은 NaN)."""
    out = df.copy()

    smk = out["sma_03z2"]
    out["smoker"] = np.where(smk.isin([1, 2]), 1.0,
                             np.where(smk.isin([3, 8]), 0.0, np.nan))

    binge = out["drb_04z1"]
    out["binge_drink"] = np.where(binge.isin([4, 5]), 1.0,
                                  np.where(binge.isin([1, 2, 3, 8]), 0.0, np.nan))

    vig = out["pha_04z1"].where(~out["pha_04z1"].isin([77, 99]))
    out["phys_inactive"] = np.where(vig == 0, 1.0,
                                    np.where(vig > 0, 0.0, np.nan))

    out["obese"] = np.where(out["bmi"].notna(), (out["bmi"] >= 25).astype(float), np.nan)
    out["underweight"] = np.where(out["bmi"].notna(), (out["bmi"] < 18.5).astype(float), np.nan)
    return out


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna()
    if m.sum() == 0:
        return float("nan")
    return float(np.average(values[m], weights=weights[m]))


def region_risk_prevalence(
    chs: pd.DataFrame | None = None,
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """시도별 위험요인 가중 유병률.

    Returns DataFrame indexed by sido with one column per factor (0~1).
    """
    if chs is None:
        chs = loaders.load_chs()
    chs = add_risk_indicators(chs)
    factors = factors or RISK_FACTORS

    rows = []
    for sido, g in chs.groupby("sido"):
        rec = {"sido": sido, "n": len(g)}
        for f in factors:
            rec[f] = _weighted_mean(g[f], g["wt_p"])
        rows.append(rec)
    return pd.DataFrame(rows).set_index("sido").sort_index()


def cell_risk_prevalence(
    chs: pd.DataFrame | None = None,
    by: tuple[str, ...] = ("sido", "sex_name", "age_band"),
    factors: list[str] | None = None,
) -> pd.DataFrame:
    """지역×성×연령 셀별 위험요인 유병률 (층화용)."""
    if chs is None:
        chs = loaders.load_chs()
    chs = add_risk_indicators(chs)
    factors = factors or RISK_FACTORS

    rows = []
    for keys, g in chs.groupby(list(by)):
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rec["n"] = len(g)
        for f in factors:
            rec[f] = _weighted_mean(g[f], g["wt_p"])
        rows.append(rec)
    return pd.DataFrame(rows)

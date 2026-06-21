"""집단 리스크 층화 + IRR(발생률비).

개인 예측이 아니라 **집단 층화**다. 각 개인에게 그가 속한
(시도×성별×연령) 셀의 KDCA 기반 기대 발생률을 부여하고, 개인가중치로
가중해 저/중/고 3분위로 나눈다. IRR = 고위험군 발생률 / 저위험군 발생률.

개인 단위 부상 라벨이 없으므로 층화축은 실측 발생률 surface(실데이터)이며,
개인 행태 위험요인(흡연·비만 등)은 각 층의 '프로파일'로 함께 보고해
어떤 집단이 더 높은 위험을 지는지 해석을 제공한다. 이 프로파일이
생태회귀(ecological.py)의 위험요인 방향성과 연결된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..data import aggregate, loaders
from . import rates

# 층화의 1차 산출 발생률(임상적으로 명확한 중증외상). 보조로 골절/입원도 부착.
PRIMARY_ITEM = "severe_trauma"


def _cell_rate_lookup(item: str) -> dict:
    """(sido, sex, age_band) → rate_per_1000py 조회 딕셔너리."""
    surf = rates.coverage_rate_surface(item)
    age_col = "discharge_age_band" if "discharge_age_band" in surf else "age_band"
    lut = {}
    for _, r in surf.iterrows():
        lut[(r["sido"], r["sex"], r[age_col])] = r["rate_per_1000py"]
    return lut, age_col


def _discharge_band(age_band: str) -> str:
    """KDCA 10세 구간 → 퇴원표 구간 매핑(근사)."""
    mapping = {
        "0-9세": "0-14세", "10-19세": "15-24세", "20-29세": "25-34세",
        "30-39세": "35-44세", "40-49세": "45-54세", "50-59세": "55-64세",
        "60-69세": "65-74세", "70-79세": "75세이상", "80세 이상": "75세이상",
    }
    return mapping.get(age_band, age_band)


def assign_expected_rates(chs: pd.DataFrame | None = None) -> pd.DataFrame:
    """각 개인에게 (시도×성×연령) 셀 기반 기대 발생률을 부여한다."""
    if chs is None:
        chs = loaders.load_chs()
    chs = aggregate.add_risk_indicators(chs)

    primary_lut, _ = _cell_rate_lookup(PRIMARY_ITEM)
    disch_lut, _ = _cell_rate_lookup("fracture")

    def lookup(row, lut, discharge=False):
        band = _discharge_band(row["age_band"]) if discharge else row["age_band"]
        return lut.get((row["sido"], row["sex_name"], band), np.nan)

    out = chs.copy()
    out["rate_severe_trauma"] = out.apply(lambda r: lookup(r, primary_lut), axis=1)
    out["rate_fracture"] = out.apply(lambda r: lookup(r, disch_lut, discharge=True), axis=1)
    return out


def _weighted_quantile_edges(values: pd.Series, weights: pd.Series, n: int) -> list:
    """가중 분위 경계 n+1개 (양끝은 ±inf로 확장해 경계값 포함 보장)."""
    order = np.argsort(values.values)
    v = values.values[order]
    w = weights.values[order]
    cum = np.cumsum(w) / w.sum()
    qs = np.linspace(0, 1, n + 1)[1:-1]
    inner = list(np.interp(qs, cum, v))
    return [-np.inf] + inner + [np.inf]


def stratify(
    chs: pd.DataFrame | None = None,
    rate_col: str = "rate_severe_trauma",
    n_strata: int = 3,
) -> dict:
    """가중 분위로 저/중/고 층화하고 IRR과 층별 프로파일을 계산한다."""
    df = assign_expected_rates(chs)
    df = df[df[rate_col].notna() & df["wt_p"].notna()].copy()

    labels = {3: ["저위험", "중위험", "고위험"]}.get(n_strata,
              [f"Q{i+1}" for i in range(n_strata)])
    df["stratum"] = pd.cut(
        df[rate_col],
        bins=_weighted_quantile_edges(df[rate_col], df["wt_p"], n_strata),
        labels=labels,
        include_lowest=True,
    )

    rows = []
    for lab in labels:
        g = df[df["stratum"] == lab]
        w = g["wt_p"]
        rec = {
            "stratum": lab,
            "n": len(g),
            "pop_share": float(w.sum() / df["wt_p"].sum()),
            "rate_severe_trauma": float(np.average(g["rate_severe_trauma"], weights=w)),
            "rate_fracture": float(np.average(
                g["rate_fracture"].fillna(g["rate_fracture"].mean()), weights=w)),
        }
        for f in aggregate.RISK_FACTORS:
            vals = g[f]
            m = vals.notna()
            rec[f] = float(np.average(vals[m], weights=w[m])) if m.any() else np.nan
        rows.append(rec)
    profile = pd.DataFrame(rows).set_index("stratum")

    lo = profile.loc[labels[0], "rate_severe_trauma"]
    hi = profile.loc[labels[-1], "rate_severe_trauma"]
    irr = float(hi / lo) if lo > 0 else float("nan")

    return {
        "profile": profile,
        "irr": irr,
        "rate_col": rate_col,
        "note": config.HONESTY_NOTE,
    }

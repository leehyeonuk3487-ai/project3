"""미래 투영 — 외인사망 추세 외삽 (parametric 로그선형).

★왜 GBM이 아니라 로그선형인가:
  모듈A의 GBM은 트리 기반이라 **훈련 연도범위(2005–2024) 밖을 외삽하지 못한다**
  (경계 leaf 값으로 클램프 → 2025+ 예측이 평탄선). GBM은 공간 캘리브레이션
  (외인 0.76 채택)·검증 역할로 유지하고, '미래 추세 투영'은 관측 시계열의
  로그선형 외삽으로 분리한다.

★점추정 금지 — 시나리오 밴드로 제시:
  외인사망률 전국 가중평균은 2005→2024 누적 −70% 감소했으나 **최근 둔화**했다.
    전기간(2005–2024)  : 연 −6.9%/년
    최근7년(2018–2024) : 연 −3.3%/년   ← 둔화
    최근(2020–2024)    : 거의 평탄
  추세는 꺾일 수 있으므로 점추정이 아니라 밴드로 제시한다:
    낙관(claims↓·예산여유) : 장기추세 재개   −6.9%/년
    기준(가장 그럴듯)      : 최근 둔화추세   −3.3%/년
    보수(claims↑·예산부담) : 감소 멈춤        0%/년(2024 수준 유지)

★한계: 과거 추세 외삽이며 구조변화(정책·인구구성·사회환경)를 반영하지 못한다.
  pricing(현재 보험료·예산)은 M0 직접관측 백본 불변 — 외삽은 미래 투영에만 쓴다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cell_panel import build_panel

EXTRAPOLATION_NOTE = (
    "과거 추세 로그선형 외삽 — 구조변화(정책·인구·사회환경) 미반영. "
    "점추정이 아닌 시나리오 밴드로 해석. 현재 pricing은 M0 직접관측 백본 불변."
)

# 시나리오 → 추세 산정 구간(데이터에서 동적으로 기울기 계산)
SCENARIO_WINDOWS = {
    "optimistic": (2005, 2024),    # 장기 추세 재개 (가장 빠른 감소)
    "base": (2018, 2024),          # 최근 둔화 추세
    "conservative": None,          # 감소 멈춤(평탄, 0%/년)
}


def _national_external_rate() -> dict[int, float]:
    """전국 외인사망률(20대 남성, 인구가중, per 100k) 연도별 시계열."""
    p = build_panel("external")
    yrs = sorted(p.year.unique())
    return {int(y): float(p[p.year == y].deaths.sum()
                          / p[p.year == y].py.sum() * 1e5) for y in yrs}


def external_trend_slopes() -> dict[str, float]:
    """관측 시계열에서 시나리오별 연 감소율(로그선형 기울기)을 산출."""
    rate = _national_external_rate()
    yrs = sorted(rate)
    out: dict[str, float] = {}
    for name, win in SCENARIO_WINDOWS.items():
        if win is None:               # 보수: 평탄
            out[name] = 0.0
            continue
        a, b = win
        ys = [y for y in yrs if a <= y <= b]
        slope = np.polyfit(ys, np.log([rate[y] for y in ys]), 1)[0]
        out[name] = float(np.exp(slope) - 1.0)   # 연 변화율
    return out


def external_trend_band(
    base_year: int = 2024,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """외인사망 추세 외삽 밴드(상대 배수, base_year=1.0).

    반환: year × {optimistic, base, conservative} 상대 배수 + 절대 투영률.
    배수는 '미래 외인 청구 변화'를 곱하는 데 쓴다(현재 pricing은 불변).
    """
    years = years or list(range(base_year, base_year + 7))
    slopes = external_trend_slopes()
    rate = _national_external_rate()
    base_rate = rate[base_year]
    rows = []
    for y in years:
        row = {"year": y}
        for name, s in slopes.items():
            mult = (1.0 + s) ** (y - base_year)
            row[f"{name}_mult"] = round(mult, 4)
            row[f"{name}_rate"] = round(base_rate * mult, 3)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["slopes_pct_per_yr"] = {k: round(v * 100, 2)
                                     for k, v in slopes.items()}
    df.attrs["base_rate_per100k"] = round(base_rate, 3)
    df.attrs["note"] = EXTRAPOLATION_NOTE
    return df


# --- 인구절벽 시나리오 -----------------------------------------------------

def population_cliff(
    sido: str = "전국",
    base_year: int = 2024,
    years: list[int] | None = None,
) -> pd.DataFrame:
    """현역 청년 모집단 추계(인구절벽). base_year=1.0 상대배수 포함.

    통계청 장래인구추계 기반(population.conscript_stock). 전국이면 20대 남성
    인구를 직접 합산해 인구절벽 규모를 보인다.
    """
    from . import population
    from ..data import loaders
    years = years or list(range(base_year, base_year + 7))

    if sido == "전국":
        pop = loaders.load_population_projection()
        m = pop[(pop.sex_name == "남자")
                & pop.age5.isin(["20-24세", "25-29세"])
                & (pop.sido != "전국")]
        nat = m.groupby("year")["value"].sum()
        base = float(nat[base_year])
        rows = [{"year": y, "stock": float(nat[y]),
                 "mult": round(float(nat[y]) / base, 4)} for y in years]
    else:
        base = population.conscript_stock(sido, base_year)
        rows = [{"year": y, "stock": population.conscript_stock(sido, y),
                 "mult": round(population.conscript_stock(sido, y) / base, 4)}
                for y in years]
    return pd.DataFrame(rows)

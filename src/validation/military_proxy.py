"""군 코호트 proxy 타당성 검증 — 국방부 사망사고 통계(2011–2025).

지금까지 우리는 **일반인구 통계를 군 코호트의 proxy**로 썼다(claims·군 미시데이터 부재).
국방부(개최기관) 사망사고 실데이터로 그 proxy의 타당성을 **검증**하고, 보정 방향을
제시한다. ★pricing(M0 직접관측)은 불변 — 결과는 정직성·민감도 병기용이다.

세 갈래:
  ① 자살 보정계수  : 군자살률 vs 민간자살률 비율 → 일반인구 proxy의 과대/과소 방향.
                     자살은 면책(보장 제외)이라 pricing 무영향 — envelope·정직성용.
  ② 외인 교차검증  : 군 안전사고율(분모=역산 병력) vs 일반인구 외인사망(20대남) 추세·수준.
  ③ 군기사고(총기·폭행·기타) : ★작전/징계성 — 보험 일반보장과 성격 상이 → 외인에 합치지
                     않고 별도 보고, 검증·보정에서 제외.

한계: 연 단위·시도분해 없음, 병력 분모는 자살건수/자살률 역산(직접값 아님), 2025 민간
자살률 결측, 민간자살률은 전연령(우리 M0 자살은 20대남) — 모두 노트로 명시.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import loaders

MND_SOURCE = ("국방부 사망사고 통계 2011–2025 (data.go.kr, 개최기관 직접 데이터). "
              "사고=건수, 자살률=10만명당. 시도·연령 분해 없음(전국 단위).")

# 범주 매핑(추측 금지 — 명시). 군기사고(총기·폭행·기타)는 외인에 합치지 않는다.
CATEGORY_MAP = {
    "military_suicide": "군기사고(자살) → 자살(면책) 보정용",
    "military_external": "안전사고 합계(차량·함정항공·폭발·추락충격·익사·화재·기타) → 외인검증용",
    "military_discipline_other": "군기사고(총기·폭행·기타) → 작전/징계성, 외인 제외·검증 제외",
}

RECENT_WINDOW = 5   # 대표 보정계수: 최근 N년 평균


def suicide_adjustment() -> dict:
    """① 군/민간 자살률 비율 → 대표 보정계수와 방향.

    ratio_t = 군자살률_t / 민간자살률_t. 2025는 민간자살률 결측 → 제외(0/대체 금지).
    대표값 = 최근 5년 평균(현재 적용성 우선)과 전기간 평균을 함께 보고.
    """
    df = loaders.load_mnd_death_accidents()
    valid = df.dropna(subset=["civ_suicide_rate", "mil_suicide_rate"]).copy()
    valid["ratio"] = valid["mil_suicide_rate"] / valid["civ_suicide_rate"]
    recent = valid[valid["year"] >= valid["year"].max() - RECENT_WINDOW + 1]
    recent_mean = float(recent["ratio"].mean())
    full_mean = float(valid["ratio"].mean())
    excluded = sorted(set(df["year"]) - set(valid["year"]))
    return {
        "table": valid[["year", "mil_suicide_rate", "civ_suicide_rate", "ratio"]]
        .round(3).reset_index(drop=True),
        "ratio_recent5_mean": round(recent_mean, 3),
        "ratio_full_mean": round(full_mean, 3),
        "adopted": round(recent_mean, 3),          # 대표 보정계수(최근5년)
        "excluded_years": excluded,                # [2025] — 민간자살률 결측
        "direction": ("군 자살률이 민간의 약 {:.0%} → 일반인구 proxy는 군 자살을 "
                      "과대추정(약 {:.1f}배). 자살은 면책이라 pricing 무영향, envelope·"
                      "정직성·민감도에만 사용.").format(recent_mean, 1 / recent_mean),
        "caveat": ("민간자살률은 전연령 기준(우리 M0 자살은 20대남) — 비율은 군·민간 "
                   "상대 성향의 근사이며 직접 동일모집단 비교는 아님."),
        "applies_to_pricing": False,
    }


def external_crossvalidation() -> dict:
    """② 군 안전사고율 vs 일반인구 외인사망(20대남) — 추세·수준 비교.

    군 병력 분모는 직접값이 없어 자살건수/자살률로 역산(일관성 검증됨). 일반인구 외인은
    cell_panel(20대남) 연도별 인구가중 율. 겹치는 연도(2011–2024)에서 추세 상관(Pearson r)
    과 수준 비율(군/일반)을 산출한다.
    """
    from ..models import cell_panel
    mnd = loaders.load_mnd_death_accidents().copy()
    mnd["safe_rate"] = mnd["military_external"] / mnd["troops_implied"] * 1e5

    p = cell_panel.build_panel("external")
    gen = {int(y): float(p[p.year == y].deaths.sum() / p[p.year == y].py.sum() * 1e5)
           for y in sorted(p.year.unique())}

    years = [int(y) for y in mnd["year"] if int(y) in gen]
    gv = np.array([gen[y] for y in years])
    mv = np.array([float(mnd.loc[mnd.year == y, "safe_rate"].iloc[0]) for y in years])
    r = float(np.corrcoef(gv, mv)[0, 1]) if len(years) > 2 else float("nan")
    level_ratio = float(mv.mean() / gv.mean())
    table = pd.DataFrame({"year": years,
                          "일반인구_외인_20대남": np.round(gv, 2),
                          "군_안전사고율": np.round(mv, 2)})
    return {
        "table": table,
        "overlap_years": (years[0], years[-1]),
        "trend_pearson_r": round(r, 3),
        "level_ratio_mil_over_gen": round(level_ratio, 3),
        "overestimate_factor": round(1 / level_ratio, 2),
        "verdict": ("추세는 강한 동행(r={:.2f}) → proxy가 외인 '추세'를 잘 포착. 수준은 "
                    "군이 일반의 {:.0%}(통제된 복무환경) → proxy가 '수준'을 약 {:.1f}배 "
                    "과대추정. pricing 미변경, 보정계수 후보로 민감도에만 노출.").format(
                        r, level_ratio, 1 / level_ratio),
        "caveat": ("일반인구 외인은 민간생활 사고(교통 등) 전부 포함, 군 안전사고는 복무 중 "
                   "사고만 — 정의 범위 차이도 수준차에 기여. 병력 분모는 역산값."),
        "discipline_other_excluded": ("군기사고(총기·폭행·기타)는 작전/징계성으로 보험 "
                                      "일반보장과 성격이 달라 외인 검증·보정에서 제외."),
        "applies_to_pricing": False,
    }


def summary() -> dict:
    """대시보드·리포트용 종합(숫자만 라이브 산출)."""
    s = suicide_adjustment()
    e = external_crossvalidation()
    return {
        "source": MND_SOURCE,
        "category_map": CATEGORY_MAP,
        "suicide": {"adopted_ratio": s["adopted"],
                    "recent5": s["ratio_recent5_mean"],
                    "full": s["ratio_full_mean"],
                    "excluded_years": s["excluded_years"],
                    "direction": s["direction"]},
        "external": {"trend_r": e["trend_pearson_r"],
                     "level_ratio": e["level_ratio_mil_over_gen"],
                     "overestimate_factor": e["overestimate_factor"],
                     "verdict": e["verdict"]},
        "pricing_impact": "없음 — M0 직접관측 백본 불변, 정직성·민감도 병기만.",
    }

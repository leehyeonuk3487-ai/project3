"""보장항목별 기준 발생률 추정 (slice 1).

KDCA 발생률표는 성·연령·시도가 **각각 주변분포(marginal)**로만 제공된다
(성×연령×시도 joint cell이 없음). 따라서 계리에서 표준적으로 쓰는
**비례 보정(proportional / marginal independence)** 으로 결합한다:

    rate(sex, age, sido)
        = national_rate × R_sex × R_age × R_sido

    R_x = rate(level x) / national_rate   (주변 상대위험)

이 방법은 성·연령·지역 효과가 곱셈적으로 독립이라는 가정을 둔다.
실제로는 상호작용이 있으므로 추정치는 근사이며, slice 3에서 KDCA 관측
주변분포 대비 캘리브레이션으로 검증한다.

퇴원손상(골절/입원)표는 성×연령이 **joint 교차표**라 그 부분은 보정 없이
직접 쓰고, 시도 차원만 중증외상 시도 상대위험을 proxy로 적용한다.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..data import loaders

# KDCA 발생률 지표 라벨(부분일치)
_RATE_METRIC = "발생률"
_DISCHARGE_RATE_METRIC = "퇴원율"
_SHARE_METRIC = "분율"


# ---------------------------------------------------------------------------
# 주변분포 추출 헬퍼
# ---------------------------------------------------------------------------

def _latest_year(df: pd.DataFrame) -> int:
    return int(df["year"].max())


def _level_map(df: pd.DataFrame, dim: str, metric_contains: str, year: int) -> dict:
    """특정 차원·지표·연도의 {수준: 값} 맵."""
    sub = df[
        (df["dim"] == dim)
        & (df["metric"].str.contains(metric_contains, na=False))
        & (df["year"] == year)
    ]
    return dict(zip(sub["level"], sub["value"]))


def _national_rate(df: pd.DataFrame, metric_contains: str, year: int) -> float:
    """전체/소계(전국) 기준 발생률."""
    nat = _level_map(df, "전체", metric_contains, year)
    # '소계' 또는 '전체' 키로 들어온다.
    for key in ("소계", "전체"):
        if key in nat:
            return float(nat[key])
    raise ValueError("전국 기준값(전체/소계)을 찾을 수 없습니다.")


def relativities(df: pd.DataFrame, metric_contains: str, year: int) -> dict:
    """차원별 주변 상대위험(R)을 계산한다.

    Returns dict: {"national": float, "sex": {...}, "age": {...}, "sido": {...}}
    """
    national = _national_rate(df, metric_contains, year)
    sex = {k: v / national for k, v in _level_map(df, "성별", metric_contains, year).items()}
    age = {k: v / national for k, v in _level_map(df, "연령별", metric_contains, year).items()}
    sido = {k: v / national for k, v in _level_map(df, "시도별", metric_contains, year).items()}
    return {"national": national, "sex": sex, "age": age, "sido": sido}


# ---------------------------------------------------------------------------
# 보장항목별 발생률 surface
# ---------------------------------------------------------------------------

def _proportional_surface(rel: dict, per_1000: bool = True) -> pd.DataFrame:
    """주변 상대위험으로 성×연령×시도 발생률 surface를 생성한다."""
    national = rel["national"]
    rows = []
    for sido, r_sido in rel["sido"].items():
        for sex_name, r_sex in rel["sex"].items():
            for age_band, r_age in rel["age"].items():
                rate_100k = national * r_sex * r_age * r_sido
                rows.append(
                    {
                        "sido": sido,
                        "sex": sex_name,
                        "age_band": age_band,
                        "rate_per_100k": rate_100k,
                        "rate_per_1000py": rate_100k / 100.0,
                    }
                )
    out = pd.DataFrame(rows)
    if not per_1000:
        out = out.drop(columns=["rate_per_1000py"])
    return out


def severe_trauma_surface() -> pd.DataFrame:
    """상해사망/후유장해 계열 — 중증외상 발생률 기반 surface."""
    df = loaders.load_severe_trauma_incidence()
    year = _latest_year(df)
    rel = relativities(df, _RATE_METRIC, year)
    surf = _proportional_surface(rel)
    surf["year"] = year
    surf["source"] = "kdca_severe_trauma"
    return surf


def _region_share_map(df: pd.DataFrame) -> dict:
    """시도별 분율(%) → {sido: fraction} (최근연도)."""
    year = _latest_year(df)
    sub = df[(df["metric"].str.contains(_SHARE_METRIC, na=False)) & (df["year"] == year)]
    return {row["sido"]: row["value"] / 100.0 for _, row in sub.iterrows()
            if row["sido"] != "전국"}


def severe_trauma_outcome_surface(outcome: str) -> pd.DataFrame:
    """중증외상 발생률 × 지역별 결과분율 = 상해사망/후유장해 surface.

    outcome: 'fatality' | 'disability' | 'severe_disability'
    분율은 시도 차원만 있으므로 발생률 surface에 시도별로 곱한다.
    전국 분율을 fallback으로 사용한다.
    """
    base = severe_trauma_surface()
    loader = {
        "fatality": loaders.load_trauma_fatality_share,
        "disability": loaders.load_trauma_disability_share,
        "severe_disability": loaders.load_trauma_severe_disability_share,
    }[outcome]
    share_df = loader()
    shares = _region_share_map(share_df)
    national = share_df[
        (share_df["dim"] == "전국")
        & share_df["metric"].str.contains(_SHARE_METRIC, na=False)
    ]["value"]
    national_share = float(national.iloc[-1]) / 100.0 if len(national) else float("nan")

    out = base.copy()
    out["share"] = out["sido"].map(shares).fillna(national_share)
    out["rate_per_100k"] = out["rate_per_100k"] * out["share"]
    out["rate_per_1000py"] = out["rate_per_100k"] / 100.0
    out["source"] = f"kdca_severe_trauma_{outcome}"
    return out.drop(columns=["share"])


def nontrauma_outcome_surface(outcome: str) -> pd.DataFrame:
    """비외상 중증 발생률 × 지역별 결과분율 = 질병사망/질병후유장해 surface.

    outcome: 'fatality'(질병사망) | 'disability'(질병후유장해) | 'severe_disability'
    외상 트랙과 동일 구조. 분율은 시도 차원만 있어 발생 surface에 시도별로 곱한다.
    """
    base = nontrauma_surface()
    loader = {
        "fatality": loaders.load_nontrauma_fatality_share,
        "disability": loaders.load_nontrauma_disability_share,
        "severe_disability": loaders.load_nontrauma_severe_disability_share,
    }[outcome]
    share_df = loader()
    shares = _region_share_map(share_df)
    national = share_df[
        (share_df["dim"] == "전국")
        & share_df["metric"].str.contains(_SHARE_METRIC, na=False)
    ]["value"]
    national_share = float(national.iloc[-1]) / 100.0 if len(national) else float("nan")

    out = base.copy()
    out["share"] = out["sido"].map(shares).fillna(national_share)
    out["rate_per_100k"] = out["rate_per_100k"] * out["share"]
    out["rate_per_1000py"] = out["rate_per_100k"] / 100.0
    out["source"] = f"kdca_nontrauma_{outcome}"
    return out.drop(columns=["share"])


def death_all_surface() -> pd.DataFrame:
    """전체 사망률 surface (KOSIS 사망원인, 시도×성×5세연령).

    질병+외인 합의 전체 사망률. 검증용 envelope. age는 5세 구간(age5)이라
    별도 컬럼명(age5)을 쓴다.
    """
    dc = loaders.load_death_cause()
    year = int(dc["year"].max())
    sub = dc[(dc["year"] == year) & dc["death_rate_100k"].notna()
             & dc["sex_name"].isin(["남자", "여자"])
             & (dc["sido"] != "전국")].copy()
    sub = sub.rename(columns={"sex_name": "sex", "death_rate_100k": "rate_per_100k"})
    sub["rate_per_1000py"] = sub["rate_per_100k"] / 100.0
    sub["year"] = year
    sub["source"] = "kosis_death_cause"
    return sub[["sido", "sex", "age5", "rate_per_100k", "rate_per_1000py", "year", "source"]]


def injury_death_outcome_share(sex: str = "남자") -> float:
    """퇴원손상 치료결과 중 '사망' 분율(성별, 최근연도). 상해사망 일반화용."""
    df = loaders.load_discharge_treatment_outcome()
    year = int(df["year"].max())
    sub = df[(df["dim"] == sex) & (df["level"] == "사망")
             & df["metric"].str.contains(_SHARE_METRIC, na=False) & (df["year"] == year)]
    return float(sub["value"].iloc[0]) / 100.0 if len(sub) else float("nan")


def severe_disability_payout_ratio(track: str = "trauma") -> float:
    """후유장해 정액 지급 보정계수 (지급률 가중).

    장해 중 중증(100% 지급)의 비중과 경증의 평균 지급률로 평균 지급비율을 추정한다.
        ratio = severe_share/disability_share × 1.0
              + (1 - severe_share/disability_share) × PARTIAL
    PARTIAL(경·중등도 평균 지급률)은 보수적으로 0.30 가정.
    """
    PARTIAL = 0.30
    if track == "trauma":
        dis = loaders.load_trauma_disability_share()
        sev = loaders.load_trauma_severe_disability_share()
    else:
        dis = loaders.load_nontrauma_disability_share()
        sev = loaders.load_nontrauma_severe_disability_share()

    def _nat(df):
        y = int(df["year"].max())
        v = df[(df["dim"] == "전국") & df["metric"].str.contains(_SHARE_METRIC, na=False)
               & (df["year"] == y)]["value"]
        return float(v.iloc[-1]) / 100.0 if len(v) else float("nan")

    d, s = _nat(dis), _nat(sev)
    if not d or d <= 0:
        return 1.0
    severe_frac = min(s / d, 1.0)
    return severe_frac * 1.0 + (1 - severe_frac) * PARTIAL


def nontrauma_surface() -> pd.DataFrame:
    """비외상 중증질환 발생률 기반 surface."""
    df = loaders.load_nontrauma_incidence()
    year = _latest_year(df)
    rel = relativities(df, _RATE_METRIC, year)
    surf = _proportional_surface(rel)
    surf["year"] = year
    surf["source"] = "kdca_non_trauma_severe"
    return surf


def _discharge_sex_age_surface() -> pd.DataFrame:
    """퇴원손상 퇴원율의 성×연령 joint 표를 long surface로.

    dim=성별(전체/남자/여자), level=연령(전체/0-14세/...). 진짜 교차표라
    비례 보정 없이 직접 사용한다. 시도 차원은 없다.
    """
    df = loaders.load_discharge_rate_by_sex_age()
    year = _latest_year(df)
    sub = df[(df["metric"].str.contains(_DISCHARGE_RATE_METRIC, na=False)) & (df["year"] == year)]
    rows = []
    for _, r in sub.iterrows():
        if r["dim"] == "전체" or r["level"] == "전체":
            continue  # 주변합은 제외, joint 셀만 사용
        rows.append(
            {
                "sex": r["dim"],
                "discharge_age_band": r["level"],
                "rate_per_100k": r["value"],
            }
        )
    return pd.DataFrame(rows), year


def _fracture_share() -> float:
    """전체 퇴원손상 중 골절이 차지하는 분율(최근연도, 전체 성별)."""
    df = loaders.load_discharge_count_by_injury_type()
    year = _latest_year(df)
    sub = df[
        (df["dim"] == "전체")
        & (df["level"] == "골절")
        & (df["metric"].str.contains(_SHARE_METRIC, na=False))
        & (df["year"] == year)
    ]
    if sub.empty:
        raise ValueError("골절 분율을 찾을 수 없습니다.")
    return float(sub["value"].iloc[0]) / 100.0


def discharge_surface(item: str) -> pd.DataFrame:
    """골절/입원 — 퇴원손상 성×연령 surface에 시도 상대위험(중증외상 proxy)을 결합.

    골절은 전체 퇴원율 × 골절분율로 근사한다(입원=전체 퇴원율 proxy).
    """
    sex_age, year = _discharge_sex_age_surface()
    multiplier = _fracture_share() if item == "fracture" else 1.0

    # 시도 상대위험은 중증외상 시도 주변분포에서 빌려온다(퇴원표에 지역축 없음).
    trauma = loaders.load_severe_trauma_incidence()
    t_year = _latest_year(trauma)
    sido_rel = relativities(trauma, _RATE_METRIC, t_year)["sido"]

    rows = []
    for _, r in sex_age.iterrows():
        for sido, r_sido in sido_rel.items():
            rate_100k = r["rate_per_100k"] * multiplier * r_sido
            rows.append(
                {
                    "sido": sido,
                    "sex": r["sex"],
                    "discharge_age_band": r["discharge_age_band"],
                    "rate_per_100k": rate_100k,
                    "rate_per_1000py": rate_100k / 100.0,
                }
            )
    out = pd.DataFrame(rows)
    out["year"] = year
    out["source"] = "kdca_discharge_injury"
    out["note"] = "시도 상대위험은 중증외상 주변분포 proxy"
    return out


# ---------------------------------------------------------------------------
# 통합 진입점
# ---------------------------------------------------------------------------

def coverage_rate_surface(item: str) -> pd.DataFrame:
    """보장항목 키 → 발생률 surface DataFrame."""
    src = config.COVERAGE_ITEMS[item]["source"]
    if src == "severe_trauma":
        surf = severe_trauma_surface()
    elif src == "severe_trauma_fatality":
        surf = severe_trauma_outcome_surface("fatality")
    elif src == "severe_trauma_disability":
        surf = severe_trauma_outcome_surface("disability")
    elif src == "nontrauma":
        surf = nontrauma_surface()
    elif src == "nontrauma_fatality":
        surf = nontrauma_outcome_surface("fatality")
    elif src == "nontrauma_disability":
        surf = nontrauma_outcome_surface("disability")
    elif src == "discharge_injury":
        surf = discharge_surface(item)
    elif src == "death_cause":
        surf = death_all_surface()
    else:
        raise ValueError(f"미지원 source: {src}")
    surf["coverage_item"] = item
    surf["coverage_label"] = config.COVERAGE_ITEMS[item]["label"]
    return surf


def conscript_item_rate(item: str, sido: str | None = None) -> pd.Series | float:
    """현역 청년(20대 남성) 보장항목 발생률(1,000명당/년).

    age5(death_cause)·discharge·KDCA10세 구간을 각각 20대 남성 셀로 맞춘다.
    sido 지정 시 스칼라, 아니면 시도 인덱스 Series 반환.
    """
    surf = coverage_rate_surface(item)
    male = surf["sex"].eq("남자")
    if "age5" in surf:                       # death_cause: 20-24 + 25-29 평균
        m = male & surf["age5"].isin(["20-24세", "25-29세"])
        sub = surf[m].groupby("sido")["rate_per_1000py"].mean()
    elif "discharge_age_band" in surf:       # 퇴원표: 15-24/25-34 평균
        m = male & surf["discharge_age_band"].isin(["15-24세", "25-34세"])
        sub = surf[m].groupby("sido")["rate_per_1000py"].mean()
    else:                                    # KDCA 10세: 20-29세
        m = male & surf["age_band"].eq(config.CONSCRIPT_KDCA_AGE_BAND)
        sub = surf[m].groupby("sido")["rate_per_1000py"].mean()
    if sido is not None:
        return float(sub.get(sido, float("nan")))
    return sub


def conscript_rate_table(items: list[str] | None = None) -> pd.DataFrame:
    """현역 청년(20대 남성) 보장항목별 시도 발생률 표(1,000명당/년).

    기본은 보장(급부) 항목 + 발생 기준 + 전체사망(검증) 전부.
    """
    items = items or list(config.COVERAGE_ITEMS)
    frames = []
    for item in items:
        sub = conscript_item_rate(item).rename("rate_per_1000py").reset_index()
        sub["coverage_label"] = config.COVERAGE_ITEMS[item]["label"]
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    return long.pivot_table(index="sido", columns="coverage_label",
                            values="rate_per_1000py").round(3)

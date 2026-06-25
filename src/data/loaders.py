"""실데이터 로더.

개인 단위(CHS/KNHANES)와 집계 단위(KDCA 발생률/퇴원) 데이터를
일관된 DataFrame으로 읽어온다. 모든 로더는 committed 실데이터만 사용한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .encoding import detect_encoding
from .reshape import read_wide_pivot

# ---------------------------------------------------------------------------
# 개인 단위 조사 데이터
# ---------------------------------------------------------------------------

# CHS 결측/모름 코드(설문 항목): 7=응답거부, 8=비해당, 9=모름 등.
# BMI 계산용 키/몸무게에는 이런 코드가 없지만 행태 변수에 존재한다.


def load_chs() -> pd.DataFrame:
    """지역사회건강조사 개인 데이터.

    BMI 파생(oba_02z1=키cm, oba_03z1=몸무게kg)과 시도명 매핑을 추가한다.
    """
    df = pd.read_csv(config.CHS_CSV, encoding=detect_encoding(config.CHS_CSV))
    df.columns = [c.strip() for c in df.columns]

    df["sido"] = df["ctprvn_code"].map(config.SIDO_CODE_TO_NAME)
    df["sex_name"] = df["sex"].map(config.SEX_CODE_TO_NAME)
    df["bmi"] = _bmi(df["oba_02z1"], df["oba_03z1"])
    df["age_band"] = df["age"].apply(age_to_kdca_band)
    return df


def load_knhanes() -> pd.DataFrame:
    """국민건강영양조사 개인 데이터 (BMI 직접 측정값 보유)."""
    df = pd.read_csv(config.KNHANES_CSV, encoding=detect_encoding(config.KNHANES_CSV))
    df.columns = [c.strip() for c in df.columns]
    df["sex_name"] = df["sex"].map(config.SEX_CODE_TO_NAME)
    df["age_band"] = df["age"].apply(age_to_kdca_band)
    # HE_BMI가 있으면 그대로 사용, 없으면 키/몸무게로 파생
    if "HE_BMI" in df:
        df["bmi"] = df["HE_BMI"]
    else:
        df["bmi"] = _bmi(df["HE_ht"], df["HE_wt"])
    return df


def load_knhanes_features() -> pd.DataFrame:
    """KNHANES 개인 피처 프레임 (M1 적재 — 만성질환·행태·BMI).

    측정 BMI + 흡연(BS3_1) + 신체활동(BE3_71 고강도/BE3_81 중강도/pa_aerobic) +
    만성질환(DI1_dg 고혈압/DE1_dg 당뇨) + 연령·성을 모델 입력용으로 정규화한다.
    값코드: DI1_dg/DE1_dg 0=무,1=유,8=비해당; BE3_* 1=예,2=아니오,8/9=결측.
    ⚠️ 손상(IJMT) 결과변수는 본 처리본에 없음 → M2 지도학습은 라벨 부재로 불가.
    """
    df = load_knhanes()
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["sex"] = df["sex"]
    out["bmi"] = df["bmi"]
    smk = df["BS3_1"]
    out["smoker"] = np.where(smk.isin([1, 2]), 1.0,
                             np.where(smk.isin([3, 8]), 0.0, np.nan))
    out["vigorous_pa"] = df["BE3_71"].map({1: 1.0, 2: 0.0})
    out["moderate_pa"] = df["BE3_81"].map({1: 1.0, 2: 0.0})
    out["aerobic_pa"] = pd.to_numeric(df.get("pa_aerobic"), errors="coerce")
    out["hypertension"] = df["DI1_dg"].map({0: 0.0, 1: 1.0})
    out["diabetes"] = df["DE1_dg"].map({0: 0.0, 1: 1.0})
    out["weight"] = df["wt_itvex"]
    return out


def _bmi(height_cm: pd.Series, weight_kg: pd.Series) -> pd.Series:
    """BMI = kg / m^2. 비정상치(키<100cm 등)는 NaN 처리."""
    h = pd.to_numeric(height_cm, errors="coerce")
    w = pd.to_numeric(weight_kg, errors="coerce")
    h = h.where((h >= 100) & (h <= 230))
    w = w.where((w >= 20) & (w <= 250))
    return w / (h / 100.0) ** 2


def age_to_kdca_band(age) -> str | float:
    """만나이 → KDCA 10세 구간 라벨."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return float("nan")
    if a < 0:
        return float("nan")
    if a >= 80:
        return "80세 이상"
    lo = int(a // 10) * 10
    return f"{lo}-{lo + 9}세"


# ---------------------------------------------------------------------------
# KDCA 집계(발생률/퇴원) 데이터 — long 포맷
# ---------------------------------------------------------------------------


def load_severe_trauma_incidence() -> pd.DataFrame:
    """중증외상 발생률 (성/연령/시도 주변분포, 인구 10만명당)."""
    return read_wide_pivot(config.TRAUMA_DIR / "trauma_incidence_by_sex_age_region.csv")


def load_nontrauma_incidence() -> pd.DataFrame:
    """비외상 중증질환 발생률 (성/연령/시도 주변분포, 인구 10만명당)."""
    return read_wide_pivot(
        config.NONTRAUMA_DIR / "nontrauma_incidence_by_sex_age_region.csv"
    )


def load_discharge_rate_by_sex_age() -> pd.DataFrame:
    """퇴원손상 퇴원율 (성×연령 교차표, 인구 10만명당).

    이 표는 dim=성별(전체/남자/여자), level=연령구간의 진짜 교차표다.
    """
    return read_wide_pivot(config.DISCHARGE_DIR / "discharge_rate_by_sex_age.csv")


def load_discharge_count_by_injury_type() -> pd.DataFrame:
    """손상양상별(골절 등) 퇴원환자수·분율 (성별 × 손상양상)."""
    return read_wide_pivot(
        config.DISCHARGE_DIR / "discharge_count_by_injury_type.csv"
    )


# ---------------------------------------------------------------------------
# 중증외상 결과 분율(시도별) — 치명률·장해율
#   '분율 (%)' = 해당 결과 건수 / 중증외상 발생 건수 (예: 사망 4467 / 발생 8170 = 54.7%).
#   시도명은 정식명칭("서울특별시")이므로 단축명으로 정규화한다.
#   시도 차원만 있으므로 read_wide_pivot(n_id_cols=1)로 읽는다.
# ---------------------------------------------------------------------------

def _load_region_share(filename: str, directory) -> pd.DataFrame:
    df = read_wide_pivot(directory / filename, n_id_cols=1)
    df["sido"] = df["dim"].map(config.KOSIS_SIDO_ALIASES).fillna(df["dim"])
    return df


def load_trauma_fatality_share() -> pd.DataFrame:
    """중증외상 치명률(시도별)."""
    return _load_region_share("trauma_fatality_by_region.csv", config.TRAUMA_DIR)


def load_trauma_disability_share() -> pd.DataFrame:
    """중증외상 장해율(시도별)."""
    return _load_region_share("trauma_disability_by_region.csv", config.TRAUMA_DIR)


def load_trauma_severe_disability_share() -> pd.DataFrame:
    """중증외상 중증장해율(시도별)."""
    return _load_region_share("trauma_severe_disability_by_region.csv", config.TRAUMA_DIR)


# 비외상성 중증손상 결과 분율(질병 트랙) — 구조는 외상과 동일
def load_nontrauma_fatality_share() -> pd.DataFrame:
    """비외상성 중증손상 치명률(시도별)."""
    return _load_region_share("nontrauma_fatality_by_region.csv", config.NONTRAUMA_DIR)


def load_nontrauma_disability_share() -> pd.DataFrame:
    """비외상성 중증손상 장해율(시도별)."""
    return _load_region_share("nontrauma_disability_by_region.csv", config.NONTRAUMA_DIR)


def load_nontrauma_severe_disability_share() -> pd.DataFrame:
    """비외상성 중증손상 중증장해율(시도별)."""
    return _load_region_share("nontrauma_severe_disability_by_region.csv", config.NONTRAUMA_DIR)


def load_discharge_treatment_outcome() -> pd.DataFrame:
    """퇴원손상 치료결과별(성별 × 결과: 사망/완쾌 등) 분율."""
    return read_wide_pivot(config.DISCHARGE_DIR / "discharge_count_by_treatment_outcome.csv")


# ---------------------------------------------------------------------------
# KOSIS — 장래인구추계 / 사망원인
#   둘 다 [식별열…, 1996년 … 2035년] 형태의 단일헤더 와이드. 연도열을 melt한다.
#   시도명은 정식명칭이라 단축명으로 정규화, 연령은 "20 - 24세" 공백 형식.
# ---------------------------------------------------------------------------

def _melt_year_columns(df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    """연도(헤더가 숫자인) 열들을 long으로 melt한다."""
    year_cols = [c for c in df.columns if str(c).strip()[:4].isdigit()]
    long = df.melt(id_vars=id_cols, value_vars=year_cols,
                   var_name="year", value_name="value")
    long["year"] = long["year"].astype(str).str.extract(r"(\d{4})").astype(int)
    long["value"] = pd.to_numeric(
        long["value"].astype(str).str.replace(",", "").str.strip()
        .replace({"-": None, "": None}), errors="coerce")
    return long


def normalize_age5(label: str) -> str:
    """'20 - 24세' / '20-24세' → '20-24세' 표준화."""
    return str(label).replace(" ", "")


def load_population_projection() -> pd.DataFrame:
    """장래인구추계 (시도×성×5세 연령, 연도별 인구).

    Returns long DataFrame[sido, sex_name, age5, year, value(명)].
    중위추계 기준. 성별 '계'와 연령 '계'도 포함하되 sido는 단축명.
    """
    raw = pd.read_csv(config.POPULATION_CSV, encoding=detect_encoding(config.POPULATION_CSV))
    raw.columns = [str(c).strip() for c in raw.columns]
    id_cols = list(raw.columns[:4])  # 시나리오, 시도, 성별, 연령
    long = _melt_year_columns(raw, id_cols)
    long = long.rename(columns={id_cols[1]: "sido_raw", id_cols[2]: "sex_name",
                                id_cols[3]: "age5"})
    long["sido"] = long["sido_raw"].map(config.KOSIS_SIDO_ALIASES).fillna(long["sido_raw"])
    long["age5"] = long["age5"].map(normalize_age5)
    return long[["sido", "sex_name", "age5", "year", "value"]]


def load_population_resident() -> pd.DataFrame:
    """주민등록 연앙인구 (2005–2019, 시도×남×20-24·25-29).

    Returns long DataFrame[sido, sex_name, age5, year, value(명, 연앙)].
    연앙인구이므로 person-years로 직접 사용. 세종은 2012 신설이라 2012–2019만 존재.
    """
    raw = pd.read_csv(config.RESIDENT_POP_CSV, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    region_col = next(c for c in raw.columns if "행정구역" in c or "시도" in c)
    age_cols = [c for c in raw.columns if "세" in c]
    raw = raw[raw[region_col] != "전국"]
    long = raw.melt(id_vars=["시점", region_col, "성별"], value_vars=age_cols,
                    var_name="age5", value_name="value")
    long["year"] = long["시점"].astype(str).str.extract(r"(\d{4})").astype(int)
    long["sido"] = long[region_col].map(config.KOSIS_SIDO_ALIASES).fillna(long[region_col])
    long["sex_name"] = long["성별"].astype(str).str.strip()
    long["age5"] = long["age5"].map(normalize_age5)
    long["value"] = pd.to_numeric(
        long["value"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return long[["sido", "sex_name", "age5", "year", "value"]].dropna(subset=["value"])


def load_population_midyear() -> pd.DataFrame:
    """2005–2024 결합 person-years: 2005–2019 주민등록 연앙 + 2020–2024 추계.

    두 소스는 연도가 disjoint(겹침 없음)라 단순 병합(경계 규칙: 2005–2019=주민등록,
    2020–2024=추계). 주민등록 범위가 남자 20-24·25-29뿐이라 2005–2019는 그 셀만 제공.
    """
    res = load_population_resident()                  # 2005–2019
    proj = load_population_projection()
    proj = proj[proj["year"].between(2020, 2024)]     # 2020–2024
    return pd.concat([res, proj], ignore_index=True)


def load_death_cause() -> pd.DataFrame:
    """사망원인 통계 (시도×성×5세, 전체 사인). cp949 인코딩.

    Returns long DataFrame[sido, sex_name, age5, year, deaths, death_rate_100k].
    이 추출본은 '계'(전체 사인)만 포함 → 전체 사망률(질병+외인 합)로 사용.
    """
    raw = pd.read_csv(config.DEATH_CAUSE_CSV, encoding="cp949")
    raw.columns = [str(c).strip() for c in raw.columns]
    # 열: 사인, 시도, 연령, 성별, 항목, 단위, 연도들…
    id_cols = list(raw.columns[:6])
    long = _melt_year_columns(raw, id_cols)
    long = long.rename(columns={id_cols[1]: "sido_raw", id_cols[2]: "age5",
                                id_cols[3]: "sex_name", id_cols[4]: "item"})
    long["sido"] = long["sido_raw"].map(config.KOSIS_SIDO_ALIASES).fillna(long["sido_raw"])
    long["age5"] = long["age5"].map(normalize_age5)
    deaths = long[long["item"].str.contains("사망자수", na=False)][
        ["sido", "sex_name", "age5", "year", "value"]].rename(columns={"value": "deaths"})
    rate = long[long["item"].str.contains("사망률", na=False)][
        ["sido", "sex_name", "age5", "year", "value"]].rename(columns={"value": "death_rate_100k"})
    return deaths.merge(rate, on=["sido", "sex_name", "age5", "year"], how="outer")


# 국방부 사망사고 통계 — 위치 기반 컬럼(원본 헤더가 인코딩 손상으로 U+FFFD 대체문자라
#   한글 복구 불가, 컬럼 '순서·구조'는 온전 → 개최기관 제공 스키마대로 위치 매핑).
#   스키마: idx, 년도, 군기사고(자살·총기·폭행·기타), 안전사고(차량·함정항공·폭발·
#           추락충격·익사·화재·기타), 군자살률(10만명당), 민간자살률(10만명당).
_MND_COLS = ["idx", "year",
             "disc_suicide", "disc_gun", "disc_assault", "disc_other",
             "safe_vehicle", "safe_shipair", "safe_explosion", "safe_fall",
             "safe_drown", "safe_fire", "safe_other",
             "mil_suicide_rate", "civ_suicide_rate"]
_MND_SAFETY = ["safe_vehicle", "safe_shipair", "safe_explosion", "safe_fall",
               "safe_drown", "safe_fire", "safe_other"]
_MND_DISC_OTHER = ["disc_gun", "disc_assault", "disc_other"]


def load_mnd_death_accidents() -> pd.DataFrame:
    """국방부 사망사고 통계(2011–2025) — 군 코호트 proxy 검증·자살 보정용(전국 단위).

    ★헤더 한글이 원본 인코딩 손상(U+FFFD)이라 위치 기반으로 컬럼명을 부여한다(개최기관
      제공 스키마 순서 + 숫자 sanity check로 검증). 사고 항목은 '건수(명)', 자살률은
      '10만명당'. 시도분해·연령분해 없음 → 전국 단위로만 쓴다.

    파생:
      · military_suicide        = disc_suicide               (자살, 면책 보정용)
      · military_external       = 안전사고 7종 합            (외인 검증용)
      · military_discipline_other = 총기+폭행+기타           (★작전/징계성 — 외인에 합치지
                                                              않음, 검증·보정 제외)
      · troops_implied          = disc_suicide / mil_suicide_rate × 1e5  (병력 분모 역산;
                                  2011 63.8만→2025 54.9만, 알려진 병력 추이와 정합)
    ★2025 민간자살률은 결측(원본 빈칸) → NaN 유지(0/빈값 대체 금지). 비율 계산에서 제외.
    """
    path = config.MND_DEATH_ACCIDENT_CSV
    df = pd.read_csv(path, header=0, names=_MND_COLS,
                     encoding=detect_encoding(path)).drop(columns="idx")
    # 숫자화(2025 민간자살률 빈칸 → NaN)
    for c in _MND_COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["military_suicide"] = df["disc_suicide"]
    df["military_external"] = df[_MND_SAFETY].sum(axis=1)
    df["military_discipline_other"] = df[_MND_DISC_OTHER].sum(axis=1)
    df["troops_implied"] = df["disc_suicide"] / df["mil_suicide_rate"] * 1e5
    return df


# ---------------------------------------------------------------------------
# 병무청 병역판정 신체검사 통계 (청별, 연도별 와이드)
#   다중 헤더(연도 / 구분 / 급|체급 / 단위)라 일반 reshaper 대신 전용 파서를 쓴다.
#   전국 추정에는 '계' 행만 필요하므로 연도별 열그룹을 인식해 합산/비율을 뽑는다.
# ---------------------------------------------------------------------------

def _mma_rows(filename: str):
    import csv
    path = config.MMA_DIR / filename
    with path.open(encoding=detect_encoding(path), newline="") as f:
        return list(csv.reader(f))


def _mma_year_cols(year_header: list[str]) -> dict[int, list[int]]:
    """연도 헤더 → {연도: [열 인덱스…]}."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(year_header):
        digits = "".join(ch for ch in str(c) if ch.isdigit())
        if len(digits) >= 4:
            groups.setdefault(int(digits[:4]), []).append(i)
    return groups


def _to_num(x: str) -> float:
    x = str(x).replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return float("nan")


def load_mma_service_rate(year: int | None = None) -> dict:
    """병무청 등급판정 → 전국 현역 판정비율(1~3급/수검인원).

    Returns {"year", "exam", "active", "active_rate", "by_grade": {...}}.
    헤더: r0=연도, r1=구분(수검인원/복무대상/…), r2=급(소계/1급/…), r3=단위(인원수/비율).
    데이터는 r4부터, '계' 행 사용.
    """
    rows = _mma_rows("judgment_by_grade.csv")
    year_h, group_h, grade_h, unit_h = rows[0], rows[1], rows[2], rows[3]
    groups = _mma_year_cols(year_h)
    year = year or max(groups)
    cols = groups[year]

    gye = next(r for r in rows[4:] if r[0].strip() == "계")
    exam = None
    grade_counts: dict[str, float] = {}
    for ci in cols:
        unit = unit_h[ci].strip()
        grade = grade_h[ci].strip()
        group = group_h[ci].strip()
        val = _to_num(gye[ci])
        if group == "수검인원":
            exam = val
        elif unit.startswith("인원수") and grade in ("1급", "2급", "3급", "4급", "5급", "6급"):
            grade_counts[grade] = val
    active = sum(grade_counts.get(g, 0.0) for g in ("1급", "2급", "3급"))
    return {
        "year": year, "exam": exam, "active": active,
        "active_rate": active / exam if exam else float("nan"),
        "by_grade": grade_counts,
    }


def load_mma_bmi_distribution(year: int | None = None) -> dict:
    """병무청 BMI 판정 → 전국 BMI 구간 분포(비율).

    Returns {"year", "total", "dist": {"저체중":r, "정상":r, "과체중":r, "비만":r}}.
    헤더: r0=연도, r1=BMI구간, r2=단위(인원수/비율). 데이터 r3부터, '계' 행 사용.
    """
    rows = _mma_rows("judgment_by_bmi.csv")
    year_h, band_h, unit_h = rows[0], rows[1], rows[2]
    groups = _mma_year_cols(year_h)
    year = year or max(groups)
    cols = groups[year]
    gye = next(r for r in rows[3:] if r[0].strip() == "계")

    label_map = {"저체중": "저체중", "정상": "정상", "과체중": "과체중", "비만": "비만"}
    dist, total = {}, None
    for ci in cols:
        unit = unit_h[ci].strip()
        band = band_h[ci].strip()
        val = _to_num(gye[ci])
        if band == "계":
            total = val
        elif unit.startswith("비율"):
            for key, norm in label_map.items():
                if key in band:
                    dist[norm] = val / 100.0
    return {"year": year, "total": total, "dist": dist}

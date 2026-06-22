"""실데이터 로더.

개인 단위(CHS/KNHANES)와 집계 단위(KDCA 발생률/퇴원) 데이터를
일관된 DataFrame으로 읽어온다. 모든 로더는 committed 실데이터만 사용한다.
"""
from __future__ import annotations

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

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

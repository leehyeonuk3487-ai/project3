"""모집단 트랙 — 지자체 현역 청년 N + BMI 분포 + 미래 투영.

수동 N을 제거하기 위해 KOSIS 장래인구추계(시도×성×5세)와 병무청 신체검사
통계(현역판정비율·BMI 분포)를 결합한다.

현역 stock 추정(표준 가정):
    연간 입영 ≈ (20-24세 남성 인구 / 5) × 현역판정비율
    현역 stock ≈ 연간 입영 × 평균 복무연수(기본 1.5년)

  - 입영은 사실상 20~22세 집중 → 단년 코호트로 연간 입영을 근사.
  - 평균 복무연수 1.5년(육군 18개월 기준; 상근·의경·의소방 포함 시 소폭 증가).
검증: 경기도 추정이 보고치(6.8만~10.5만)와 부합하는지 대조.

⚠️ 인구추계는 시도 단위까지만 제공된다. 시군구(화천군 등) 단위는 별도
주민등록 인원이 필요하므로 시도값에 local_share를 곱해 근사하거나 수동 입력한다.
"""
from __future__ import annotations

import pandas as pd

from ..data import loaders

CONSCRIPT_AGE5 = ("20-24세", "25-29세")
DEFAULT_SERVICE_YEARS = 1.5  # 평균 복무연수(육군 18개월)

_pop_cache: pd.DataFrame | None = None


def _population() -> pd.DataFrame:
    global _pop_cache
    if _pop_cache is None:
        _pop_cache = loaders.load_population_projection()
    return _pop_cache


def young_male_pop(sido: str, year: int, bands=CONSCRIPT_AGE5) -> float:
    """해당 시도·연도의 20대 남성 인구(20-24 + 25-29)."""
    pop = _population()
    m = (pop["sido"] == sido) & (pop["sex_name"] == "남자") & \
        (pop["age5"].isin(bands)) & (pop["year"] == year)
    return float(pop[m]["value"].sum())


def service_rate() -> float:
    """전국 현역 판정비율(1~3급)."""
    return float(loaders.load_mma_service_rate()["active_rate"])


def conscript_stock(sido: str, year: int,
                    service_years: float = DEFAULT_SERVICE_YEARS) -> float:
    """현역 복무 청년 stock 추정(피보험 모집단)."""
    annual_intake = young_male_pop(sido, year, ("20-24세",)) / 5.0
    return annual_intake * service_rate() * service_years


def project_conscripts(sido: str, years: range | list[int],
                       service_years: float = DEFAULT_SERVICE_YEARS) -> pd.DataFrame:
    """연도별 현역 청년 stock 투영(인구절벽 시뮬)."""
    rows = [{"year": y, "conscripts": round(conscript_stock(sido, y, service_years)),
             "young_male_pop": round(young_male_pop(sido, y))}
            for y in years]
    return pd.DataFrame(rows)


def bmi_distribution() -> dict:
    """전국 현역 신체검사 BMI 구간 분포."""
    return loaders.load_mma_bmi_distribution()["dist"]


def available_years() -> tuple[int, int]:
    pop = _population()
    return int(pop["year"].min()), int(pop["year"].max())

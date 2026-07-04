"""지자체 군복무 청년 상해보험 급부 스케줄 (구조화).

출처: docs/research/local_gov_insurance_benefits.md
금액 단위는 '원'. 보장항목 키는 발생률 모델(config.COVERAGE_ITEMS)과 연결된다.
지급 구조(payout_type):
  - lump      : 사건당 정액(사망). 후유장해도 lump이나 지급률(3~100%) 가중 보정을
                예산 단계에서 적용한다.
  - per_event : 발생 회당 진단금(골절 등)
  - per_day   : 입원일당(평균 재원일수 × 일당, 한도일수 적용)

다수 지자체가 "상해·질병 사망 및 후유장해"를 동일 금액으로 보장하므로
질병 항목(death_disease/disease_disability)에도 같은 금액을 매핑한다.
일부 지자체(중구·금천구·동작구)는 보고서에 일부 금액만 공개되어 동종 지자체의
표준액으로 근사하고 approx=True로 표기한다.
"""
from __future__ import annotations

HOSPITAL_DAY_CAP = 180
_MAN = 10_000  # 만원

# 후유장해 지급률 가중에 영향받는 항목
DISABILITY_ITEMS = ("disability", "disease_disability")


def _schedule(death, disability, fracture=None, hosp=None, *, disease=True, approx=False):
    """상해/질병 대칭 급부를 한 번에 구성하는 헬퍼."""
    s = {
        "death_injury": {"payout_type": "lump", "amount": death},
        "disability": {"payout_type": "lump", "amount": disability},
    }
    if disease:
        s["death_disease"] = {"payout_type": "lump", "amount": death}
        s["disease_disability"] = {"payout_type": "lump", "amount": disability}
    if fracture is not None:
        s["fracture"] = {"payout_type": "per_event", "amount": fracture}
    if hosp is not None:
        s["hospitalization"] = {"payout_type": "per_day", "amount": hosp}
    if approx:
        for v in s.values():
            v["approx"] = True
    return s


SCHEDULES: dict[str, dict] = {
    "경기도":      _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 4*_MAN),
    "인천광역시":   _schedule(1000*_MAN, 1000*_MAN, 10*_MAN, 1.5*_MAN),
    "광주광역시":   _schedule(3000*_MAN, 3000*_MAN, 30*_MAN, 3*_MAN),
    "강원_화천군":  _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 3*_MAN),
    "충남_공주시":  _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 3*_MAN),
    "충남_서산시":  _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 3*_MAN),
    "서울_강북구":  _schedule(3000*_MAN, 2000*_MAN, 15*_MAN, 3*_MAN),
    # 보고서 일부 공개 — 표준액 근사(approx)
    "서울_동작구":  _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 3*_MAN, approx=True),
    "서울_중구":    _schedule(5000*_MAN, 5000*_MAN, 30*_MAN, 3*_MAN, approx=True),
    "서울_금천구":  _schedule(5000*_MAN, 5000*_MAN, 15*_MAN, 3*_MAN, approx=True),
}

# 지자체 → 보험료 적용 시도(발생률 surface 조회) + 시도 내 인원 비중(시군구 보정)
#   광역(시도 전체)은 local_share=1.0, 시군구는 별도 주민등록 기준이 필요해
#   추정이 어려운 경우 None(예산 시 명시적 인원 입력 권장).
SCHEDULE_SIDO = {
    "경기도": "경기", "인천광역시": "인천", "광주광역시": "광주",
    "강원_화천군": "강원", "충남_공주시": "충남", "충남_서산시": "충남",
    "서울_강북구": "서울", "서울_동작구": "서울", "서울_중구": "서울",
    "서울_금천구": "서울",
}

# 광역 단위 = True(인구추계로 N 자동 산출 가능), 시군구 = False(인원 입력 권장)
IS_PROVINCE_WIDE = {
    "경기도": True, "인천광역시": True, "광주광역시": True,
    "강원_화천군": False, "충남_공주시": False, "충남_서산시": False,
    "서울_강북구": False, "서울_동작구": False, "서울_중구": False,
    "서울_금천구": False,
}

# 보고된 1인당 보험료(원) — 검증·비교용. 대부분 지자체는 보험료를 비공개하고
# 보장금액만 공개한다. 아래는 공개 확인된 소수 사례:
#   경기도  47,920원 (도 공개, 2019 제정·최근 조정)
#   동작구  48,452원 (서울 자치구 최초, 2024; 2026년 49,843원)
# ※실제 계약은 입원비·수술비·PTSD 등 부가항목을 포함해, 핵심 6항목만 모델링하는
#   본 추정치(하한)보다 높은 게 정상이다(경기도·동작구 모두 동일 구도).
REPORTED_PREMIUM = {"경기도": 47_920, "서울_동작구": 48_452}


def list_schedules() -> list[str]:
    return list(SCHEDULES)


def is_approx(name: str) -> bool:
    return any(v.get("approx") for v in SCHEDULES[name].values())

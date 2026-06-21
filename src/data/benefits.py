"""지자체 군복무 청년 상해보험 급부 스케줄 (구조화).

출처: docs/research/local_gov_insurance_benefits.md
금액 단위는 '원'. 보장항목 키는 발생률 모델(config.COVERAGE_ITEMS)과 연결된다.
지급 구조(payout_type):
  - lump   : 사건당 정액 지급(사망·후유장해). 후유장해는 지급률 가중이라
             정액을 곱하면 상한 추정(over-estimate)임을 명시.
  - per_event : 발생 회당 진단금(골절 등)
  - per_day   : 입원일당(평균 재원일수 × 일당, 한도일수 적용)

매칭되지 않는 항목(정신질환위로금·수술비·화상·뇌졸중 등)은 발생률 소스가
없어 제외한다(예산 추정은 핵심 4개 항목의 하한·근사).
"""
from __future__ import annotations

# 입원일당 한도(일) — 대부분 지자체 공통 180일
HOSPITAL_DAY_CAP = 180

# 단위: 원
_MAN = 10_000  # 만원

# 지자체별 급부 스케줄. value는 보장금액(원).
SCHEDULES: dict[str, dict] = {
    "경기도": {
        "death_injury": {"payout_type": "lump", "amount": 5000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 5000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 30 * _MAN},
        "hospitalization": {"payout_type": "per_day", "amount": 4 * _MAN},
    },
    "인천광역시": {
        "death_injury": {"payout_type": "lump", "amount": 1000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 1000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 10 * _MAN},  # 골절위로금
        "hospitalization": {"payout_type": "per_day", "amount": 1.5 * _MAN},
    },
    "광주광역시": {
        "death_injury": {"payout_type": "lump", "amount": 3000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 3000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 30 * _MAN},
        "hospitalization": {"payout_type": "per_day", "amount": 3 * _MAN},
    },
    "강원_화천군": {
        "death_injury": {"payout_type": "lump", "amount": 5000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 5000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 30 * _MAN},
        "hospitalization": {"payout_type": "per_day", "amount": 3 * _MAN},
    },
    "충남_공주시": {
        "death_injury": {"payout_type": "lump", "amount": 5000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 5000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 30 * _MAN},
        "hospitalization": {"payout_type": "per_day", "amount": 3 * _MAN},
    },
    "서울_강북구": {
        "death_injury": {"payout_type": "lump", "amount": 3000 * _MAN},
        "disability": {"payout_type": "lump", "amount": 2000 * _MAN},
        "fracture": {"payout_type": "per_event", "amount": 15 * _MAN},
        "hospitalization": {"payout_type": "per_day", "amount": 3 * _MAN},
    },
}

# 지자체 → 보험료 적용 시도(발생률 surface 조회용)
SCHEDULE_SIDO = {
    "경기도": "경기", "인천광역시": "인천", "광주광역시": "광주",
    "강원_화천군": "강원", "충남_공주시": "충남", "서울_강북구": "서울",
}

# 참고: 보고된 1인당 보험료(원) — 검증·비교용
REPORTED_PREMIUM = {
    "경기도": 47_920,  # 2024년 기준
}


def list_schedules() -> list[str]:
    return list(SCHEDULES)

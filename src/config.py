"""경로·상수·코드 매핑 중앙 정의.

OS 종속 코드를 피하기 위해 모든 경로는 pathlib 기준으로 잡는다
(Windows/Cursor에서도 동일하게 동작).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"


def load_env(path: Path | None = None) -> dict:
    """간단한 .env 파서(외부 의존성 없이). 키는 환경변수로도 읽는다."""
    import os
    env = {}
    path = path or (ROOT / ".env")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # 실제 환경변수가 있으면 우선
    for k in list(env):
        env[k] = os.environ.get(k, env[k])
    return env

# 개인 단위 처리 데이터 (utf-8-sig, BOM 포함)
CHS_CSV = PROCESSED / "chs_filtered_2021_2025.csv"
KNHANES_CSV = PROCESSED / "knhanes_filtered_2021_2024.csv"

# KDCA 발생률/퇴원 (와이드 2중헤더 피벗)
TRAUMA_DIR = RAW / "kdca_severe_trauma"
NONTRAUMA_DIR = RAW / "kdca_non_trauma_severe"
DISCHARGE_DIR = RAW / "kdca_discharge_injury"

# KOSIS
POPULATION_CSV = RAW / "kosis_population" / "population_projection_sex_age5_region.csv"
# 주민등록 연앙인구(2005–2019, 시도×남×20-24·25-29) — 모듈 A 외인 패널 과거 분모.
RESIDENT_POP_CSV = RAW / "kosis_population" / "resident_registration_by_region_sex_age5.csv"
DEATH_CAUSE_CSV = RAW / "kosis_death_cause" / "death_cause_by_region_sex_age5.csv"
# M0: 사인분류 사망(시도×사인×성×5세연령×연도, 사망자수+사망률, ICD 코드 동반).
#     22개 상호배타 leaf — 질병/외인/자살/기타 4범주 직접 분리(src/data/mortality.py).
DEATH_CAUSE_BY_CAUSE_CSV = RAW / "kosis_death_cause" / "death_cause_by_region_sex_age5_detail.csv"
# 국방부 사망사고 통계(2011–2025, data.go.kr 개최기관 직접). 군기사고/안전사고 건수 +
#   군·민간 자살률(10만명당). 군 코호트 proxy 타당성 검증·자살 보정용(전국 단위).
MND_DEATH_ACCIDENT_CSV = RAW / "kosis_death_cause" / "death_cause_mnd.csv"

# 병무청
MMA_DIR = RAW / "mma_judgment_status"

# ---------------------------------------------------------------------------
# 시도 코드 ↔ 명 crosswalk
#   CHS ctprvn_code(숫자) ↔ KDCA/KOSIS 시도명
#   CHS는 강원/전북 특별자치도 개편 이전 코드(42/45)를 사용한다.
# ---------------------------------------------------------------------------
SIDO_CODE_TO_NAME = {
    11: "서울", 26: "부산", 27: "대구", 28: "인천", 29: "광주",
    30: "대전", 31: "울산", 36: "세종", 41: "경기", 42: "강원",
    43: "충북", 44: "충남", 45: "전북", 46: "전남", 47: "경북",
    48: "경남", 50: "제주",
}
SIDO_NAME_TO_CODE = {v: k for k, v in SIDO_CODE_TO_NAME.items()}

# KOSIS는 정식 명칭("서울특별시" 등)을 쓰므로 단축명으로 정규화한다.
KOSIS_SIDO_ALIASES = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구",
    "인천광역시": "인천", "광주광역시": "광주", "대전광역시": "대전",
    "울산광역시": "울산", "세종특별자치시": "세종", "경기도": "경기",
    "강원도": "강원", "강원특별자치도": "강원", "충청북도": "충북",
    "충청남도": "충남", "전라북도": "전북", "전북특별자치도": "전북",
    "전라남도": "전남", "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}

# ---------------------------------------------------------------------------
# 성별 코드 (CHS/KNHANES 공통: 1=남, 2=여)
# ---------------------------------------------------------------------------
SEX_CODE_TO_NAME = {1: "남자", 2: "여자"}
SEX_NAME_TO_CODE = {"남자": 1, "여자": 2}

# ---------------------------------------------------------------------------
# 연령대 정의
#   KDCA 발생률표는 10세 구간(0-9, 10-19, ...), 퇴원율표는 (0-14, 15-24, ...)을 쓴다.
#   현역 청년(피보험 모집단)은 사실상 20대 남성에 집중되므로 별도로 표시한다.
# ---------------------------------------------------------------------------
CONSCRIPT_AGE_MIN = 18
CONSCRIPT_AGE_MAX = 29  # 현역 복무 청년의 실질 집중 구간

# KDCA 10세 구간 라벨(발생률표 기준)
KDCA_AGE_BANDS_10Y = [
    "0-9세", "10-19세", "20-29세", "30-39세", "40-49세",
    "50-59세", "60-69세", "70-79세", "80세 이상",
]
CONSCRIPT_KDCA_AGE_BAND = "20-29세"

# ---------------------------------------------------------------------------
# 보장항목 정의
#   지자체 상해보험 보장표(docs/research/local_gov_insurance_benefits.md)의 항목을
#   계리 추정이 가능한 데이터 소스에 매핑한다.
#   source: 어떤 raw 데이터로 발생률을 추정하는지
#
#   [slice 1 범위] 직접 관측되는 '기준 발생률 surface'만 산출한다:
#     - 중증외상 발생, 비외상 중증질환 발생, 골절, 손상 입원
#   [slice 2] 상해사망/후유장해를 중증외상 발생률 × 지역별 치명률·장해율
#     (trauma_fatality_by_region / trauma_disability_by_region의 분율)로 분해.
#     분율 = 결과건수/발생건수 (사망 4467/발생 8170 = 54.7%)임을 검증함.
# ---------------------------------------------------------------------------
#   [질병 트랙] 비외상성 중증손상 발생률 × 비외상 치명률/장해율로 질병사망·
#     질병후유장해를 산출(외상 트랙과 동일 구조). 전체 사망률(death_cause)은
#     질병+외인 합의 envelope로 검증에 사용.
#
#   track: '상해'|'질병'|'공통'   role: '보장'(급부)|'기준'(발생 base)|'검증'
COVERAGE_ITEMS = {
    # --- 기준 발생(base) ---
    "severe_trauma": {
        "label": "중증외상 발생", "source": "severe_trauma",
        "track": "상해", "role": "기준", "unit": "per_100k",
    },
    "nontrauma_severe": {
        "label": "비외상 중증 발생", "source": "nontrauma",
        "track": "질병", "role": "기준", "unit": "per_100k",
    },
    # --- 상해 트랙 보장 ---
    "death_injury": {
        "label": "상해사망", "source": "m0_external",
        "track": "상해", "role": "보장", "unit": "per_100k",
    },
    "disability": {
        "label": "상해후유장해", "source": "severe_trauma_disability",
        "track": "상해", "role": "보장", "unit": "per_100k",
    },
    "fracture": {
        "label": "골절", "source": "discharge_injury",
        "track": "상해", "role": "보장", "unit": "per_100k",
    },
    "hospitalization": {
        "label": "손상 입원", "source": "discharge_injury",
        "track": "상해", "role": "보장", "unit": "per_100k",
    },
    # --- 질병 트랙 보장 ---
    "death_disease": {
        "label": "질병사망", "source": "m0_disease",
        "track": "질병", "role": "보장", "unit": "per_100k",
    },
    "disease_disability": {
        "label": "질병후유장해", "source": "nontrauma_disability",
        "track": "질병", "role": "보장", "unit": "per_100k",
    },
    # --- 자살(면책) — 보장·예산 미포함, envelope·정직성 노트에만 ---
    "death_suicide_excluded": {
        "label": "자살사망(면책)", "source": "m0_suicide",
        "track": "면책", "role": "면책", "unit": "per_100k",
    },
    # --- 검증(envelope) ---
    "death_all": {
        "label": "전체사망(검증)", "source": "death_cause",
        "track": "공통", "role": "검증", "unit": "per_100k",
    },
}

# 보장(급부) 항목만 — 예산·표 산출의 기본 집합
BENEFIT_ITEMS = [k for k, v in COVERAGE_ITEMS.items() if v["role"] == "보장"]

# ---------------------------------------------------------------------------
# 정직성 라벨 (모든 산출물에 부착)
# ---------------------------------------------------------------------------
HONESTY_NOTE = (
    "[추정·참고용] 본 산출물은 공개 통계(KDCA/KOSIS/CHS/KNHANES/병무청)에 기반한 "
    "집단 단위 계리 추정이며 개인 예측이 아니다. 횡단면 조사 특성상 개인 인과 추정에는 "
    "사용할 수 없고, 발생률은 주변분포 기반 비례 보정으로 결합되었다."
)

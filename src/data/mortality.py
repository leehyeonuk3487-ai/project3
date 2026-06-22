"""M0 — 사망원인 104항목 → 4범주 매핑 + 질병/상해/자살 분리.

⚠️ 데이터 상태: 현재 repo의 사망원인 파일은 전체사인('계')만 포함한다(검증됨).
   104 사인분류 재추출본(config.DEATH_CAUSE_BY_CAUSE_CSV)이 드롭인되면 본 모듈이
   자동으로 4범주 발생률을 산출한다. KOSIS는 egress 차단(403)이라 이 환경에서
   직접 재추출은 불가하므로, 아래 재추출 사양과 매핑 규칙을 제공한다.

4범주(작업지시 고정):
  ① 질병사(disease)   : 암(C00–C97) + 순환계(I00–I99) + 기타 질병(A00–R99)
  ② 상해사(injury)    : 외인사(V01–Y89) 중 자살 제외
  ③ 자살(suicide)     : 고의적 자해(X60–X84, Y87.0) — 보장 면책, envelope에만 반영
  ④ 기타(other)       : 위에 속하지 않는 잔여

연결: 질병사망 보장 = ①질병사 직접(비외상 proxy 폐기). 상해사망 = ②상해사 직접.
정직성: 자살은 면책 플래그로 분리해 보장 산출에 미포함, envelope·노트에만 반영.
"""
from __future__ import annotations

import pandas as pd

from .. import config

# ICD-10 코드 접두 → 범주. KOSIS 104항목은 코드범위를 동반하므로 코드 우선 매핑.
ICD_RULES = [
    # (범주, 설명, 코드 판정 함수)
    ("suicide", "고의적 자해 X60–X84, Y87.0", lambda c: _in_letter_range(c, "X", 60, 84) or c.startswith("Y870")),
    ("injury", "외인사 V01–Y89 (자살 제외)", lambda c: c[:1] in ("V", "W") or _in_letter_range(c, "X", 0, 59) or _in_letter_range(c, "Y", 0, 89)),
    ("disease", "암 C·순환계 I·기타 질병 A00–R99", lambda c: "A" <= c[:1] <= "R"),
]

# 코드가 없는(한글 사인명만) 추출본 대비 키워드 보조 매핑
KEYWORD_RULES = [
    ("suicide", ["고의적 자해", "자살"]),
    ("injury", ["운수", "추락", "낙상", "익사", "익수", "화재", "중독", "타살", "가해",
                "손상", "외인", "사고"]),
    ("disease", ["신생물", "악성", "암", "순환", "심장", "뇌혈관", "호흡", "소화",
                 "감염", "내분비", "당뇨", "신경", "질환", "질병"]),
]


def _in_letter_range(code: str, letter: str, lo: int, hi: int) -> bool:
    code = (code or "").strip().upper()
    if not code.startswith(letter):
        return False
    digits = "".join(ch for ch in code[1:3] if ch.isdigit())
    return digits != "" and lo <= int(digits) <= hi


def classify_cause(label_or_code: str) -> str:
    """사인 라벨/코드 → 4범주. 코드 우선, 없으면 키워드."""
    s = str(label_or_code).strip()
    code = s.upper()
    if code[:1].isalpha() and any(ch.isdigit() for ch in code[:4]):
        for cat, _desc, fn in ICD_RULES:
            if fn(code):
                return cat
        return "other"
    for cat, kws in KEYWORD_RULES:
        if any(k in s for k in kws):
            return cat
    return "other"


def mapping_table() -> pd.DataFrame:
    """4범주 매핑 규칙표(출력용 — 임의 단순화 없음)."""
    rows = [
        ("① 질병사 (disease)", "암 C00–C97, 순환계 I00–I99, 기타 질병 A00–R99",
         "질병사망 보장 직접 산출(비외상 proxy 폐기)"),
        ("② 상해사 (injury)", "외인사 V01–Y89 중 자살(X60–X84) 제외",
         "상해사망 보장 직접 산출"),
        ("③ 자살 (suicide)", "고의적 자해 X60–X84, Y87.0",
         "면책 — 보장 미포함, envelope·정직성 노트에만"),
        ("④ 기타 (other)", "위 범주 외 잔여(R/U 등 불명 포함)", "envelope 정합용"),
    ]
    return pd.DataFrame(rows, columns=["범주", "ICD/사인 범위", "연결·용도"])


def available() -> bool:
    """104 사인분류 재추출 파일이 드롭인되어 있는가."""
    return config.DEATH_CAUSE_BY_CAUSE_CSV.exists()


def load_mortality_by_cause() -> pd.DataFrame | None:
    """104 사인분류 사망률을 4범주로 집계한 long 프레임.

    재추출 파일이 없으면 None을 반환(상위에서 전체사인 envelope로 폴백).
    스키마 가정: 첫 컬럼=사인(코드 또는 명), 그다음 시도/성/연령, 이후 연도열.
    """
    if not available():
        return None
    from .loaders import _melt_year_columns, normalize_age5
    raw = pd.read_csv(config.DEATH_CAUSE_BY_CAUSE_CSV,
                      encoding=_sniff(config.DEATH_CAUSE_BY_CAUSE_CSV))
    raw.columns = [str(c).strip() for c in raw.columns]
    id_cols = list(raw.columns[:4])  # 사인, 시도, 성, 연령 가정
    long = _melt_year_columns(raw, id_cols)
    long = long.rename(columns={id_cols[0]: "cause", id_cols[1]: "sido_raw",
                                id_cols[2]: "sex_name", id_cols[3]: "age5"})
    long["category"] = long["cause"].map(classify_cause)
    long["sido"] = long["sido_raw"].map(config.KOSIS_SIDO_ALIASES).fillna(long["sido_raw"])
    long["age5"] = long["age5"].map(normalize_age5)
    return long


def _sniff(path):
    from .encoding import detect_encoding
    return detect_encoding(path)


def status() -> dict:
    """M0 데이터·재추출 상태 요약."""
    return {
        "by_cause_available": available(),
        "current_file": "전체사인('계')만 — 질병/상해/자살 분리 불가",
        "blocker": "KOSIS egress 차단(403) → 이 환경에서 재추출 불가",
        "needed_csv": str(config.DEATH_CAUSE_BY_CAUSE_CSV.relative_to(config.ROOT)),
        "reextract_spec": (
            "KOSIS 사망원인통계(통계표: 사망원인(104항목)/시도/성/5세연령) → "
            "지표 '사망률(십만명당)' + '사망자수', 연도 2020~2024, 전체 시도·성·연령 "
            "다운로드. 사인 코드(ICD) 열 포함 권장(키워드 매핑 의존도 ↓)."
        ),
    }

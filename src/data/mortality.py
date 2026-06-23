"""M0 — 사망원인 분류 → 4범주(질병/상해/자살/기타) 분리 + 직접 사망률 산출.

데이터: data/raw/kosis_death_cause/death_cause_by_region_sex_age5_detail.csv
   (시도×사망원인×연령(5세)×성×연도, 사망자수+사망률 십만명당, 1996–2024, 남자 20대).
   각 사인 라벨에 ICD-10 코드범위가 동반되므로 **코드 기반 결정적 매핑**을 쓴다(추측 없음).

4범주(작업지시 고정 — ICD-10 챕터 규칙):
  ① disease  : A00–Q99 질병사인(감염·신생물·순환·내분비·신경·호흡·소화 등)
  ② external : 외인사 V01–Y89 중 자살 제외(운수·추락·익사·화재·중독·가해·기타외인)
  ③ suicide  : 고의적 자해 X60–X84 — 보장 면책, envelope·정직성 노트에만 반영
  ④ other    : 증상·징후 불명확 R00–R99 (한 범주에만 귀속)

★중복합산 방지: 본 추출본의 사인은 상호배타 leaf(검증: 22사인 합 = 전체사망('계') ±0.01%).
   '계'·상위 집계행 없음. 항목 '모든 기타 외인'은 잔차(외인 총합 아님 — 수치 검증됨).

연결: 질병사망 보장 = ①disease 직접(★기존 '비외상 중증손상 proxy' 폐기).
      상해사망 보장 = ②external 직접(기존 중증외상×치명률 협의 추정 대체).
      ③suicide는 suicide_excluded 플래그 → 예산·보장 미포함.

율 산출: 시도×20대남성×사인 셀은 소표본이라
  · 2021–2024 사망자수 합산 ÷ 동기간 인구추계 person-years 합산 × 100,000 (다년 안정화).
  · 잔여 소셀 변동은 Marshall 전역 경험적 베이즈(Poisson–Gamma)로 전국 사인율로 shrink.
재현성: 결정적 매핑·집계라 난수 미사용.
"""
from __future__ import annotations

import re

import pandas as pd

from .. import config

# 분석 대상(현역 청년 모집단): 남자 20-24·25-29세, 최근 4년 풀링.
POOL_YEARS = [2021, 2022, 2023, 2024]
AGE5_20S = ["20-24세", "25-29세"]

_CODE_RE = re.compile(r"([A-Z])(\d{2})")


def classify_code(letter: str, num: int) -> str:
    """ICD-10 선두코드(글자+2자리) → 4범주. 결정적 규칙(챕터 기반)."""
    if letter == "X" and 60 <= num <= 84:
        return "suicide"               # 고의적 자해
    if letter in ("V", "W", "Y") or letter == "X":
        return "external"              # 외인사(자살 제외 — X60–84는 위에서 분기)
    if letter == "R":
        return "other"                 # 증상·징후 불명확(달리 분류되지 않은)
    if "A" <= letter <= "Q":
        return "disease"               # 질병사인 챕터 A00–Q99
    return "other"                     # U 단독 등 — 본 추출본에선 발생 안 함


# 코드가 없는(한글명만) 추출본 대비 키워드 보조 매핑(현 데이터는 코드 동반이라 미사용 경로)
_KEYWORD_RULES = [
    ("suicide", ["고의적 자해", "자살"]),
    ("external", ["운수", "추락", "낙상", "익사", "익수", "화재", "불꽃", "중독",
                  "타살", "가해", "외인", "사고"]),
    ("disease", ["신생물", "악성", "암", "순환", "심장", "뇌혈관", "호흡", "소화",
                 "감염", "기생충", "내분비", "대사", "당뇨", "신경", "정신",
                 "혈액", "면역", "피부", "근골격", "비뇨", "생식", "선천", "질환"]),
]


def classify_cause(label: str) -> str:
    """사인 라벨 → 4범주. ICD 선두코드 우선, 없으면 키워드 보조."""
    s = str(label)
    m = _CODE_RE.search(s)
    if m:
        return classify_code(m.group(1), int(m.group(2)))
    for cat, kws in _KEYWORD_RULES:
        if any(k in s for k in kws):
            return cat
    return "other"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "").str.strip().replace({"-": "0", "": "0"}),
        errors="coerce",
    ).fillna(0.0)


def available() -> bool:
    """사인분류 사망 데이터(detail)가 존재하는가."""
    return config.DEATH_CAUSE_BY_CAUSE_CSV.exists()


def load_mortality_by_cause() -> pd.DataFrame:
    """detail 파일 → long[cause, category, sido, age5, sex_name, year, deaths, rate_100k].

    남자·20대·POOL_YEARS로 한정. 사인은 ICD 코드 기반 4범주로 분류.
    """
    # KOSIS 사망원인표 계열은 cp949 고정(load_death_cause와 동일).
    raw = pd.read_csv(config.DEATH_CAUSE_BY_CAUSE_CSV, encoding="cp949", dtype=str)
    raw.columns = [str(c).strip() for c in raw.columns]
    cause_col, sido_col, age_col, sex_col, yr_col = raw.columns[:5]
    death_col = next(c for c in raw.columns if "사망자수" in c)
    rate_col = next(c for c in raw.columns if "사망률" in c)

    df = pd.DataFrame({
        "cause": raw[cause_col].astype(str).str.strip(),
        "sido": raw[sido_col].map(config.KOSIS_SIDO_ALIASES).fillna(raw[sido_col]),
        "age5": raw[age_col].astype(str).str.replace(" ", ""),
        "sex_name": raw[sex_col].astype(str).str.strip(),
        "year": raw[yr_col].astype(str).str.extract(r"(\d{4})")[0].astype(int),
        "deaths": _num(raw[death_col]),
        "rate_100k": pd.to_numeric(raw[rate_col].astype(str).str.strip(), errors="coerce"),
    })
    df["category"] = df["cause"].map(classify_cause)
    return df[(df["sex_name"] == "남자") & df["age5"].isin(AGE5_20S)
              & df["year"].isin(POOL_YEARS)].reset_index(drop=True)


def mapping_table() -> pd.DataFrame:
    """22개 실제 사인 라벨 → 4범주 매핑표(데이터에서 산출, 임의 단순화 없음)."""
    df = load_mortality_by_cause()
    g = (df.groupby(["category", "cause"])["deaths"].sum()
         .reset_index().sort_values(["category", "deaths"], ascending=[True, False]))
    g["deaths"] = g["deaths"].astype(int)
    return g.rename(columns={"category": "범주", "cause": "사인(ICD)", "deaths": "사망_4년"})


def four_category_summary() -> pd.DataFrame:
    """4범주 전국 분해(사망자수·점유율) + 트랙·면책 라벨."""
    df = load_mortality_by_cause()
    g = df.groupby("category")["deaths"].sum()
    tot = g.sum()
    role = {"disease": "질병사망 트랙(직접)", "external": "상해사망 트랙(자살제외, 직접)",
            "suicide": "suicide_excluded(면책)", "other": "불명확(R) — envelope만"}
    rows = [{"범주": k, "사망_4년": int(g.get(k, 0)),
             "점유율%": round(g.get(k, 0) / tot * 100, 1), "연결": role[k]}
            for k in ["disease", "external", "suicide", "other"]]
    return pd.DataFrame(rows)


def _population_py() -> pd.DataFrame:
    """20대 남성 시도×age5 person-years(POOL_YEARS 합)."""
    from .loaders import load_population_projection
    pop = load_population_projection()
    sub = pop[(pop["sex_name"] == "남자") & pop["age5"].isin(AGE5_20S)
              & pop["year"].isin(POOL_YEARS) & (pop["sido"] != "전국")]
    return sub.groupby(["sido", "age5"])["value"].sum().rename("py").reset_index()


def _marshall_eb(deaths: pd.Series, py: pd.Series) -> pd.Series:
    """Marshall 전역 경험적 베이즈(Poisson–Gamma) — 소셀 율을 전국율로 shrink.

    r_i=deaths_i/py_i, m=Σd/Σpy, φ=Σpy(r-m)²/Σpy, s²=φ−m/mean(py).
    C_i=s²/(s²+m/py_i), θ_i=m+C_i(r_i−m). s²≤0이면 전부 m으로 수렴.
    반환은 per-person 율(θ_i).
    """
    d = deaths.to_numpy(dtype=float)
    n = py.to_numpy(dtype=float)
    m = d.sum() / n.sum() if n.sum() > 0 else 0.0
    r = d / n
    phi = (n * (r - m) ** 2).sum() / n.sum() if n.sum() > 0 else 0.0
    s2 = max(phi - m / n.mean(), 0.0) if n.mean() > 0 else 0.0
    denom = s2 + m / n
    c = pd.Series(0.0, index=deaths.index)
    nz = denom > 0
    c[nz] = (s2 / denom)[nz]
    theta = m + c.to_numpy() * (r - m)
    return pd.Series(theta, index=deaths.index)


def mortality_rate_surface(category: str, eb_shrink: bool = True) -> pd.DataFrame:
    """범주별 20대 남성 시도×age5 직접 사망률 surface(per-100k, 4년 풀링).

    deaths(4년 합) ÷ person-years(4년 합) × 100k. eb_shrink=True면 age5 층 내
    17개 시도에 Marshall 경험적 베이즈로 안정화. 출력 컬럼은 rates 파이프라인 호환.
    """
    df = load_mortality_by_cause()
    cat = df[df["category"] == category]
    deaths = (cat.groupby(["sido", "age5"])["deaths"].sum()
              .reindex(pd.MultiIndex.from_product(
                  [sorted(df["sido"].unique()), AGE5_20S], names=["sido", "age5"]),
                  fill_value=0.0).rename("deaths").reset_index())
    py = _population_py()
    surf = deaths.merge(py, on=["sido", "age5"], how="left")
    surf = surf[surf["py"] > 0].copy()
    surf["rate_raw_100k"] = surf["deaths"] / surf["py"] * 1e5
    if eb_shrink:
        parts = []
        for age5, grp in surf.groupby("age5"):
            grp = grp.copy()
            grp["rate_eb"] = _marshall_eb(grp["deaths"], grp["py"]) * 1e5
            parts.append(grp)
        surf = pd.concat(parts, ignore_index=True)
    else:
        surf["rate_eb"] = surf["rate_raw_100k"]
    surf["rate_per_100k"] = surf["rate_eb"]
    surf["rate_per_1000py"] = surf["rate_per_100k"] / 100.0
    surf["sex"] = "남자"
    surf["year"] = max(POOL_YEARS)
    surf["source"] = f"kosis_death_cause_m0_{category}"
    return surf[["sido", "sex", "age5", "deaths", "py",
                 "rate_raw_100k", "rate_per_100k", "rate_per_1000py", "year", "source"]]


def envelope_check() -> dict:
    """검증: 4범주 사망자수 합 ≈ 전체사망('계'). raw 율 합 = 전체율(파티션) 확인."""
    df = load_mortality_by_cause()
    cause_sum = int(df["deaths"].sum())
    # 전체사망('계') — 기존 all-cause 추출본에서 동일 필터로.
    from .loaders import load_death_cause
    dc = load_death_cause()
    allc = dc[(dc["sex_name"] == "남자") & dc["age5"].isin(AGE5_20S)
              & dc["year"].isin(POOL_YEARS) & (dc["sido"] != "전국")]["deaths"].sum()
    allc = int(allc) if pd.notna(allc) else None
    return {
        "category_sum_deaths": cause_sum,
        "all_cause_deaths": allc,
        "diff": (cause_sum - allc) if allc is not None else None,
        "rel_error_pct": round(abs(cause_sum - allc) / allc * 100, 3) if allc else None,
    }


def status() -> dict:
    """M0 데이터·산출 상태 요약."""
    s = {"by_cause_available": available(),
         "file": str(config.DEATH_CAUSE_BY_CAUSE_CSV.relative_to(config.ROOT))}
    if available():
        s.update(envelope_check())
    return s

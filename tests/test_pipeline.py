"""핵심 파이프라인 smoke 테스트 (실데이터 기반).

pytest 없이도 `python -m tests.test_pipeline` 로 실행 가능.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.data import aggregate, benefits, loaders, mma_api
from src.models import calibration, ecological, population, rates, stratify
from src.optimize import budget
from src.validation import report


# --- 데이터 로더 ---------------------------------------------------------

def test_loaders_shapes():
    chs = loaders.load_chs()
    assert len(chs) > 200_000
    assert chs["sido"].nunique() == 17
    assert chs["bmi"].between(10, 60).mean() > 0.95


def test_population_projection_and_death_cause():
    pop = loaders.load_population_projection()
    m = pop[(pop.sex_name == "남자") & (pop.age5.isin(["20-24세", "25-29세"]))
            & (pop.sido == "전국")]
    p2024 = m[m.year == 2024]["value"].sum()
    p2035 = m[m.year == 2035]["value"].sum()
    assert p2024 > p2035 > 0           # 인구절벽: 감소
    dc = loaders.load_death_cause()
    assert (dc["death_rate_100k"].dropna() > 0).all()


def test_mma_loaders():
    sr = loaders.load_mma_service_rate()
    assert 0.7 < sr["active_rate"] < 0.95   # 현역판정 ~86%
    bmi = loaders.load_mma_bmi_distribution()
    assert abs(sum(bmi["dist"].values()) - 1.0) < 0.05


# --- 발생률 surface + 트랙 분해 ------------------------------------------

def test_rate_surface_positive_and_decomposed():
    table = rates.conscript_rate_table()
    assert (table.fillna(0) >= 0).all().all()
    assert (table["상해사망"] < table["중증외상 발생"]).all()
    # 질병 트랙 존재
    assert "질병사망" in table.columns and "질병후유장해" in table.columns


def test_region_disparity_meaningful():
    table = rates.conscript_rate_table()
    s = table["골절"].dropna()
    assert s.max() / s.min() > 2.0


def test_disability_payout_ratio_corrects_down():
    r = rates.severe_disability_payout_ratio("trauma")
    assert 0.3 < r < 1.0    # 정액 대비 하향 보정


# --- 모집단 트랙 ---------------------------------------------------------

def test_conscript_population_matches_reported():
    # 경기도 보고 현역 규모 6.8만~10.5만 범위
    n = population.conscript_stock("경기", 2024)
    assert 60_000 < n < 120_000


# --- 생태회귀 + 보정 -----------------------------------------------------

def test_ecological_fit_and_calibration():
    efit = ecological.fit()
    assert efit.n_obs == 17 and (efit.irr > 0).all()
    cf = calibration.calibration_factors()
    assert cf["calibration"].between(0.5, 2.0).all()
    # 보정 적용 fit도 동작
    assert ecological.fit(calibrate=True).n_obs == 17


# --- 층화 ----------------------------------------------------------------

def test_stratify_irr_increases():
    st = stratify.stratify()
    rt = st["profile"]["rate_severe_trauma"]
    assert rt.iloc[0] < rt.iloc[-1] and st["irr"] > 1.0


# --- 예산 (자동 N · 질병항목 · 투영) -------------------------------------

def test_budget_auto_population_all_schedules():
    for name in benefits.list_schedules():
        est = budget.expected_claims(name)
        assert est.population > 0
        assert 1_000 < est.per_capita_claims < 200_000
        # 질병 트랙 항목이 청구액에 포함
        assert (est.by_item["track"] == "질병").any()


def test_budget_respects_constraint():
    opt = budget.optimize_under_budget("경기도", annual_budget=98_000 * 15_000)
    assert opt["estimate"].premium_per_capita <= 15_000 + 1
    for v in opt["coverage_scale"].values():
        assert 0.0 <= v <= 1.0


def test_budget_projection_declines():
    proj = budget.project_budget("경기도", [2024, 2030])
    assert proj.set_index("year").loc[2030, "population"] < \
        proj.set_index("year").loc[2024, "population"]


# --- 검증 (LORO + 예측구간 + 정합성) -------------------------------------

def test_loro_calibration_and_pi():
    res = report.loro_calibration()
    assert len(res["predictions"]) == 17
    m = res["metrics"]
    assert m["model"]["pearson_r"] > 0.3
    assert m["mae_improvement"] > 0          # 베이스라인 대비 개선
    assert m["pi_coverage"] >= 0.8           # 예측구간 적중률


def test_disease_track_consistency():
    con = report.disease_track_consistency()
    assert con["all_death_envelope_ok"]
    assert con["disease_death_le_incidence_ok"]


# --- API + 외부 의존성 ---------------------------------------------------

def test_api_endpoints():
    from fastapi.testclient import TestClient
    from src.api.main import app
    c = TestClient(app)
    for ep in ["/api/meta", "/api/rates", "/api/ecological", "/api/stratify",
               "/api/validation", "/api/schedules", "/api/calibration",
               "/api/consistency", "/api/population?schedule=경기도"]:
        assert c.get(ep).status_code == 200
    r = c.post("/api/budget", json={"schedule": "경기도"})
    assert r.status_code == 200 and r.json()["population"] > 0


def test_mma_api_client_graceful():
    # egress 차단 환경에서도 오류 없이 상태/폴백을 반환해야 한다
    st = mma_api.status()
    assert "has_key" in st
    r = mma_api.fetch("getRecruitPhysicalList", {"numOfRows": 1})
    assert "ok" in r and r["source"] in ("live", "cache", "error")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

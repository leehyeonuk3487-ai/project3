"""핵심 파이프라인 smoke 테스트 (실데이터 기반).

pytest 없이도 `python -m tests.test_pipeline` 로 실행 가능.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.data import aggregate, benefits, loaders, mma_api, mortality
from src.models import calibration, ecological, panel, population, rates, stratify
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
    # 상해사망·질병사망(M0 직접)은 전체사망 이내 (부분집합 불변식).
    assert (table["상해사망"] < table["전체사망(검증)"]).all()
    assert (table["질병사망"] < table["전체사망(검증)"]).all()
    # 질병 트랙 + 자살(면책) 분리 존재
    assert "질병사망" in table.columns and "질병후유장해" in table.columns
    assert "자살사망(면책)" in table.columns


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
    assert con["coverage_le_all_death_ok"]
    assert con["disease_death_le_incidence_ok"]
    # M0 4범주 partition이 전체사망과 ±1% 이내
    assert con["m0_envelope"]["rel_error_pct"] < 1.0


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


def test_m1_bmi_wired_and_knhanes_features():
    # M1: bmi_distribution()이 코호트 프로파일에 실제 연결(dead code 0)
    prof = population.cohort_profile("경기", 2024)
    assert abs(sum(prof["bmi_distribution"].values()) - 1.0) < 0.05
    assert prof["conscripts"] > 0 and prof["risk_prevalence"]["smoker"] > 0
    kf = loaders.load_knhanes_features()
    assert {"bmi", "smoker", "hypertension", "diabetes"}.issubset(kf.columns)
    # 손상 결과변수는 부재해야 정상(M2 라벨 부재 근거)
    assert not any("inj" in c.lower() or "손상" in c for c in kf.columns)


def test_m0_cause_mapping_and_separation():
    # M0: ICD 코드 결정적 매핑 (추측 없음)
    assert mortality.classify_cause("고의적 자해(자살) (X60-X84)") == "suicide"
    assert mortality.classify_cause("신생물 (C00-D48)") == "disease"
    assert mortality.classify_cause("운수사고 (V01-V99)") == "external"
    assert mortality.classify_cause("가해(타살) (X85-Y09)") == "external"   # X85>84 → 자살 아님
    assert mortality.classify_cause("달리 분류되지 않은 증상, 징후 (R00-R99)") == "other"
    # 자살은 external과 분리 (★자살을 외인에 합치지 않음)
    assert mortality.classify_cause("X70") == "suicide"
    assert mortality.classify_cause("X45") == "external"
    # 데이터 적재 + 4범주 분리
    assert mortality.available() is True
    df = mortality.load_mortality_by_cause()
    assert set(df["category"].unique()) == {"disease", "external", "suicide", "other"}


def test_m0_envelope_mece_and_eb():
    # MECE: 4범주 사망자수 합 = 전체사망('계') ±오차
    env = mortality.envelope_check()
    assert env["rel_error_pct"] < 0.1            # 0.015% 수준
    # 자살이 20대 남성 최다 사인 (면책 분리 대상)
    summ = mortality.four_category_summary().set_index("범주")
    assert summ.loc["suicide", "사망_4년"] > summ.loc["disease", "사망_4년"]
    # 경험적 베이즈가 소셀 변동을 전국율 쪽으로 축소 (결정적 → 재현성)
    raw = mortality.mortality_rate_surface("external", eb_shrink=False)
    eb = mortality.mortality_rate_surface("external", eb_shrink=True)
    j = raw.merge(eb, on=["sido", "age5"], suffixes=("_raw", "_eb"))
    nat = (raw["deaths"].sum() / raw["py"].sum()) * 1e5
    spread_raw = (j["rate_per_100k_raw"] - nat).abs().mean()
    spread_eb = (j["rate_per_100k_eb"] - nat).abs().mean()
    assert spread_eb < spread_raw               # 전국율 쪽으로 수축
    # 결정적(난수 미사용) → 두 번 호출 동일
    eb2 = mortality.mortality_rate_surface("external", eb_shrink=True)
    assert eb["rate_per_100k"].round(9).tolist() == eb2["rate_per_100k"].round(9).tolist()


def test_m3_panel_gbm_vs_baseline():
    # M3: CV 성능 보고 + 채택규칙. 시드 고정 재현성.
    r1 = panel.evaluate()
    r2 = panel.evaluate()
    assert r1.temporal["gbm_rmse"] == r2.temporal["gbm_rmse"]   # 재현성
    assert r1.n_obs > 200
    # 두 모델 모두 RMSE 보고
    assert r1.temporal["baseline_rmse"] > 0 and r1.temporal["gbm_rmse"] > 0
    if r1.adopt_gbm:   # 채택 시 surface 예측 동작
        s = panel.predict_surface(2023)
        assert (s["rate_per_100k"] > 0).all()


# --- M2: 개인 손상위험 ML --------------------------------------------------

def test_m2_label_consistency_and_pooling():
    from src.models import injury_ml
    feat = injury_ml.load_features()
    # 라벨 이진, 유효표본·유병률(작업지시 수치와 일치)
    assert set(feat["y"].unique()) == {0, 1}
    assert len(feat) == 5730 and int(feat["y"].sum()) == 386
    # 전 연령·전성별 풀링(20대만 학습 금지) — 학습표본이 20대남보다 크다
    m20 = feat[(feat["male"] == 1) & feat["age"].between(20, 29)]
    assert len(m20) == 1086 and len(feat) > len(m20)
    # 고정 예측변수 집합(누수 변수 미포함)
    assert injury_ml.FEATURES == ["age", "male", "HE_BMI", "sm_presnt",
                                  "dr_month", "pa_aerobic", "htn_dx", "dm_dx"]


def test_m2_auc_baseline_and_reproducible():
    from src.models import injury_ml
    r1 = injury_ml.evaluate()
    r2 = injury_ml.evaluate()
    # 재현성(시드 고정)
    assert r1.holdout["logit"]["auc"] == r2.holdout["logit"]["auc"]
    assert r1.kfold["lgbm"]["auc_mean"] == r2.kfold["lgbm"]["auc_mean"]
    # 베이스라인·두 모델 모두 AUC 보고(0.4~0.8 합리 범위, modest 허용)
    for m in ("baseline", "logit", "lgbm"):
        assert 0.4 < r1.kfold[m]["auc_mean"] < 0.85
    # 비가중 캘리브레이션: 예측평균이 관측 유병률에 근사(절대확률 교정)
    h = r1.holdout["logit"]
    assert abs(h["pred_mean_cal"] - h["obs_rate"]) < 0.03
    # 집단 요약만(개인 점수 키 부재)
    assert set(r1.group_risk) >= {"n", "observed_rate", "pred_mean"}
    assert "individual_scores" not in r1.group_risk


# --- 모듈 A: 셀 패널 발생률 GBM (공간·시간 CV) ----------------------------

def test_module_a_panel_and_spatial_cv():
    from src.models import cell_panel
    # 패널 구성: 시도 차원 포함, 작업지시 확정 셀수
    pa = cell_panel.build_panel("allcause")
    assert pa["sido"].nunique() == 17 and len(pa) == 2890
    assert {"deaths", "py"}.issubset(pa.columns) and (pa["py"] > 0).all()
    pe = cell_panel.build_panel("external")          # 2005–2024 확장 (주민등록+추계 분모)
    assert len(pe) == 666 and set(pe["sex"].unique()) == {"남자"}
    assert pe["year"].min() == 2005 and pe["year"].max() == 2024
    # 세종 2005–2011은 분모 부재로 제외(2012 신설)
    assert pe[pe["sido"] == "세종"]["year"].min() == 2012
    # GBM 예측이 합리적 발생률(폭주 없음): 전체사인 학습 적합
    booster = cell_panel._fit_gbm(pa)
    pred = cell_panel._pred_gbm(booster, pa) * 1e5      # per-100k
    obs = (pa["deaths"] / pa["py"]).values * 1e5
    assert pred.max() < obs.max() * 3                   # offset 정상(과거 100000/100k 버그 방지)


def test_module_a_cv_baselines_and_adopt():
    from src.models import cell_panel
    r = cell_panel.evaluate("allcause")
    # 공간 CV는 시도 수 만큼 fold → 세 모델 모두 지표 보고
    for cv in (r.spatial, r.temporal):
        for m in ("gbm", "proportional", "simple_mean"):
            assert cv[m]["deviance"] > 0 and "calib_slope" in cv[m]
    # 채택은 boolean(공간·시간 모두 deviance↓ + 캘리브레이션 안정일 때만)
    assert isinstance(r.adopt_gbm, bool)
    # 재현성(시드 고정)
    assert cell_panel.evaluate("allcause").spatial["gbm"]["deviance"] == \
        r.spatial["gbm"]["deviance"]
    # 피처에 지역 행태(CHS) 포함 — 공간 일반화 신호
    assert "binge_drink" in set(r.importance["feature"])


def test_module_a_external_extended_temporal_stable():
    # 외인 패널 확장(666셀)으로 시간CV 캘리브레이션이 안정([0.5,2.0])되는지 — 확장의 핵심 목적.
    from src.models import cell_panel
    r = cell_panel.evaluate("external")              # 2005–2024
    assert r.n_cells == 666 and r.years == (2005, 2024)
    # 이전(136셀)은 시간CV calib 7.5로 불안정했음 → 확장 후 안정 범위 확인
    assert 0.5 <= r.temporal["gbm"]["calib_slope"] <= 2.0
    # 소스전환연도(2020) 제외 민감도도 안정 → 경계 불연속이 검증을 오염시키지 않음
    assert r.temporal_ex_switch is not None
    assert 0.5 <= r.temporal_ex_switch["gbm"]["calib_slope"] <= 2.0
    # 공간 CV: GBM이 두 베이스라인 deviance를 이김(지역 신호 유지)
    assert r.spatial["gbm"]["deviance"] < r.spatial["proportional"]["deviance"]
    assert r.spatial["gbm"]["deviance"] < r.spatial["simple_mean"]["deviance"]
    # 검증본 surface는 17개 시도 양수 발생률
    surf = cell_panel.external_gbm_surface(2024)
    assert surf["sido"].nunique() == 17 and (surf["rate_per_100k"] > 0).all()


# --- M4: 코호트 리스크 통합·보정 ------------------------------------------

def test_m4_cohort_risk_index():
    from src.models import cohort
    idx = cohort.cohort_risk_index()
    assert len(idx) == 17 and (idx["risk_index"] > 0).all()
    # 전국 인구가중 평균 = 1.0 (정규화; risk_index 4자리 반올림 오차 허용)
    w = cohort._conscript_weights().reindex(idx.index)
    assert abs(np.average(idx["risk_index"], weights=w) - 1.0) < 1e-3
    # 결정적(재현성)
    assert cohort.cohort_risk_index()["risk_index"].round(6).tolist() == \
        idx["risk_index"].round(6).tolist()


def test_m4_bmi_selection_and_claims():
    from src.models import cohort
    from src.data import benefits
    sel = cohort.conscript_bmi_selection()
    # 병무청 BMI≥25 = 과체중+비만, CHS obese≥25 — 둘 다 [0,1], 정의 호환
    assert 0 < sel["conscript_bmi_ge25"] < 1 and 0 < sel["general_bmi_ge25"] < 1
    # 통합 청구액: 검증 surface 기준(이중계산 방지), 지수 병기
    c = cohort.cohort_adjusted_claims(benefits.list_schedules()[0])
    assert c["per_capita_claims"] > 0 and c["cohort_risk_index"] > 0
    assert c["apply_index"] is False                 # 기본은 이중계산 없음


# --- M5: 계리 보험료 산정 -------------------------------------------------

def test_m5_actuarial_premium_structure():
    from src.optimize import premium
    pr = premium.actuarial_premium("경기도")
    # 순 < 위험할증포함 < 총(사업비), 모두 양수
    assert 0 < pr.net_pc < pr.net_pc + pr.risk_margin_pc <= pr.gross_pc
    assert 0 < pr.cv < 1 and pr.implied_loading > 1.0
    # 결정적
    assert premium.actuarial_premium("경기도").gross_pc == pr.gross_pc


def test_m5_small_pool_higher_loading():
    from src.optimize import premium
    big = premium.actuarial_premium("경기도")          # N~98k
    small = premium.actuarial_premium("강원_화천군")    # N~12k
    # 소형 풀일수록 변동계수↑ → 1인당 위험할증↑ (신뢰도/풀링 효과)
    assert small.cv > big.cv
    assert small.risk_margin_pc > big.risk_margin_pc


def test_mma_api_client_graceful():
    # egress 차단 환경에서도 오류 없이 상태/폴백을 반환해야 한다
    st = mma_api.status()
    assert "has_key" in st
    r = mma_api.fetch("getRecruitPhysicalList", {"numOfRows": 1})
    assert "ok" in r and r["source"] in ("live", "cache", "error")


# --- 혜택 최대화 LP (M6 선행 최적화 레이어) ------------------------------

def test_benefit_lp_optimizes_respects_budget_and_floor():
    full = budget.expected_claims("경기도").total_claims * budget.DEFAULT_LOADING
    lp = budget.optimize_benefit_lp("경기도", full * 0.8)
    assert lp["feasible"]
    # 예산 준수(±1원 반올림)
    assert lp["estimate"].total_claims * budget.DEFAULT_LOADING <= full * 0.8 + 1
    # 고위험(생명·장해) floor=1.0 → 배제 금지
    for it in budget.CATASTROPHIC_ITEMS:
        assert lp["coverage_scale"][it] >= 1.0 - 1e-6
    # 재현성(시드/입력 동일)
    assert budget.optimize_benefit_lp("경기도", full * 0.8)["coverage_scale"] \
        == lp["coverage_scale"]


def test_benefit_lp_floor_prevents_exclusion():
    # floor=0이면 순수 수혜자최대화가 사망보장을 배제 → floor의 필요성 입증(정직)
    full = budget.expected_claims("경기도").total_claims * budget.DEFAULT_LOADING
    free = budget.optimize_benefit_lp("경기도", full * 0.6, catastrophic_floor=0.0)
    assert free["coverage_scale"]["death_injury"] < 0.5      # 고액·저빈도 → 밀려남
    # 같은 예산에서 floor=1.0이면 고위험 완전보장 비용이 초과 → infeasible 정직 보고
    floored = budget.optimize_benefit_lp("경기도", full * 0.6, catastrophic_floor=1.0)
    assert floored["feasible"] is False


def test_lp_at_least_greedy_and_monotone_budget():
    from src.optimize import scenarios
    tab = scenarios.optimization_under_scenarios("경기도")
    feas = tab[tab["LP_feasible"]]
    # LP 수혜자 ≥ 그리디 (정의상 최적; 동률 허용)
    assert (feas["LP_수혜자"] >= feas["그리디_수혜자"] - 1e-6).all()
    assert (feas["LP_추가수혜"] >= -1e-6).all()
    # 예산↑ → 수혜자 단조 비감소
    benef = feas.sort_values("예산액")["LP_수혜자"].tolist()
    assert benef == sorted(benef)


# --- M6: 미래 투영 추세 외삽 + 시나리오·민감도 ----------------------------

def test_projection_trend_band_data_grounded():
    from src.models import projection
    slopes = projection.external_trend_slopes()
    # 장기 감소 > 최근 둔화 > 평탄(보수) — 데이터 기반, 점추정 아님
    assert slopes["optimistic"] < slopes["base"] < 0      # 둘 다 감소
    assert slopes["conservative"] == 0.0                  # 보수=평탄
    band = projection.external_trend_band(2024, [2024, 2027, 2030])
    # base_year=1.0, 미래로 갈수록 낙관 ≤ 기준 ≤ 보수 (배수)
    row = band[band["year"] == 2030].iloc[0]
    assert row["optimistic_mult"] <= row["base_mult"] <= row["conservative_mult"]
    assert abs(band[band["year"] == 2024].iloc[0]["base_mult"] - 1.0) < 1e-9
    assert "외삽" in band.attrs["note"]                   # 한계 노트 존재


def test_population_cliff_declines():
    from src.models import projection
    cliff = projection.population_cliff("전국", 2024, [2024, 2027, 2030])
    # 인구절벽: 현역 모집단 단조 감소, 2030 ≈ 270만(작업지시 수치)
    stocks = cliff.set_index("year")["stock"]
    assert stocks[2030] < stocks[2027] < stocks[2024]
    assert 2.6e6 < stocks[2030] < 2.8e6


def test_tornado_ranks_drivers_and_pricing_invariant():
    from src.optimize import scenarios
    base_pc = scenarios.sensitivity_table("경기도").attrs["base_premium_pc"]
    # 기준 = 현재 M0 백본 보험료(외삽 미주입) — 계리 보험료와 일치
    from src.optimize import premium
    assert abs(base_pc - round(premium.actuarial_premium("경기도").gross_pc)) <= 1
    tor = scenarios.tornado("경기도")
    # 토네이도 swing 내림차순 정렬
    assert tor["swing"].tolist() == sorted(tor["swing"].tolist(), reverse=True)
    # 보장한도가 1인당 보험료 최대 변동요인(선형 ±20%)
    assert tor.iloc[0]["변수"] == "보장 한도(급부)"
    # 인구절벽은 1인당 가격엔 거의 영향 없음(총 규모 변수임을 정직히 드러냄)
    pop_row = tor[tor["변수"].str.contains("인구절벽")].iloc[0]
    assert pop_row["swing%기준"] < 2.0


def test_multiyear_band_monotone_and_population_driven():
    from src.optimize import scenarios
    band = scenarios.multiyear_band("경기도", years=[2024, 2027, 2030])
    # 각 연도 낙관 ≤ 기준 ≤ 보수 (외인추세 밴드)
    for _, r in band.iterrows():
        assert r["optimistic_총예산"] <= r["base_총예산"] <= r["conservative_총예산"]
    # 인구절벽으로 총예산(기준)이 2024 대비 감소
    assert band[band["year"] == 2030].iloc[0]["base_총예산"] < \
        band[band["year"] == 2024].iloc[0]["base_총예산"]


# --- M7: 통합 리포트 (조립기 — 필수 섹션·미션·라이선스 렌더 검증) ----------

def _synthetic_collected():
    """build_report 입력 스키마에 맞는 합성 데이터(개별 수치는 모듈 테스트가 검증)."""
    import pandas as pd
    return {
        "schedule": "경기도",
        "m0_envelope": {"rel_error_pct": 0.015},
        "moduleA": {"adopt_gbm": True, "years": (2005, 2024), "n_cells": 100,
                    "spatial_calib": 1.28, "temporal_calib": 0.76, "note": "ok"},
        "m2": {"n_obs": 1000, "prevalence": 0.1, "kfold_best_auc": 0.565,
               "holdout_baseline_auc": 0.528, "adopt": "baseline", "adopt_auc": 0.528},
        "loro": {"model_mae": 4.0, "baseline_mae": 5.82,
                 "mae_improvement_pct": 31.3, "pi_coverage": 0.94},
        "m5": {"gross_pc": 30668, "cv": 0.114, "loading_implied": 1.392},
        "trend": {"optimistic": -6.86, "base": -3.32, "conservative": 0.0},
        "lp": {"budget": 2162161709, "objective": "welfare_weighted",
               "lp_welfare": 118.5, "greedy_welfare": 118.5, "lp_gain_welfare": 0.0,
               "lp_beneficiaries": 1203.6, "greedy_beneficiaries": 1203.6},
        "weight_sens": pd.DataFrame([
            {"가중셋": "W0 머릿수(균등)", "그리디_가중혜택": 1203.6, "LP_가중혜택": 1203.6,
             "LP우위": 0.0, "비고": "동률(역산 안 함)"},
            {"가중셋": "W3 임상acuity(입원>골절)", "그리디_가중혜택": 1579.0,
             "LP_가중혜택": 2400.8, "LP우위": 821.8, "비고": "LP우위"}]),
        "perturbation": pd.DataFrame([
            {"예산(full대비)": "80%", "그리디_기본_수혜자": 1203.6, "그리디_섭동_수혜자": 1157.3,
             "LP_수혜자(불변)": 1203.6, "LP_추가수혜_vs섭동": 46.3}]),
        "tornado": pd.DataFrame([
            {"변수": "보장 한도(급부)", "낙관(예산↓)": 24534, "기준": 30668,
             "보수(예산↑)": 36802, "swing": 12268, "swing%기준": 40.0, "근거": "x"}]),
        "multiyear": pd.DataFrame([
            {"year": 2024, "population": 98132, "optimistic_총예산": 2.7e9,
             "base_총예산": 2.7e9, "conservative_총예산": 2.7e9}]),
        "cliff": {"y2024": 3364971, "y2030": 2703130, "pct": -19.7},
        "military": {
            "source": "국방부 사망사고 통계 2011–2025 (data.go.kr)",
            "suicide": {"adopted_ratio": 0.443, "recent5": 0.443, "full": 0.397,
                        "excluded_years": [2025],
                        "direction": "군 자살률이 민간의 약 44% → 과대추정 약 2.3배."},
            "external": {"trend_r": 0.881, "level_ratio": 0.318,
                         "overestimate_factor": 3.14,
                         "verdict": "추세 동행, 수준 과대추정."},
            "pricing_impact": "없음 — M0 직접관측 백본 불변.",
        },
        "ai": {
            "cv_deviance_reduction": {"공간CV(leave-1-시도-out)": 37.5,
                                      "시간CV(최근2년 홀드아웃)": 81.1},
            "calib_slope_oos": 0.991, "chs_deviance_reduction_pct": 37.9,
            "pi_coverage": 0.911,
            "efficiency": {"gbm_fit_ms": 18.0, "lp_solver_ms": 1.66,
                           "deterministic": True},
        },
        "ai_calib": pd.DataFrame([
            {"bin": 1, "예측률_per100k": 8.93, "관측률_per100k": 8.18, "n_cells": 134},
            {"bin": 5, "예측률_per100k": 31.49, "관측률_per100k": 31.13, "n_cells": 133}]),
    }


def test_m7_report_renders_all_required_sections():
    from src.validation import integrated_report
    md = integrated_report.build_report(collected=_synthetic_collected())
    # 평가기준 4개 매핑 섹션
    for sec in ["공공데이터 활용", "AI 성능 검증", "독창성", "발전가능성"]:
        assert sec in md
    # 정직성·한계 / 윤리·미션 / 데이터 출처·라이선스
    assert "정직성·한계" in md and "윤리" in md
    assert "미션" in md and "더 많은 장병이 더 많은 혜택" in md
    assert "개인" in md and "스코어링 금지" in md          # 개인 차등 아님
    assert "라이선스" in md and "공공누리" in md
    # ★pricing 불변 + 외삽 미래한정 원칙이 리포트에 박혀 있어야 함
    assert "M0 직접관측 백본" in md
    assert "구조변화" in md and "미반영" in md             # 외삽 한계 노트
    # 2층 AI 검증 + 베이스라인 정직 비교
    assert "baseline 채택" in md and "0.565" in md
    # ★AI 성능 아티팩트(A+B): CV deviance 감소·OOS 캘리브레이션·ablation·예측구간·효율성
    assert "deviance" in md and "캘리브레이션" in md
    assert "ablation" in md and "예측구간" in md
    assert "0.991" in md and "linprog" in md
    # ★복지가중 목적함수 + 동률 정직 공개 + 섭동 증거 + 역산 안 함
    assert "복지가중" in md and "머릿수" in md
    assert "동률" in md and "역산" in md            # 조작 차단 투명성
    assert "섭동" in md                             # Option5 증거
    # ★군 코호트 proxy 검증(국방부) 섹션
    assert "군 코호트 proxy 타당성 검증" in md
    assert "국방부" in md and "면책" in md
    assert "pricing" in md.lower() or "M0 직접관측" in md   # 미주입 명시


def test_m7_collect_keys_and_pricing_invariant():
    """라이브 collect()가 필수 키를 내고, M5 보험료가 M0 백본과 일치(외삽 미주입)."""
    from src.validation import integrated_report
    from src.optimize import premium
    c = integrated_report.collect("경기도")
    for k in ["m0_envelope", "moduleA", "m2", "loro", "m5", "trend",
              "lp", "weight_sens", "perturbation", "tornado", "multiyear",
              "cliff", "military", "ai", "ai_calib"]:
        assert k in c
    # pricing = M0 백본(계리 보험료)와 일치 — 외삽 주입 없음
    assert abs(c["m5"]["gross_pc"] - round(premium.actuarial_premium("경기도").gross_pc)) <= 1


# --- Option1/Option5 보완책 검증 -------------------------------------------

def test_lp_default_objective_is_welfare_not_headcount():
    """LP 기본 목적이 복지가중(중대도)이며, 가중치는 결과 아닌 독립 근거(사망>장해>경상)."""
    full = budget.expected_claims("경기도").total_claims * budget.DEFAULT_LOADING
    lp = budget.optimize_benefit_lp("경기도", full * 0.8)
    assert lp["objective"] == "welfare_weighted"
    w = budget.DEFAULT_WELFARE_WEIGHTS
    # 중대도 순서: 사망 > 후유장해 > 경상(골절·입원). 골절=입원(knife-edge 미악용)
    assert w["death_injury"] > w["disability"] > w["fracture"]
    assert w["fracture"] == w["hospitalization"]


def test_lp_guarantees_welfare_ge_greedy():
    """LP는 '복지가중 혜택'에서 그리디 이상을 보장(머릿수 아님). 현 표는 동률."""
    full = budget.expected_claims("경기도").total_claims * budget.DEFAULT_LOADING
    cmp = budget.compare_allocation("경기도", full * 0.8)
    assert cmp["objective"] == "welfare_weighted"
    assert cmp["lp"]["welfare_benefit"] >= cmp["greedy"]["welfare_benefit"] - 1e-6


def test_weight_sensitivity_not_cherry_picked():
    """가중치 민감도: 원리적 가중(골절≥입원)에선 동률, 입원>골절에서만 LP우위 → 역산 아님."""
    from src.optimize import scenarios
    ws = scenarios.weight_sensitivity("경기도").set_index("가중셋")
    for name in ws.index:
        if "입원>골절" not in name and ws.loc[name, "LP_가중혜택"] is not None:
            assert abs(ws.loc[name, "LP우위"]) < 1.0     # 원리적 가중 → 동률
    w3 = [n for n in ws.index if "입원>골절" in n][0]
    assert ws.loc[w3, "LP우위"] > 1.0                    # 입원>골절에서만 우위


def test_perturbation_lp_robust_greedy_fragile():
    """Option5: 우선순위 가정 섭동 시 LP가 그리디를 추월(표·예산·가중 불변)."""
    from src.optimize import scenarios
    pt = scenarios.perturbation_sensitivity("경기도")
    assert (pt["그리디_섭동_수혜자"] <= pt["그리디_기본_수혜자"] + 1e-6).all()
    assert (pt["LP_추가수혜_vs섭동"] > 0).all()           # LP가 섭동 그리디 추월


# --- 국방부 사망사고 통계: 군 코호트 proxy 검증 ----------------------------

def test_mnd_loader_parses_and_2025_civ_missing():
    df = loaders.load_mnd_death_accidents()
    assert df["year"].min() == 2011 and df["year"].max() == 2025
    # ★2025 민간자살률 결측 → NaN(0/빈값 대체 금지)
    assert df.loc[df.year == 2025, "civ_suicide_rate"].isna().iloc[0]
    assert df.loc[df.year == 2025, "mil_suicide_rate"].notna().iloc[0]   # 군은 존재


def test_mnd_category_mapping_excludes_discipline_other():
    """military_external = 안전사고 7종 합. 군기사고(총기·폭행·기타)는 외인에 미포함."""
    df = loaders.load_mnd_death_accidents()
    r = df[df.year == 2011].iloc[0]
    assert r["military_external"] == r[loaders._MND_SAFETY].sum()
    assert r["military_discipline_other"] == r[loaders._MND_DISC_OTHER].sum()
    # 외인 합에 총기·폭행·기타가 섞이지 않음
    assert "disc_gun" not in loaders._MND_SAFETY


def test_mnd_troops_implied_plausible():
    """병력 역산(자살건수/자살률)이 알려진 ROK 병력대(50만~70만)와 정합."""
    df = loaders.load_mnd_death_accidents()
    assert df["troops_implied"].between(5.0e5, 7.0e5).all()


def test_suicide_adjustment_below_one_excludes_2025():
    from src.validation import military_proxy
    s = military_proxy.suicide_adjustment()
    assert s["adopted"] < 1.0                       # 군 < 민간(과대추정 보정 방향)
    assert 2025 in s["excluded_years"]              # 민간자살률 결측 → 제외
    assert s["applies_to_pricing"] is False         # 면책·pricing 미주입


def test_external_crossval_trend_and_level():
    from src.validation import military_proxy
    e = military_proxy.external_crossvalidation()
    assert e["trend_pearson_r"] > 0.5               # 추세 동행
    assert 0 < e["level_ratio_mil_over_gen"] < 1.0  # 군 수준이 일반보다 낮음(과대추정)
    assert e["applies_to_pricing"] is False


def test_military_validation_does_not_change_pricing():
    """군 보정 모듈 호출 전후 M0 백본 보험료 불변(자동 주입 없음)."""
    from src.validation import military_proxy
    from src.optimize import premium
    before = premium.actuarial_premium("경기도").gross_pc
    _ = military_proxy.summary()                     # 검증 수행
    after = premium.actuarial_premium("경기도").gross_pc
    assert before == after


# --- AI 성능 검증 아티팩트 (A+B) -------------------------------------------

def test_ai_gbm_beats_baselines_on_external_cv():
    """외인 GBM이 공간·시간 CV에서 두 베이스라인 deviance를 모두 이긴다(감소율>0)."""
    from src.validation import ai_performance
    cvd = ai_performance.cv_deviance_comparison("external")
    assert cvd["adopt_gbm"] is True
    for v in cvd["deviance_reduction"].values():
        assert v > 0                                 # 베이스라인 대비 deviance 감소


def test_ai_oos_calibration_near_identity():
    """OOS(leave-1-시도-out) 캘리브레이션 기울기가 1 근처 → 잘 보정됨."""
    from src.validation import ai_performance
    cal = ai_performance.calibration_reliability("external")
    assert 0.8 <= cal["calib_slope_oos"] <= 1.2


def test_ai_chs_ablation_contributes():
    """건강행태(CHS) 특성이 OOS deviance를 줄인다(지역 위험요인 기여 입증)."""
    from src.validation import ai_performance
    abl = ai_performance.chs_ablation("external")
    assert abl["chs_deviance_reduction_pct"] > 0
    assert abl["deviance_full"] < abl["deviance_struct_only"]


def test_ai_prediction_interval_coverage_reasonable():
    from src.validation import ai_performance
    pic = ai_performance.prediction_interval_coverage("external")
    assert 0.8 <= pic["empirical_coverage"] <= 1.0   # 과신/과소신뢰 아님


def test_ai_efficiency_deterministic_and_fast():
    from src.validation import ai_performance
    eff = ai_performance.efficiency_metrics()
    assert eff["deterministic_gbm"] and eff["deterministic_lp"]
    assert eff["lp_solver_ms"] < 100                 # 최적화 코어 빠름


def test_ai_artifacts_do_not_change_pricing():
    from src.validation import ai_performance
    from src.optimize import premium
    before = premium.actuarial_premium("경기도").gross_pc
    _ = ai_performance.scorecard()
    after = premium.actuarial_premium("경기도").gross_pc
    assert before == after


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

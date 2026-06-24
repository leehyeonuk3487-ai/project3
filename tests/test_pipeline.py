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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

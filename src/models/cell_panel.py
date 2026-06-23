"""모듈 A — 셀 패널 발생률 GBM (집단 추정, 개인 예측 아님).

M2(개인 손상 ML, AUC 0.56·baseline 미달)의 정직한 음성결과를 보완하는 '집단(셀)
발생률 학습·검증' 모듈. M3(성×연령×연도)와 달리 **시도 차원을 포함**해 M3가 불가능했던
**leave-one-시도-out 공간 CV**를 수행한다(M0의 시도×성×연령×연도 joint 사망 관측 활용).

대원칙: 기존 통계 백본·비례보정은 유지. 본 모듈은 추가. 베이스라인(비례보정·단순평균)을
   공간·시간 CV로 정직 비교 — GBM이 둘 다에서 이길 때만 surface 교체, 아니면 baseline 유지.
   신규 데이터 없음(기존 사망·인구·CHS 재구성). 시드 고정. 개인 예측 금지(셀 단위만).

타깃(작업지시 확정):
  · allcause : 전체사인 사망률, 시도×성×17연령밴드×5연도(2020–24) = 2,890셀 — 1차(방법론 검증).
  · external : 외인(상해)사망률, 20대남 시도×2밴드×4연도(2021–24) = 136셀 — 보조(도메인 한계).
노출 offset = log(person-years), 인구추계 분모. 피처 = 성·연령·연도 + CHS 시도 행태유병률.
모델 = LightGBM Poisson(offset=log PY), 정규화·얕은 트리. 지표 = Poisson deviance·RMSE(log)·
   MAE(per-100k)·캘리브레이션 기울기.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import aggregate, loaders, mortality

SEED = 42
PANEL_YEARS = [2020, 2021, 2022, 2023, 2024]
AGE_BANDS = ["5-9세", "10-14세", "15-19세", "20-24세", "25-29세", "30-34세", "35-39세",
             "40-44세", "45-49세", "50-54세", "55-59세", "60-64세", "65-69세", "70-74세",
             "75-79세", "80-84세", "85-89세"]
CHS_FEATURES = ["smoker", "binge_drink", "phys_inactive", "obese", "underweight"]
STRUCT_FEATURES = ["age_mid", "male", "year"]
FEATURES = STRUCT_FEATURES + CHS_FEATURES

_LGB_PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=15, max_depth=3,
                   min_child_samples=30, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, seed=SEED, verbosity=-1,
                   deterministic=True, force_row_wise=True, num_threads=1)
_LGB_ROUNDS = 200  # 고정 라운드(테스트 누수 방지 — early stopping 미사용, 정규화로 과적합 억제)


def _age_mid(band: str) -> float:
    nums = [int(x) for x in band.replace("세", "").split("-") if x.strip().isdigit()]
    return float(np.mean(nums)) if nums else np.nan


def _region_chs() -> pd.DataFrame:
    """시도별 CHS 행태 유병률(공간 CV에서 held-out 시도 예측에 쓰이는 지역 피처)."""
    rp = aggregate.region_risk_prevalence()[CHS_FEATURES].reset_index()
    return rp


def build_panel(target: str = "allcause") -> pd.DataFrame:
    """셀 패널[sido, sex, age5, year, deaths, py, + 피처]. 신규 데이터 없음."""
    pop = loaders.load_population_projection()
    pop = pop[(pop["sido"] != "전국") & pop["sex_name"].isin(["남자", "여자"])
              & pop["age5"].isin(AGE_BANDS)].rename(columns={"value": "py", "sex_name": "sex"})

    if target == "allcause":
        dc = loaders.load_death_cause()
        dc = dc[dc["year"].isin(PANEL_YEARS) & dc["sex_name"].isin(["남자", "여자"])
                & (dc["sido"] != "전국") & dc["age5"].isin(AGE_BANDS)]
        deaths = dc.rename(columns={"sex_name": "sex"})[["sido", "sex", "age5", "year", "deaths"]]
    elif target == "external":
        det = mortality.load_mortality_by_cause()           # 남자 20대, 2021–24
        ext = det[det["category"] == "external"].groupby(
            ["sido", "age5", "year"])["deaths"].sum().reset_index()
        ext["sex"] = "남자"
        deaths = ext[["sido", "sex", "age5", "year", "deaths"]]
    else:
        raise ValueError(f"미지원 target: {target}")

    panel = deaths.merge(pop[["sido", "sex", "age5", "year", "py"]],
                         on=["sido", "sex", "age5", "year"], how="inner")
    panel = panel.merge(_region_chs(), on="sido", how="left")
    panel["deaths"] = panel["deaths"].fillna(0.0)
    panel["age_mid"] = panel["age5"].map(_age_mid)
    panel["male"] = (panel["sex"] == "남자").astype(int)
    panel = panel[panel["py"] > 0].reset_index(drop=True)
    return panel


# ---------------------------------------------------------------------------
# 모델 + 베이스라인 (모두 per-person rate 예측)
# ---------------------------------------------------------------------------

def _fit_gbm(tr: pd.DataFrame):
    """가중 Poisson: label=rate, weight=person-years (offset Poisson과 등가).

    init_score offset 방식은 poisson_max_delta_step에 민감해 수렴이 불안정하므로,
    노출가중 rate 회귀(boost_from_average가 기저율을 잡음)로 동등하게 적합한다.
    """
    import lightgbm as lgb
    rate = tr["deaths"].values / tr["py"].values
    dtr = lgb.Dataset(tr[FEATURES], label=rate, weight=tr["py"].values,
                      free_raw_data=False)
    return lgb.train(_LGB_PARAMS, dtr, num_boost_round=_LGB_ROUNDS)


def _pred_gbm(booster, te: pd.DataFrame) -> np.ndarray:
    # predict() = 예측 rate per person (노출가중 Poisson)
    return np.clip(np.asarray(booster.predict(te[FEATURES])), 0.0, None)


def _pred_simple_mean(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """베이스라인②: (성·연령·연도) 그룹 발생률 평균 — 지역정보 없음.

    미관측 (성·연령·연도) 셀(시간CV의 홀드아웃 연도 등)은 (성·연령) → 전국 순으로
    공정하게 폴백한다(연도 카테고리 부재로 인한 불이익 제거).
    """
    def _rate(df, keys):
        return (df.groupby(keys).apply(lambda g: g["deaths"].sum() / g["py"].sum(),
                                       include_groups=False)).to_dict()
    g_full = _rate(tr, ["male", "age5", "year"])
    g_ma = _rate(tr, ["male", "age5"])
    nat = tr["deaths"].sum() / tr["py"].sum()
    out = []
    for m, a, y in zip(te["male"], te["age5"], te["year"]):
        out.append(g_full.get((m, a, y), g_ma.get((m, a), nat)))
    return np.array(out)


def _pred_proportional(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """베이스라인①: 비례보정 — national × R_male × R_age × R_year (주변 독립)."""
    nat = tr["deaths"].sum() / tr["py"].sum()

    def rel(dim):
        g = tr.groupby(dim).apply(lambda x: (x["deaths"].sum() / x["py"].sum()) / nat,
                                  include_groups=False)
        return g.to_dict()
    r_male, r_age, r_year = rel("male"), rel("age5"), rel("year")
    return np.array([nat * r_male.get(m, 1.0) * r_age.get(a, 1.0) * r_year.get(y, 1.0)
                     for m, a, y in zip(te["male"], te["age5"], te["year"])])


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------

def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)
    return float(2.0 * np.mean(term))


def _metrics(te: pd.DataFrame, rate_pp: np.ndarray) -> dict:
    obs_rate = (te["deaths"] / te["py"]).values * 1e5
    pred_rate = rate_pp * 1e5
    mu_count = rate_pp * te["py"].values
    # 캘리브레이션 기울기(PY 가중 OLS): 관측률 = a + b·예측률
    w = te["py"].values
    pm = np.average(pred_rate, weights=w)
    om = np.average(obs_rate, weights=w)
    cov = np.average((pred_rate - pm) * (obs_rate - om), weights=w)
    var = np.average((pred_rate - pm) ** 2, weights=w)
    slope = float(cov / var) if var > 0 else float("nan")
    return {
        "deviance": round(_poisson_deviance(te["deaths"].values, mu_count), 4),
        "rmse_log": round(float(np.sqrt(np.mean(
            (np.log10(pred_rate + 1) - np.log10(obs_rate + 1)) ** 2))), 4),
        "mae_100k": round(float(np.mean(np.abs(pred_rate - obs_rate))), 3),
        "calib_slope": round(slope, 3),
    }


def _avg(dicts: list[dict]) -> dict:
    return {k: round(float(np.mean([d[k] for d in dicts])), 4) for k in dicts[0]}


def _run_cv(panel: pd.DataFrame, folds: list[tuple]) -> dict:
    """folds = [(train_df, test_df), ...]. 3모델 폴드평균 지표."""
    res = {"gbm": [], "proportional": [], "simple_mean": []}
    for tr, te in folds:
        res["gbm"].append(_metrics(te, _pred_gbm(_fit_gbm(tr), te)))
        res["proportional"].append(_metrics(te, _pred_proportional(tr, te)))
        res["simple_mean"].append(_metrics(te, _pred_simple_mean(tr, te)))
    return {m: _avg(v) for m, v in res.items()}


def spatial_cv(panel: pd.DataFrame) -> dict:
    """leave-one-시도-out (시도 수 만큼 fold)."""
    folds = [(panel[panel["sido"] != s], panel[panel["sido"] == s])
             for s in sorted(panel["sido"].unique())]
    return _run_cv(panel, folds)


def temporal_cv(panel: pd.DataFrame, holdout_years: int = 2) -> dict:
    """최근 N개 연도 홀드아웃(미래 일반화)."""
    yrs = sorted(panel["year"].unique())
    cut = yrs[-holdout_years]
    folds = [(panel[panel["year"] < cut], panel[panel["year"] >= cut])]
    return _run_cv(panel, folds)


def feature_importance(panel: pd.DataFrame) -> pd.DataFrame:
    booster = _fit_gbm(panel)
    imp = pd.DataFrame({"feature": FEATURES,
                        "gain": booster.feature_importance("gain")})
    imp["gain%"] = (imp["gain"] / imp["gain"].sum() * 100).round(1)
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


_CALIB_OK = (0.5, 2.0)   # 채택 캘리브레이션 게이트(소표본 과적합 차단)


def _adopt(spatial: dict, temporal: dict) -> tuple[bool, str]:
    """채택 조건(보수적): GBM이 공간·시간 CV '모두'에서 ① 두 베이스라인보다 deviance가
    낮고 ② 캘리브레이션 기울기가 [0.5, 2.0] 안일 때만. deviance만 낮고 캘리브레이션이
    불안정(소표본 과적합)하면 채택하지 않는다 — 가격 적합성 우선."""
    def beats(cv):
        return (cv["gbm"]["deviance"] < cv["proportional"]["deviance"]
                and cv["gbm"]["deviance"] < cv["simple_mean"]["deviance"])

    def calibrated(cv):
        s = cv["gbm"]["calib_slope"]
        return s == s and _CALIB_OK[0] <= s <= _CALIB_OK[1]   # NaN 제외

    dev_ok = beats(spatial) and beats(temporal)
    cal_ok = calibrated(spatial) and calibrated(temporal)
    if dev_ok and cal_ok:
        return True, "GBM이 공간·시간 CV 모두에서 베이스라인 deviance를 이기고 캘리브레이션도 안정 → surface 교체 가능."
    reasons = []
    if not beats(spatial): reasons.append("공간 deviance 미달")
    if not beats(temporal): reasons.append("시간 deviance 미달")
    if not calibrated(spatial): reasons.append("공간 캘리브레이션 불안정")
    if not calibrated(temporal): reasons.append("시간 캘리브레이션 불안정")
    return False, ("baseline 유지 — " + ", ".join(reasons)
                   + ". '단순 구조모델로 충분/소표본 과적합'도 유효한 검증 결과(과적합 억지 추월 안 함).")


@dataclass
class ModuleAResult:
    target: str
    n_cells: int
    n_zero_pct: float
    spatial: dict
    temporal: dict
    importance: pd.DataFrame
    adopt_gbm: bool
    note: str


def evaluate(target: str = "allcause") -> ModuleAResult:
    """모듈 A 전체 평가(결정적). target='allcause'(1차) | 'external'(보조)."""
    panel = build_panel(target)
    sp, tp = spatial_cv(panel), temporal_cv(panel)
    adopt, note = _adopt(sp, tp)
    return ModuleAResult(
        target=target, n_cells=len(panel),
        n_zero_pct=round(float((panel["deaths"] == 0).mean() * 100), 2),
        spatial=sp, temporal=tp, importance=feature_importance(panel),
        adopt_gbm=adopt, note=note,
    )

"""FastAPI 백엔드 — 발생률·생태회귀·층화·검증·예산 추정 제공.

무거운 계산(발생률 surface, 생태회귀, 층화, LORO)은 최초 호출 시 1회 계산해
캐시한다. 대시보드(dashboard/index.html)가 이 엔드포인트들을 소비한다.

실행: uvicorn src.api.main:app --reload
"""
from __future__ import annotations

import math
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import config
from ..data import benefits
from ..models import (calibration, cell_panel, cohort, ecological, injury_ml,
                      population, rates, stratify)
from ..optimize import budget, premium
from ..validation import military_proxy, report

app = FastAPI(title="군복무 청년 상해보험 리스크·계리 대시보드")

DASHBOARD = config.ROOT / "dashboard" / "index.html"


def _clean(obj):
    """NaN/numpy 스칼라를 JSON 안전 값으로 변환."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _records(df, reset_index=True):
    d = df.reset_index() if reset_index else df
    return [{k: _clean(v) for k, v in row.items()} for row in d.to_dict("records")]


# --- 캐시된 계산 ----------------------------------------------------------

@lru_cache(maxsize=1)
def _rate_table():
    return rates.conscript_rate_table()


@lru_cache(maxsize=1)
def _ecological():
    return ecological.fit()


@lru_cache(maxsize=1)
def _stratify():
    return stratify.stratify()


@lru_cache(maxsize=1)
def _validation():
    return report.loro_calibration()


# --- 엔드포인트 -----------------------------------------------------------

@app.get("/")
def index():
    if DASHBOARD.exists():
        return FileResponse(DASHBOARD)
    return {"message": "dashboard/index.html not found", "note": config.HONESTY_NOTE}


@app.get("/api/meta")
def meta():
    return {"honesty_note": config.HONESTY_NOTE,
            "coverage_items": {k: v["label"] for k, v in config.COVERAGE_ITEMS.items()}}


@app.get("/api/rates")
def api_rates():
    return {"note": config.HONESTY_NOTE, "rows": _records(_rate_table())}


@app.get("/api/ecological")
def api_ecological():
    efit = _ecological()
    return {
        "n_obs": efit.n_obs,
        "deviance_explained": _clean(efit.pseudo_r2),
        "rows": _records(ecological.summary_table(efit), reset_index=False),
    }


@app.get("/api/stratify")
def api_stratify():
    st = _stratify()
    return {"irr": _clean(st["irr"]), "rows": _records(st["profile"])}


@app.get("/api/validation")
def api_validation():
    val = _validation()
    return {"metrics": _to_jsonable(val["metrics"]),
            "rows": _records(val["predictions"])}


@app.get("/api/schedules")
def api_schedules():
    out = []
    for name in benefits.list_schedules():
        sched = benefits.SCHEDULES[name]
        out.append({"name": name, "sido": benefits.SCHEDULE_SIDO[name],
                    "province_wide": benefits.IS_PROVINCE_WIDE.get(name, False),
                    "approx": benefits.is_approx(name),
                    "items": {k: v["amount"] for k, v in sched.items()}})
    return {"schedules": out}


@app.get("/api/calibration")
def api_calibration():
    return {"rows": _records(calibration.calibration_factors())}


@app.get("/api/consistency")
def api_consistency():
    con = report.disease_track_consistency()
    return {"coverage_le_all_death_ok": con["coverage_le_all_death_ok"],
            "disease_death_le_incidence_ok": con["disease_death_le_incidence_ok"],
            "m0_envelope": con["m0_envelope"],
            "rows": _records(con["table"].round(4))}


@app.get("/api/military_validation")
def api_military_validation():
    """군 코호트 proxy 타당성 검증(국방부 사망사고 통계). pricing 미변경 — 정직성/검증용."""
    s = military_proxy.suicide_adjustment()
    e = military_proxy.external_crossvalidation()
    return _clean({
        "source": military_proxy.MND_SOURCE,
        "category_map": military_proxy.CATEGORY_MAP,
        "suicide": {"adopted_ratio": s["adopted"], "recent5": s["ratio_recent5_mean"],
                    "full": s["ratio_full_mean"], "excluded_years": s["excluded_years"],
                    "direction": s["direction"], "rows": _records(s["table"])},
        "external": {"trend_pearson_r": e["trend_pearson_r"],
                     "level_ratio_mil_over_gen": e["level_ratio_mil_over_gen"],
                     "overestimate_factor": e["overestimate_factor"],
                     "verdict": e["verdict"], "rows": _records(e["table"])},
        "pricing_impact": "없음 — M0 직접관측 백본 불변, 민감도·정직성 병기만.",
    })


def _module_a_dict(r):
    d = {"n_cells": r.n_cells, "n_zero_pct": r.n_zero_pct,
         "years": list(r.years), "spatial": r.spatial, "temporal": r.temporal,
         "adopt_gbm": r.adopt_gbm, "note": r.note,
         "importance": _records(r.importance, reset_index=False)}
    if r.temporal_ex_switch is not None:
        d["temporal_ex_switch"] = r.temporal_ex_switch   # 소스전환연도(2020) 제외 민감도
    return d


@lru_cache(maxsize=1)
def _ai_performance():
    """AI 성능 2층: 모듈 A(집단 셀 발생률 학습·공간시간CV) + M2(개인 손상 ML, 음성결과)."""
    mods = {
        "allcause": _module_a_dict(cell_panel.evaluate("allcause")),
        "external": _module_a_dict(cell_panel.evaluate("external")),           # 2005–2024 확장
        "external_prior136": _module_a_dict(                                   # 이전 136셀(대조)
            cell_panel.evaluate("external", years=[2021, 2022, 2023, 2024])),
    }
    m2 = injury_ml.evaluate()
    return {
        "module_a": mods,
        "m2_individual": {
            "n_obs": m2.n_obs, "n_pos": m2.n_pos, "prevalence": m2.prevalence,
            "kfold": m2.kfold, "holdout": m2.holdout, "adopt": m2.adopt,
            "note": m2.note,
            "odds_ratios": _records(m2.odds_ratios, reset_index=False)},
        "framing": ("2층 구조: 개인 예측은 약함(M2 AUC≈0.56, age+sex 베이스라인 미달) → "
                    "집단(셀) 발생률을 학습모델로 추정·공간시간 CV 검증(모듈 A). "
                    "두 모듈 모두 baseline 대비 정직 비교 결과를 그대로 보고한다."),
    }


@app.get("/api/ai_performance")
def api_ai_performance():
    return _ai_performance()


@lru_cache(maxsize=1)
def _cohort():
    s = cohort.summary()
    return {"risk_index": _records(s["risk_index_table"]),
            "bmi_selection": {k: _clean(v) for k, v in s["bmi_selection"].items()},
            "irr_source": _records(s["irr_source"], reset_index=False),
            "predictors": s["predictors"], "n_regions": s["n_regions"]}


@app.get("/api/cohort")
def api_cohort():
    """M4 — 코호트 리스크 지수 + 현역 BMI 선택 + 생태회귀 IRR 출처."""
    return _cohort()


@app.get("/api/premium")
def api_premium(schedule: str = "경기도", alpha: float = premium.DEFAULT_ALPHA,
                expense_ratio: float = premium.DEFAULT_EXPENSE_RATIO):
    """M5 — 계리 보험료(순+위험할증+사업비) + 보고치/단순할증 대조."""
    pr = premium.actuarial_premium(schedule, alpha=alpha, expense_ratio=expense_ratio)
    return {
        "schedule": pr.schedule, "sido": pr.sido, "population": pr.population,
        "net_pc": pr.net_pc, "risk_margin_pc": pr.risk_margin_pc,
        "expense_ratio": pr.expense_ratio, "gross_pc": pr.gross_pc,
        "cv": pr.cv, "alpha": pr.alpha, "implied_loading": pr.implied_loading,
        "reported_pc": _clean(pr.reported_pc), "note": pr.note,
        "compare": {k: _clean(v) for k, v in premium.compare_to_reported(schedule).items()},
        "sensitivity": _records(premium.loading_sensitivity(schedule), reset_index=False),
        "by_item": _records(pr.by_item, reset_index=False),
    }


@app.get("/api/population")
def api_population(schedule: str, year: int = 2024):
    sido = benefits.SCHEDULE_SIDO[schedule]
    years = list(range(2024, 2036, 2))
    proj = budget.project_budget(schedule, years)
    return {"schedule": schedule, "sido": sido,
            "province_wide": benefits.IS_PROVINCE_WIDE.get(schedule, False),
            "projection": _records(proj, reset_index=False)}


class BudgetRequest(BaseModel):
    schedule: str
    population: int | None = None       # None이면 인구추계·병무청 기반 자동 산출
    year: int = 2024
    annual_budget: float | None = None  # None이면 전체 보장 기대청구액만


@app.post("/api/budget")
def api_budget(req: BudgetRequest):
    est = budget.expected_claims(req.schedule, req.population, req.year)
    out = {
        "schedule": req.schedule,
        "sido": est.sido,
        "year": est.year,
        "population": est.population,
        "population_source": est.population_source,
        "total_claims": _clean(est.total_claims),
        "per_capita_claims": _clean(est.per_capita_claims),
        "premium_per_capita": _clean(est.premium_per_capita),
        "reported_premium": benefits.REPORTED_PREMIUM.get(req.schedule),
        "by_item": _records(est.by_item, reset_index=False),
    }
    if req.annual_budget is not None:
        opt = budget.optimize_under_budget(req.schedule, req.annual_budget,
                                           req.population, req.year)
        out["optimization"] = {
            "budget": req.annual_budget,
            "feasible_full": opt["feasible_full"],
            "coverage_scale": {k: _clean(round(v, 3)) for k, v in opt["coverage_scale"].items()},
            "adjusted_premium_per_capita": _clean(opt["estimate"].premium_per_capita),
            "by_item": _records(opt["estimate"].by_item, reset_index=False),
        }
    return out


def _to_jsonable(d):
    if isinstance(d, dict):
        return {k: _to_jsonable(v) for k, v in d.items()}
    return _clean(d)

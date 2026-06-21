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
from ..models import ecological, rates, stratify
from ..optimize import budget
from ..validation import report

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
                    "items": {k: v["amount"] for k, v in sched.items()}})
    return {"schedules": out}


class BudgetRequest(BaseModel):
    schedule: str
    population: int = 85_000
    annual_budget: float | None = None  # None이면 전체 보장 기대청구액만


@app.post("/api/budget")
def api_budget(req: BudgetRequest):
    est = budget.expected_claims(req.schedule, req.population)
    out = {
        "schedule": req.schedule,
        "sido": est.sido,
        "population": req.population,
        "total_claims": _clean(est.total_claims),
        "per_capita_claims": _clean(est.per_capita_claims),
        "premium_per_capita": _clean(est.premium_per_capita),
        "by_item": _records(est.by_item, reset_index=False),
    }
    if req.annual_budget is not None:
        opt = budget.optimize_under_budget(req.schedule, req.population, req.annual_budget)
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

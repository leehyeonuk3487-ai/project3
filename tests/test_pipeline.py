"""핵심 파이프라인 smoke 테스트 (실데이터 기반).

pytest 없이도 `python -m tests.test_pipeline` 로 실행 가능.
"""
from __future__ import annotations

import numpy as np

from src import config
from src.data import aggregate, benefits, loaders
from src.models import ecological, rates, stratify
from src.optimize import budget


def test_loaders_shapes():
    chs = loaders.load_chs()
    assert len(chs) > 200_000
    assert chs["sido"].nunique() == 17
    assert chs["bmi"].between(10, 60).mean() > 0.95  # BMI 현실 범위


def test_rate_surface_positive_and_decomposed():
    table = rates.conscript_rate_table()
    assert (table.fillna(0) >= 0).all().all()
    # 상해사망 < 중증외상 발생 (치명률<1)
    assert (table["상해사망"] < table["중증외상 발생"]).all()


def test_region_disparity_meaningful():
    table = rates.conscript_rate_table()
    s = table["골절"].dropna()
    assert s.max() / s.min() > 2.0  # 지역 격차 유의미


def test_ecological_fit():
    efit = ecological.fit()
    assert efit.n_obs == 17
    assert set(efit.predictors).issubset(set(efit.irr.index))
    assert (efit.irr > 0).all()


def test_stratify_irr_increases():
    st = stratify.stratify()
    prof = st["profile"]
    rt = prof["rate_severe_trauma"]
    assert rt.iloc[0] < rt.iloc[-1]          # 저<고
    assert st["irr"] > 1.0


def test_budget_estimate_reasonable():
    est = budget.expected_claims("경기도", population=85_000)
    assert est.total_claims > 0
    # 1인당 기대청구액은 수천~수만원 수준(자릿수 sanity)
    assert 1_000 < est.per_capita_claims < 200_000


def test_budget_respects_constraint():
    pop = 85_000
    opt = budget.optimize_under_budget("경기도", pop, annual_budget=pop * 15_000)
    assert opt["estimate"].premium_per_capita <= 15_000 + 1
    for v in opt["coverage_scale"].values():
        assert 0.0 <= v <= 1.0


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

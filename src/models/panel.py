"""M3 — 셀 패널 GBM 발생률 surface (모듈 A).

목적: 비례보정(주변 독립 가정 = 로그 가산모형)을 **학습된 상호작용**으로
대체할 수 있는지 검증한다. 베이스라인을 교차검증에서 이길 때만 채택한다.

가용 joint 패널: 퇴원손상 퇴원율(성 × 연령 × 연도, 2005–2023). 이 표는 실제
성×연령 교차표라 비례보정 없이 관측되며, 시간축까지 있어 패널 학습이 가능하다.
⚠️ 지역(시도) 축이 joint로 관측되지 않으므로(중증외상은 주변분포만) **공간 CV
(leave-one-시도-out)는 불가** — 본 모듈은 성×연령×연도 패널의 시간 일반화만 평가.

베이스라인(고정): log(rate) ~ 성 + 연령(범주) + 연도  (가산=비례보정의 로그 등가).
ML(고정): LightGBM 회귀, 정규화(얕은 트리·min_child·early stopping).
CV(고정): ① leave-recent-years-out(최근 5년 홀드아웃) ② 셔플 k-fold. 시드 고정.
채택 규칙: 시간 홀드아웃 RMSE가 베이스라인보다 낮을 때만 'GBM 채택', 아니면
베이스라인 유지하고 그 사실을 기록.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import loaders

SEED = 42
AGE_MIDPOINT = {
    "0-14세": 7, "15-24세": 20, "25-34세": 30, "35-44세": 40, "45-54세": 50,
    "55-64세": 60, "65-74세": 70, "75세이상": 80,
}
HOLDOUT_YEARS = 5  # 최근 N년 시간 홀드아웃


def build_panel() -> pd.DataFrame:
    """성×연령×연도 퇴원손상 발생률 패널(전체 행 제외)."""
    df = loaders.load_discharge_rate_by_sex_age()
    sub = df[df["metric"].str.contains("퇴원율", na=False)].copy()
    sub = sub[(sub["dim"].isin(["남자", "여자"])) & (sub["level"].isin(AGE_MIDPOINT))]
    sub = sub.rename(columns={"dim": "sex", "level": "age_band", "value": "rate_per_100k"})
    sub["age_mid"] = sub["age_band"].map(AGE_MIDPOINT)
    sub["male"] = (sub["sex"] == "남자").astype(int)
    sub["log_rate"] = np.log(sub["rate_per_100k"])
    return sub[["sex", "male", "age_band", "age_mid", "year", "rate_per_100k", "log_rate"]].dropna()


_FEATURES = ["male", "age_mid", "year"]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    return df[_FEATURES].astype(float)


# --- 베이스라인: 가산(비례보정 등가) 로그선형 -------------------------------

def _fit_baseline(train: pd.DataFrame):
    """log(rate) ~ male + C(age) + year (가산모형). OLS(정규방정식)."""
    import numpy as np
    X = _design_additive(train)
    y = train["log_rate"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _design_additive(df: pd.DataFrame) -> np.ndarray:
    ages = sorted(AGE_MIDPOINT, key=AGE_MIDPOINT.get)
    cols = [np.ones(len(df)), df["male"].to_numpy(float),
            (df["year"].to_numpy(float) - 2014)]
    for a in ages[1:]:  # 첫 연령 기준 더미
        cols.append((df["age_band"] == a).to_numpy(float))
    return np.column_stack(cols)


def _predict_baseline(beta, df: pd.DataFrame) -> np.ndarray:
    return _design_additive(df) @ beta


# --- ML: LightGBM 회귀(정규화) ----------------------------------------------

def _fit_gbm(train: pd.DataFrame, valid: pd.DataFrame | None = None):
    import lightgbm as lgb
    params = dict(objective="regression", n_estimators=400, learning_rate=0.03,
                  num_leaves=8, max_depth=3, min_child_samples=10,
                  subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
                  random_state=SEED, verbose=-1)
    model = lgb.LGBMRegressor(**params)
    fit_kw = {}
    if valid is not None and len(valid):
        import lightgbm as _l
        fit_kw = dict(eval_set=[(_features(valid), valid["log_rate"])],
                      callbacks=[_l.early_stopping(40, verbose=False)])
    model.fit(_features(train), train["log_rate"], **fit_kw)
    return model


def _rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


@dataclass
class PanelResult:
    temporal: dict          # 시간 홀드아웃 RMSE(log) 베이스라인/GBM
    kfold: dict             # 셔플 k-fold 평균 RMSE
    adopt_gbm: bool         # 시간 홀드아웃에서 GBM이 베이스라인을 이겼는가
    n_obs: int
    note: str


def fit_full_gbm():
    """전체 패널로 GBM 적합(채택 시 surface 예측용)."""
    return _fit_gbm(build_panel())


def predict_surface(year: int = 2023) -> pd.DataFrame:
    """채택된 GBM으로 성×연령 손상 발생률 surface 예측(특정 연도).

    지역축이 없으므로 기존 비례보정의 '성×연령 기준면'을 GBM 추정치로 대체·평활하는
    용도(지역 상대위험은 비례보정 백본 유지).
    """
    model = fit_full_gbm()
    rows = [{"sex": "남자" if male else "여자", "male": male,
             "age_band": ab, "age_mid": AGE_MIDPOINT[ab], "year": year}
            for male in (1, 0) for ab in AGE_MIDPOINT]
    g = pd.DataFrame(rows)
    g["rate_per_100k"] = np.exp(model.predict(_features(g)))
    return g[["sex", "age_band", "year", "rate_per_100k"]]


def evaluate(k: int = 5) -> PanelResult:
    """베이스라인 vs GBM을 시간 홀드아웃·k-fold로 비교."""
    rng = np.random.default_rng(SEED)
    panel = build_panel()

    # ① 시간 홀드아웃(최근 HOLDOUT_YEARS년)
    cutoff = int(panel["year"].max()) - HOLDOUT_YEARS
    tr, te = panel[panel["year"] <= cutoff], panel[panel["year"] > cutoff]
    base = _fit_baseline(tr)
    gbm = _fit_gbm(tr, te)
    t_base = _rmse(te["log_rate"], _predict_baseline(base, te))
    t_gbm = _rmse(te["log_rate"], gbm.predict(_features(te)))

    # ② 셔플 k-fold
    idx = rng.permutation(len(panel))
    folds = np.array_split(idx, k)
    kb, kg = [], []
    for i in range(k):
        te_i = panel.iloc[folds[i]]
        tr_i = panel.iloc[np.concatenate([folds[j] for j in range(k) if j != i])]
        b = _fit_baseline(tr_i)
        g = _fit_gbm(tr_i)
        kb.append(_rmse(te_i["log_rate"], _predict_baseline(b, te_i)))
        kg.append(_rmse(te_i["log_rate"], g.predict(_features(te_i))))

    adopt = t_gbm < t_base
    return PanelResult(
        temporal={"baseline_rmse": round(t_base, 4), "gbm_rmse": round(t_gbm, 4),
                  "train_n": len(tr), "test_n": len(te), "holdout_years": HOLDOUT_YEARS},
        kfold={"baseline_rmse": round(float(np.mean(kb)), 4),
               "gbm_rmse": round(float(np.mean(kg)), 4), "k": k},
        adopt_gbm=bool(adopt),
        n_obs=len(panel),
        note=("시간 홀드아웃에서 GBM이 베이스라인(비례보정 등가)을 이김 → 채택 가능"
              if adopt else
              "GBM이 베이스라인을 이기지 못함 → 비례보정 백본 유지(작업지시 채택규칙)"),
    )

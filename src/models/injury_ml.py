"""M2 — KNHANES 개인 손상위험 ML (집단 가격용, 개인 스코어링 아님).

데이터: data/processed/knhanes_filtered_2021_2024.csv (IJMT 모듈 포함, 만15–39세).

결과변수: ij_expr (1=지난1년 손상경험, 0=없음). AC1_yr(1/2/9)의 정제 이진본으로
   교차검증 결과 정의 완전 일치(AC1_yr==1↔ij_expr==1, ==2↔0, 9·NaN↔NaN). 9(모름)·
   NaN은 제외. 유효표본 5,730명(양성 386, 6.7%).

예측변수(작업지시 고정): age, sex(male), HE_BMI, 흡연(sm_presnt), 음주(dr_month),
   활동(pa_aerobic), 고혈압 의사진단(DI1_dg==1), 당뇨 의사진단(DE1_dg==1).
   ※AC3/MH/MO 등 손상 이후 결과변수는 누수 위험으로 제외.

표본 전략(★필수): 20대 남성 양성은 85개(N≈1,086)로 과소 → 전 연령(15–39)·전성별
   풀링 학습 + age·sex 피처, 적용·보고는 20대 남성. 20대만 학습 금지.
   불균형은 class_weight='balanced'로 보정. 표본가중(wt_*)은 학습에 미적용(설계가중은
   유병률 추정용이며 본 모듈은 위험요인 판별·순위가 목적 → 미적용 명시).

방법: ① L2 로지스틱(표준화+중앙값 임퓨테이션 파이프라인) ② LightGBM(얕은 트리·정규화).
   계층 5-fold + 연도(2024) 홀드아웃. 베이스라인=age+sex.

정직성(필수): 자가보고·일반인구 손상이라 군 훈련손상과 다름(전이 한계). 산출물은
   집단 단위 위험요인·상대위험이며 개인 장병 스코어링/보험료 차등에 쓰지 않는다.
   결측 임퓨테이션·소표본으로 AUC는 modest일 수 있으며 그대로 보고한다.
재현성: 모든 추정기 random_state=SEED 고정.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config

SEED = 42
FEATURES = ["age", "male", "HE_BMI", "sm_presnt", "dr_month",
            "pa_aerobic", "htn_dx", "dm_dx"]
BASELINE = ["age", "male"]
HOLDOUT_YEAR = 2024


def load_features() -> pd.DataFrame:
    """KNHANES → 분석용 피처 프레임[y, year, FEATURES]. ij_expr 결측 제외."""
    df = pd.read_csv(config.KNHANES_CSV, encoding="utf-8-sig")
    df = df[df["ij_expr"].notna()].copy()
    out = pd.DataFrame({
        "y": df["ij_expr"].astype(int),
        "year": df["year"].astype(int),
        "age": df["age"].astype(float),
        "male": (df["sex"] == 1).astype(int),
        "HE_BMI": pd.to_numeric(df["HE_BMI"], errors="coerce"),
        "sm_presnt": pd.to_numeric(df["sm_presnt"], errors="coerce"),
        "dr_month": pd.to_numeric(df["dr_month"], errors="coerce"),
        "pa_aerobic": pd.to_numeric(df["pa_aerobic"], errors="coerce"),
        # 의사진단=1만 양성, 0/8(비해당)은 진단없음으로 처리
        "htn_dx": (df["DI1_dg"] == 1).astype(int),
        "dm_dx": (df["DE1_dg"] == 1).astype(int),
    })
    return out.reset_index(drop=True)


def _logit_pipeline(balanced: bool = True):
    # L2(ridge)는 sklearn 기본 — penalty 인자는 1.8+ deprecated이므로 미지정.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # 폴드 train 기준 → 누수 없음
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced" if balanced else None,
                                  max_iter=2000, random_state=SEED)),
    ])


def _lgbm(balanced: bool = True):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=3,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced" if balanced else None,
        random_state=SEED, n_jobs=1, verbose=-1,
    )


def _model(name: str, balanced: bool = True):
    return _logit_pipeline(balanced) if name in ("baseline", "logit") else _lgbm(balanced)


def _kfold_auc(name: str, data: pd.DataFrame, features: list[str], k: int = 5) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    scores = cross_val_score(_model(name), data[features], data["y"],
                             cv=skf, scoring="roc_auc")
    return {"auc_mean": round(float(scores.mean()), 4),
            "auc_std": round(float(scores.std()), 4), "k": k}


def _year_holdout(name: str, data: pd.DataFrame, features: list[str]) -> dict:
    """연도 홀드아웃: 판별(AUC)은 불균형 가중 모델, 캘리브레이션은 비가중(교정) 모델.

    class_weight='balanced'는 결정경계를 50:50로 이동시켜 절대확률을 비교정으로 만든다
    (AUC 같은 순위지표엔 무영향). 따라서 절대확률·Brier는 비가중 적합으로 보고한다.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score
    tr, te = data[data["year"] < HOLDOUT_YEAR], data[data["year"] == HOLDOUT_YEAR]
    auc = roc_auc_score(te["y"], _model(name, balanced=True)
                        .fit(tr[features], tr["y"]).predict_proba(te[features])[:, 1])
    pc = _model(name, balanced=False).fit(tr[features], tr["y"]).predict_proba(te[features])[:, 1]
    return {"holdout_year": HOLDOUT_YEAR, "n_test": int(len(te)),
            "auc": round(float(auc), 4),
            "brier": round(float(brier_score_loss(te["y"], pc)), 5),
            "pred_mean_cal": round(float(pc.mean()), 4),     # 비가중 → 유병률 교정
            "obs_rate": round(float(te["y"].mean()), 4)}


def odds_ratios(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """전체데이터 적합 L2 로지스틱의 위험요인 상대위험(OR).

    연속형(age, HE_BMI)은 +1 SD당 OR, 이진형은 유(1) vs 무(0) OR.
    집단 단위 위험요인 해석용 — 개인 예측 점수가 아니다.
    """
    data = load_features() if data is None else data
    pipe = _logit_pipeline(balanced=True).fit(data[FEATURES], data["y"])
    coef = pipe.named_steps["lr"].coef_[0]
    cont = {"age", "HE_BMI"}
    rows = [{"위험요인": f, "단위": "+1 SD" if f in cont else "유 vs 무",
             "OR": round(float(np.exp(c)), 3)} for f, c in zip(FEATURES, coef)]
    return pd.DataFrame(rows).sort_values("OR", ascending=False).reset_index(drop=True)


def lgbm_importance(data: pd.DataFrame | None = None) -> pd.DataFrame:
    """LightGBM gain 중요도(전체데이터 적합). 위험요인 상대 기여 순위."""
    data = load_features() if data is None else data
    mdl = _lgbm().fit(data[FEATURES], data["y"])
    imp = pd.DataFrame({"위험요인": FEATURES,
                        "gain": mdl.booster_.feature_importance("gain")})
    imp["gain%"] = (imp["gain"] / imp["gain"].sum() * 100).round(1)
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def conscript_group_risk(data: pd.DataFrame | None = None) -> dict:
    """20대 남성 집단의 예측 손상위험 분포(★집단 요약만, 개인 점수 미출력).

    비가중(교정) 모델로 절대위험이 유병률에 정렬되게 한다. 절대 가격은 계리
    베이스율(M0/rates)에서 오며, 본 분포는 집단 위험요인 분산 참고용이다.
    """
    data = load_features() if data is None else data
    mdl = _logit_pipeline(balanced=False).fit(data[FEATURES], data["y"])
    m20 = data[(data["male"] == 1) & data["age"].between(20, 29)]
    p = mdl.predict_proba(m20[FEATURES])[:, 1]
    return {"n": int(len(m20)), "observed_rate": round(float(m20["y"].mean()), 4),
            "pred_mean": round(float(p.mean()), 4),
            "pred_p25": round(float(np.percentile(p, 25)), 4),
            "pred_p75": round(float(np.percentile(p, 75)), 4),
            "note": "집단 평균·분위만 — 개인 장병 스코어링 아님"}


@dataclass
class M2Result:
    n_obs: int
    n_pos: int
    prevalence: float
    kfold: dict
    holdout: dict
    adopt: str
    odds_ratios: pd.DataFrame
    importance: pd.DataFrame
    group_risk: dict
    note: str


def evaluate() -> M2Result:
    """3모델(베이스라인/로지스틱/LightGBM) CV·홀드아웃 + 위험요인. 결정적(시드 고정)."""
    data = load_features()
    kf, ho = {}, {}
    for name, feats in [("baseline", BASELINE), ("logit", FEATURES), ("lgbm", FEATURES)]:
        kf[name] = _kfold_auc(name, data, feats)
        ho[name] = _year_holdout(name, data, feats)

    # 채택: 연도 홀드아웃 AUC가 베이스라인보다 높은 모델 중 최고(동률이면 단순한 로지스틱)
    base = ho["baseline"]["auc"]
    cands = {m: ho[m]["auc"] for m in ("logit", "lgbm") if ho[m]["auc"] > base}
    adopt = max(cands, key=cands.get) if cands else "baseline"
    lift = (cands[adopt] - base) if cands else 0.0

    note = (f"홀드아웃 AUC 베이스라인(age+sex) {base} → 채택 {adopt} "
            f"{ho[adopt]['auc']} (Δ{lift:+.4f}). 자가보고·일반인구 손상 → 군 훈련손상과 "
            f"다름(전이 한계). 집단 위험요인용, 개인 스코어링 금지.")
    return M2Result(
        n_obs=len(data), n_pos=int(data["y"].sum()),
        prevalence=round(float(data["y"].mean()), 4),
        kfold=kf, holdout=ho, adopt=adopt,
        odds_ratios=odds_ratios(data), importance=lgbm_importance(data),
        group_risk=conscript_group_risk(data), note=note,
    )

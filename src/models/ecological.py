"""생태학적 포아송 회귀 — 지역 위험요인 ↔ 중증외상 발생률.

KDCA는 성×연령×지역 joint cell을 제공하지 않으므로, 지역(시도 17개) 수준에서
중증외상 발생 건수를 위험요인 유병률로 회귀한다:

    count_r ~ Poisson(mu_r),   log(mu_r) = beta0 + Σ beta_k z_{r,k} + log(pop_r)

  - z : 시도별 위험요인 유병률(표준화)
  - offset : log(인구) — 발생건수/발생률로 역산한 노출인구
  - 결과 : 위험요인 SD 1 증가당 발생률비(IRR = exp(beta))

⚠️ 생태학적 회귀다. 계수는 '지역 단위' 연관이며 개인 단위 인과로 해석하면
생태학적 오류(ecological fallacy)에 빠진다. 관측치가 17개(시도)뿐이라
검정력이 낮으므로, 소수의 행태 위험요인만 사용하고 효과크기·방향의
참고치로만 쓴다. 층화(stratify.py)에서는 이 계수의 부호·상대크기를
가중치로 활용하되 절대 IRR을 개인 위험으로 주장하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..data import aggregate, loaders

# 생태회귀에 사용할 행태 위험요인(검정력 고려해 소수만)
DEFAULT_PREDICTORS = ["smoker", "binge_drink", "phys_inactive", "obese"]


@dataclass
class EcologicalFit:
    predictors: list[str]
    coef: pd.Series          # 표준화 계수(log scale), const 포함
    irr: pd.Series           # exp(coef) = SD당 발생률비
    pvalues: pd.Series
    conf_int: pd.DataFrame   # IRR 95% CI
    means: pd.Series = field(repr=False)   # 표준화 전 예측변수 평균
    stds: pd.Series = field(repr=False)    # 표준화 전 예측변수 SD
    n_obs: int = 0
    pseudo_r2: float = float("nan")   # 편차 설명 비율(deviance explained)
    cov: pd.DataFrame = field(default=None, repr=False)  # 모수 공분산(과산포 반영)
    scale: float = 1.0                # quasi-Poisson 과산포 scale φ

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        """표준화 설계행렬(const 포함, coef 순서)."""
        z = (X[self.predictors] - self.means) / self.stds
        z.insert(0, "const", 1.0)
        return z[self.coef.index]

    def linear_predictor(self, X: pd.DataFrame) -> pd.Series:
        """원척도 위험요인 값 → 상대 로그위험(절편 제외, 표준화 후 가중합)."""
        z = (X[self.predictors] - self.means) / self.stds
        beta = self.coef[self.predictors]
        return z.mul(beta, axis=1).sum(axis=1)

    def predict_rate_per_100k(self, X: pd.DataFrame) -> pd.Series:
        """원척도 위험요인 값 → 예측 발생률(인구 10만명당). 절편 포함."""
        return np.exp(self.coef["const"] + self.linear_predictor(X)) * 1e5

    def predict_rate_ci(self, X: pd.DataFrame, z: float = 1.96,
                        exposure: np.ndarray | pd.Series | None = None) -> pd.DataFrame:
        """예측 발생률(10만명당)과 95% 구간.

        SE(eta)² = xᵀVx (모수 불확실성). exposure(인구)가 주어지면 관측 발생률에
        대한 **예측구간**으로, 과산포·표본변동 φ/μ_count 를 더한다(μ_count=rate×pop).
        없으면 평균에 대한 신뢰구간.
        """
        Xd = self._design(X)
        eta = Xd.values @ self.coef.values
        V = self.cov.values
        var = np.einsum("ij,jk,ik->i", Xd.values, V, Xd.values)
        if exposure is not None:
            mu_count = np.exp(eta) * np.asarray(exposure)
            var = var + self.scale / np.maximum(mu_count, 1e-9)
        se = np.sqrt(var)
        return pd.DataFrame({
            "rate": np.exp(eta) * 1e5,
            "lo": np.exp(eta - z * se) * 1e5,
            "hi": np.exp(eta + z * se) * 1e5,
        }, index=X.index)


def _region_exposure() -> pd.DataFrame:
    """시도별 중증외상 발생건수와 노출인구(건수/발생률×1e5)."""
    df = loaders.load_severe_trauma_incidence()
    year = int(df["year"].max())
    sub = df[(df["dim"] == "시도별") & (df["year"] == year)]
    cnt = sub[sub["metric"].str.contains("건수", na=False)].set_index("level")["value"]
    rate = sub[sub["metric"].str.contains("발생률", na=False)].set_index("level")["value"]
    pop = (cnt / rate) * 1e5
    return pd.DataFrame({"count": cnt, "rate_per_100k": rate, "pop": pop})


def build_dataset(predictors: list[str] | None = None,
                  calibrate: bool = False) -> pd.DataFrame:
    """시도별 위험요인 유병률 + 발생건수/노출인구 병합 데이터셋.

    calibrate=True 이면 KNHANES 측정값으로 CHS 자가보고 유병률을 보정한다.
    """
    predictors = predictors or DEFAULT_PREDICTORS
    prev = aggregate.region_risk_prevalence(factors=predictors)
    if calibrate:
        from . import calibration
        prev = calibration.apply_calibration(prev)
    expo = _region_exposure()
    return prev.join(expo, how="inner").dropna(subset=predictors + ["count", "pop"])


def fit_dataset(data: pd.DataFrame, predictors: list[str]) -> EcologicalFit:
    """준비된 데이터셋으로 생태회귀를 적합한다(LORO 재사용)."""
    means = data[predictors].mean()
    stds = data[predictors].std(ddof=0).replace(0, 1.0)
    Z = (data[predictors] - means) / stds
    X = sm.add_constant(Z)

    model = sm.GLM(
        data["count"], X,
        family=sm.families.Poisson(),
        offset=np.log(data["pop"].values),
    )
    # quasi-Poisson: Pearson 카이제곱 기반 scale로 과산포 보정(시도 17개·대형 count
    # 환경에서 표준오차가 비현실적으로 좁아지는 것을 막는다).
    res = model.fit(scale="X2")

    irr = np.exp(res.params)
    ci = np.exp(res.conf_int())
    ci.columns = ["irr_lo", "irr_hi"]

    # 편차 설명 비율(deviance explained) — scale에 불변이라 quasi-Poisson에서도 안정적.
    null = sm.GLM(data["count"], np.ones((len(data), 1)),
                  family=sm.families.Poisson(),
                  offset=np.log(data["pop"].values)).fit()
    pseudo = 1.0 - res.deviance / null.deviance

    return EcologicalFit(
        predictors=predictors,
        coef=res.params,
        irr=irr,
        pvalues=res.pvalues,
        conf_int=ci,
        means=means,
        stds=stds,
        n_obs=len(data),
        pseudo_r2=float(pseudo),
        cov=res.cov_params(),
        scale=float(res.scale),
    )


def fit(predictors: list[str] | None = None, calibrate: bool = False) -> EcologicalFit:
    """생태학적 포아송 회귀를 적합한다(전체 시도)."""
    predictors = predictors or DEFAULT_PREDICTORS
    data = build_dataset(predictors, calibrate=calibrate)
    return fit_dataset(data, predictors)


def summary_table(efit: EcologicalFit) -> pd.DataFrame:
    """위험요인별 IRR·CI·p값 요약표."""
    rows = []
    for p in efit.predictors:
        rows.append({
            "위험요인": p,
            "IRR(SD당)": round(float(efit.irr[p]), 3),
            "CI_lo": round(float(efit.conf_int.loc[p, "irr_lo"]), 3),
            "CI_hi": round(float(efit.conf_int.loc[p, "irr_hi"]), 3),
            "p": round(float(efit.pvalues[p]), 4),
        })
    return pd.DataFrame(rows)

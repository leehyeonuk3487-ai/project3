"""검증 리포트 — LORO 캘리브레이션 + 베이스라인 대비.

개인 부상 라벨이 없으므로 AUC/C-index 대신 **지역 발생률 캘리브레이션**을
주지표로 쓴다. 생태회귀가 위험요인으로 지역 발생률을 설명하는지를
**leave-one-region-out(LORO)** 으로 외부검증한다:

  각 시도를 한 번씩 빼고 나머지 16개로 적합 → 빠진 시도의 발생률 예측 →
  관측값과 비교. 동시에 '전국 평균률(지역정보 없음)' 베이스라인과 견준다.

지표: Pearson r, MAE, RMSE, 캘리브레이션 기울기(관측 = a + b·예측).
모델 MAE < 베이스라인 MAE 이면 지역 위험요인 정보가 예측을 개선한 것이다.
정직성: 17개 시도 생태 데이터의 한계로 결과는 참고 수준이며, 음의 개선도
그대로 보고한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import ecological


def loro_calibration(predictors: list[str] | None = None) -> dict:
    """LORO 캘리브레이션을 수행하고 시도별 예측·지표를 반환한다."""
    predictors = predictors or ecological.DEFAULT_PREDICTORS
    data = ecological.build_dataset(predictors)

    rows = []
    for sido in data.index:
        train = data.drop(index=sido)
        test = data.loc[[sido]]
        fit = ecological.fit_dataset(train, predictors)
        pred = float(fit.predict_rate_per_100k(test).iloc[0])
        # 베이스라인: 학습 시도들의 인구가중 평균 발생률(지역 위험요인 무시)
        baseline = float((train["count"].sum() / train["pop"].sum()) * 1e5)
        rows.append({
            "sido": sido,
            "observed": float(test["rate_per_100k"].iloc[0]),
            "predicted": pred,
            "baseline": baseline,
        })
    pred_df = pd.DataFrame(rows).set_index("sido").sort_index()

    metrics = {
        "model": _error_metrics(pred_df["observed"], pred_df["predicted"]),
        "baseline": _error_metrics(pred_df["observed"], pred_df["baseline"]),
    }
    metrics["mae_improvement"] = (
        metrics["baseline"]["mae"] - metrics["model"]["mae"]
    )
    metrics["mae_improvement_pct"] = (
        100.0 * metrics["mae_improvement"] / metrics["baseline"]["mae"]
        if metrics["baseline"]["mae"] else float("nan")
    )
    return {"predictions": pred_df, "metrics": metrics, "predictors": predictors}


def _error_metrics(observed: pd.Series, predicted: pd.Series) -> dict:
    obs = observed.values
    pred = predicted.values
    err = pred - obs
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    # Pearson r (예측이 상수면 NaN)
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else float("nan")
    # 캘리브레이션 기울기 observed = a + b·predicted
    if np.std(pred) > 0:
        b, a = np.polyfit(pred, obs, 1)
    else:
        a, b = float("nan"), float("nan")
    return {"mae": mae, "rmse": rmse, "pearson_r": r,
            "calib_slope": float(b), "calib_intercept": float(a)}


def format_report(result: dict) -> str:
    """사람이 읽는 텍스트 리포트."""
    m = result["metrics"]
    lines = [
        "LORO(leave-one-region-out) 캘리브레이션 — 중증외상 발생률(10만명당)",
        f"예측변수: {', '.join(result['predictors'])}",
        "-" * 70,
        f"{'':10}{'MAE':>10}{'RMSE':>10}{'Pearson r':>12}{'calib slope':>14}",
        f"{'모델':<10}{m['model']['mae']:>10.2f}{m['model']['rmse']:>10.2f}"
        f"{m['model']['pearson_r']:>12.3f}{m['model']['calib_slope']:>14.3f}",
        f"{'베이스라인':<8}{m['baseline']['mae']:>10.2f}{m['baseline']['rmse']:>10.2f}"
        f"{m['baseline']['pearson_r']:>12.3f}{m['baseline']['calib_slope']:>14.3f}",
        "-" * 70,
        f"MAE 개선: {m['mae_improvement']:+.2f} ({m['mae_improvement_pct']:+.1f}%) "
        f"— 양수면 지역 위험요인 정보가 예측을 개선.",
    ]
    return "\n".join(lines)

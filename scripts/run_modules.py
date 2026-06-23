"""M0~M3 모듈 실행 리포트 (작업지시서).

  M0  사망원인 104→4범주 매핑·재추출 상태
  M1  병무청 BMI 연결 + KNHANES 피처 적재
  M3  셀 패널 GBM vs 비례보정 베이스라인 (시간 홀드아웃·k-fold)
  M2  개인 손상 ML — 라벨 부재 보고

실행: python -m scripts.run_modules
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data import loaders, mortality
from src.models import panel, population, rates

pd.set_option("display.width", 150)


def line(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    print(config.HONESTY_NOTE)

    line("M0 — 사망원인 → 4범주(질병/상해/자살/기타) 분리 + 직접 사망률")
    print("4범주 분해 (2021–24, 남자 20대, 전국):")
    print(mortality.four_category_summary().to_string(index=False))
    env = mortality.envelope_check()
    print(f"\nMECE envelope: 4범주합 {env['category_sum_deaths']:,}명 vs 전체사망(계) "
          f"{env['all_cause_deaths']:,}명  (오차 {env['rel_error_pct']}%)")
    cov = rates.conscript_item_rate
    inj, dis = cov("death_injury"), cov("death_disease")
    print(f"트랙 연결: 상해사망(external 직접) 전국 {inj.mean():.4f} / "
          f"질병사망(disease 직접) {dis.mean():.4f} (1,000명·년) — 비외상 proxy 폐기")
    print("자살(X60–84)은 death_suicide_excluded(면책) → 예산·보장 미포함, envelope만.")
    # 상위 사인 sanity check (순위표 검증용)
    top = (mortality.load_mortality_by_cause().groupby("cause")["deaths"].sum()
           .sort_values(ascending=False).head(5))
    print("상위 사인(검증용 순위):")
    for c, n in top.items():
        print(f"   {int(n):5d}  {c}")

    line("M1 — 병무청 BMI 연결 + KNHANES 피처 적재")
    prof = population.cohort_profile("경기", 2024)
    print(f"경기 현역 코호트 N={prof['conscripts']:,} (CHS 20대남 표본 {prof['n_chs']})")
    bmi = {k: round(v, 3) for k, v in prof["bmi_distribution"].items()}
    print(f"  병무청 BMI 분포(실연결): {bmi}")
    print("  CHS 위험요인 유병률:",
          {k: round(v, 3) for k, v in prof["risk_prevalence"].items()})
    kf = loaders.load_knhanes_features()
    print(f"  KNHANES 피처 적재: {kf.shape[0]}행 × {list(kf.columns)}")

    line("M3 — 셀 패널 GBM vs 비례보정(로그가산) 베이스라인")
    r = panel.evaluate()
    print(f"패널(성×연령×연도) 셀: {r.n_obs}")
    print(f"시간 홀드아웃(최근{r.temporal['holdout_years']}년) RMSE(log): "
          f"베이스라인 {r.temporal['baseline_rmse']} → GBM {r.temporal['gbm_rmse']}")
    print(f"셔플 {r.kfold['k']}-fold RMSE(log): "
          f"베이스라인 {r.kfold['baseline_rmse']} → GBM {r.kfold['gbm_rmse']}")
    print(f"채택 규칙(시간 홀드아웃 기준): GBM 채택={r.adopt_gbm}")
    print(f"판정: {r.note}")
    print("⚠️ 지역(시도) joint 관측 부재 → 공간 CV 불가, 비례보정 지역 백본 유지.")

    line("M2 — 개인 손상위험 ML (모듈 B)")
    print("블로커: KNHANES 처리본에 손상(IJMT) 결과변수가 없음 → 지도학습 라벨 부재.")
    print("임의구현 금지 원칙에 따라 결과변수 임의대체 없이 보류(데이터 보강 필요).")


if __name__ == "__main__":
    main()

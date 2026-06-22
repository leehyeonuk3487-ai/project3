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
from src.models import panel, population

pd.set_option("display.width", 150)


def line(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main() -> None:
    print(config.HONESTY_NOTE)

    line("M0 — 사망원인 104항목 → 4범주(질병/상해/자살/기타)")
    print(mortality.mapping_table().to_string(index=False))
    st = mortality.status()
    print(f"\n재추출 파일 존재: {st['by_cause_available']}  ({st['current_file']})")
    print(f"블로커: {st['blocker']}")
    print(f"필요 파일: {st['needed_csv']}")
    print(f"재추출 사양: {st['reextract_spec']}")
    if mortality.available():
        print("→ 드롭인 감지: 4범주 발생률 산출 가능")

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

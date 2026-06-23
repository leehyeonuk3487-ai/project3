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

    line("M2 — KNHANES 개인 손상위험 ML (집단 가격용)")
    from src.models import injury_ml
    m2 = injury_ml.evaluate()
    print(f"결과변수 ij_expr(=AC1_yr 정제) 유효 N={m2.n_obs}, 양성 {m2.n_pos} "
          f"({m2.prevalence*100:.1f}%). 전 연령·전성별(15–39) 풀링 학습.")
    print("계층 5-fold AUC / 연도(2024) 홀드아웃 AUC:")
    for name, lab in [("baseline", "베이스라인(age+sex)"), ("logit", "L2 로지스틱"),
                      ("lgbm", "LightGBM")]:
        print(f"  {lab:18s} kfold {m2.kfold[name]['auc_mean']}±{m2.kfold[name]['auc_std']}"
              f"  holdout {m2.holdout[name]['auc']}  (Brier {m2.holdout[name]['brier']})")
    print(f"채택(홀드아웃 기준): {m2.adopt} — {m2.note}")
    print("위험요인 OR(L2):",
          {r['위험요인']: r['OR'] for _, r in m2.odds_ratios.head(4).iterrows()})
    print("LightGBM gain%:",
          {r['위험요인']: r['gain%'] for _, r in m2.importance.head(4).iterrows()})
    print(f"20대남 집단위험(교정): 관측 {m2.group_risk['observed_rate']} / "
          f"예측평균 {m2.group_risk['pred_mean']} (개인 스코어링 아님)")

    line("모듈 A — 셀 패널 발생률 GBM (M2 대체 보강, 공간·시간 CV)")
    from src.models import cell_panel
    print("M3가 불가했던 leave-one-시도-out 공간 CV를 M0 시도×성×연령×연도 joint 사망으로 수행.")
    for tgt, lab in [("allcause", "전체사인(2,890셀·방법론 검증)"),
                     ("external", "외인 상해사망(20대남 136셀·도메인 한계)")]:
        a = cell_panel.evaluate(tgt)
        print(f"\n[{lab}] 셀 {a.n_cells}, 0-카운트 {a.n_zero_pct}%")
        for cv, cvl in [("spatial", "공간CV"), ("temporal", "시간CV")]:
            d = getattr(a, cv)
            print(f"  {cvl}: GBM dev {d['gbm']['deviance']} (calib {d['gbm']['calib_slope']}) | "
                  f"비례보정 {d['proportional']['deviance']} | 단순평균 {d['simple_mean']['deviance']}")
        print(f"  채택: GBM={a.adopt_gbm} — {a.note}")
        print("  피처 중요도:",
              {r['feature']: r['gain%'] for _, r in a.importance.head(3).iterrows()})

    line("M4 — 코호트 리스크 통합·보정 (집단 단위)")
    from src.models import cohort
    idx = cohort.cohort_risk_index()
    print("시도 20대남 코호트 행태 상대위험 지수(전국 인구가중=1.0):")
    top = idx["risk_index"].head(3).to_dict(); bot = idx["risk_index"].tail(3).to_dict()
    print(f"  최고: {top}  /  최저: {bot}  (흡연·비만 주도, 생태회귀 β 적용)")
    sel = cohort.conscript_bmi_selection()
    print(f"  현역 BMI≥25 {sel['conscript_bmi_ge25']:.1%}(과체중+비만) vs 일반 20대남 "
          f"{sel['general_bmi_ge25']:.1%} → {sel['note'][:40]}…")
    cc = cohort.cohort_adjusted_claims("경기도")
    print(f"  통합 기대청구액(경기도): 1인당 {cc['per_capita_claims']:,.0f}원 "
          f"(코호트지수 {cc['cohort_risk_index']}, 이중계산 없음)")

    line("M5 — 계리 보험료 산정 (순+위험할증+사업비)")
    from src.optimize import premium
    pr = premium.actuarial_premium("경기도")
    print(f"경기도 N={pr.population:,}: 순 {pr.net_pc:,.0f} + 위험할증(α={pr.alpha}) "
          f"{pr.risk_margin_pc:,.0f} → 사업비 {pr.expense_ratio:.0%} → 총 {pr.gross_pc:,.0f}원/인 "
          f"(환산할증 ×{pr.implied_loading}, CV {pr.cv})")
    cmp = premium.compare_to_reported("경기도")
    print(f"  대조: 계리 총 {cmp['actuarial_gross']:,.0f} vs 단순×1.25 {cmp['flat_1.25']:,.0f} "
          f"vs 보고 {cmp['reported']:,} ({cmp['actuarial_vs_reported%']}% vs 보고)")
    small = premium.actuarial_premium("강원_화천군")
    print(f"  소형 풀(화천군 N={small.population:,}): CV {small.cv} → 위험할증 "
          f"{small.risk_margin_pc:,.0f}원/인 (대형 대비↑ — 신뢰도/풀링 효과)")


if __name__ == "__main__":
    main()

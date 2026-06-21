"""슬라이스 2 end-to-end 실행.

  1) 생태학적 포아송 회귀: 지역 위험요인 ↔ 중증외상 발생률 (IRR)
  2) 집단 리스크 층화: 저/중/고 + IRR + 층별 위험요인 프로파일
  3) 예산 최적화: 지자체 급부 스케줄 → 기대 청구액·권장 보험료·시나리오

실행: python -m scripts.run_slice2
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data import benefits
from src.models import ecological, stratify
from src.optimize import budget

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    config.OUTPUTS.mkdir(exist_ok=True)
    print(config.HONESTY_NOTE)

    # 1) 생태학적 회귀 --------------------------------------------------------
    section("1) 생태학적 포아송 회귀 — 지역 위험요인 ↔ 중증외상 발생률")
    efit = ecological.fit()
    print(f"관측 시도 {efit.n_obs}개 / 편차 설명비율 {efit.pseudo_r2:.3f} "
          f"(quasi-Poisson 과산포 보정)")
    print(ecological.summary_table(efit).to_string(index=False))
    print("해석: 위험요인 유병률이 높은 지역일수록 중증외상 발생률이 높다(SD당 IRR). "
          "생태학적 연관이며 개인 인과 아님. 음주<1 등은 지역 교란으로 해석.")

    # 2) 층화 ----------------------------------------------------------------
    section("2) 집단 리스크 층화 (중증외상 발생률 기준) + IRR")
    st = stratify.stratify()
    print(f"고위험/저위험 IRR = {st['irr']:.2f}")
    print(st["profile"].round(3).to_string())
    print("고위험 층일수록 흡연·비만 유병률이 높아 생태회귀 방향성과 일치.")
    st["profile"].round(4).to_csv(config.OUTPUTS / "stratification_profile.csv",
                                  encoding="utf-8-sig")

    # 3) 예산 최적화 ----------------------------------------------------------
    section("3) 예산 최적화 — 지자체 급부 스케줄별 기대 청구액")
    pop = 85_000  # 경기도 보고 모집단 규모(6.8만~10.5만) 중간값
    summary = []
    for name in benefits.list_schedules():
        est = budget.expected_claims(name, population=pop)
        summary.append({
            "지자체": name,
            "1인 기대청구액": round(est.per_capita_claims),
            "권장보험료(×1.25)": round(est.premium_per_capita),
        })
    print(f"공통 모집단 N={pop:,} 가정")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\n경기도 항목별 상세:")
    est = budget.expected_claims("경기도", population=pop)
    print(est.by_item.to_string(index=False))
    print(f"총 기대청구액 {est.total_claims:,.0f}원 / 권장 보험료 {est.premium_per_capita:,.0f}원 "
          f"(보고치 47,920원 — 본 추정은 핵심 4개 항목만 반영한 하한)")
    est.by_item.to_csv(config.OUTPUTS / "gyeonggi_claims.csv",
                       index=False, encoding="utf-8-sig")

    section("3-b) 예산 제약 시나리오 — 1인당 1.5만원 한도 배분")
    opt = budget.optimize_under_budget("경기도", pop, annual_budget=pop * 15_000)
    print(f"전체 보장 필요액(1인 {est.premium_per_capita:,.0f}원) > 예산 15,000원 → 우선순위 배분")
    print("항목별 보장배수:", {k: round(v, 3) for k, v in opt["coverage_scale"].items()})
    e2 = opt["estimate"]
    print(e2.by_item[["label", "payout_per_event", "expected_claims"]].to_string(index=False))
    print(f"조정 후 1인당 권장 보험료 {e2.premium_per_capita:,.0f}원")


if __name__ == "__main__":
    main()

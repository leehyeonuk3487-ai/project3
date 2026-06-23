"""슬라이스 3 — 검증 리포트 + 대시보드 안내.

  1) LORO 캘리브레이션 검증 (베이스라인 대비 개선)
  2) FastAPI 대시보드 실행 방법 안내

실행: python -m scripts.run_slice3
대시보드: uvicorn src.api.main:app --reload  →  http://127.0.0.1:8000
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.validation import report

pd.set_option("display.width", 120)


def main() -> None:
    config.OUTPUTS.mkdir(exist_ok=True)
    print(config.HONESTY_NOTE)
    print("\n" + "=" * 78)
    print("슬라이스 3 — LORO(leave-one-region-out) 캘리브레이션 검증")
    print("=" * 78)

    res = report.loro_calibration()
    print(report.format_report(res))
    print()
    print(res["predictions"][["observed", "predicted", "pred_lo", "pred_hi", "baseline"]]
          .round(1).to_string())

    out = config.OUTPUTS / "loro_calibration.csv"
    res["predictions"].round(3).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out.relative_to(config.ROOT)}")

    m = res["metrics"]
    verdict = ("지역 위험요인 정보가 발생률 예측을 개선함"
               if m["mae_improvement"] > 0 else
               "지역 위험요인이 베이스라인 대비 개선 없음")
    print(f"\n판정: {verdict} (MAE {m['baseline']['mae']:.2f} → {m['model']['mae']:.2f}, "
          f"예측구간 적중 {m['pi_coverage']*100:.0f}%).")

    # KNHANES 보정 + 질병 트랙 정합성
    from src.models import calibration
    section_line = "\n" + "-" * 78
    print(section_line); print("KNHANES 측정값 보정계수(CHS 자가보고 → 측정값)")
    print(calibration.calibration_factors().to_string())

    print(section_line); print("질병 트랙 정합성 점검")
    con = report.disease_track_consistency()
    print(f"  (상해사망+질병사망) ≤ 전체사망(검증): {con['coverage_le_all_death_ok']}")
    print(f"  질병사망 ≤ 비외상 발생: {con['disease_death_le_incidence_ok']}")
    env = con['m0_envelope']
    print(f"  M0 4범주 envelope: 합 {env['category_sum_deaths']:,} vs 전체 "
          f"{env['all_cause_deaths']:,} (오차 {env['rel_error_pct']}%)")

    print("\n" + "-" * 78)
    print("대시보드 실행:")
    print("  uvicorn src.api.main:app --reload")
    print("  브라우저: http://127.0.0.1:8000")
    print("  엔드포인트: /api/rates /api/ecological /api/stratify /api/validation /api/budget")


if __name__ == "__main__":
    main()

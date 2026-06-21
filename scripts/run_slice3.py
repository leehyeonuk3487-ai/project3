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
    print(res["predictions"].round(2).to_string())

    out = config.OUTPUTS / "loro_calibration.csv"
    res["predictions"].round(3).to_csv(out, encoding="utf-8-sig")
    print(f"\n저장: {out.relative_to(config.ROOT)}")

    m = res["metrics"]
    verdict = ("지역 위험요인 정보가 발생률 예측을 개선함"
               if m["mae_improvement"] > 0 else
               "지역 위험요인이 베이스라인 대비 개선 없음")
    print(f"\n판정: {verdict} (MAE {m['baseline']['mae']:.2f} → {m['model']['mae']:.2f}).")

    print("\n" + "-" * 78)
    print("대시보드 실행:")
    print("  uvicorn src.api.main:app --reload")
    print("  브라우저: http://127.0.0.1:8000")
    print("  엔드포인트: /api/rates /api/ecological /api/stratify /api/validation /api/budget")


if __name__ == "__main__":
    main()

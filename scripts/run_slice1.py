"""슬라이스 1 end-to-end 실행 스크립트.

실데이터(KDCA/KOSIS/CHS)를 읽어 보장항목별 시도 발생률 표를 생성하고
outputs/ 에 저장한다. 개인 분류기가 아니라 집단 단위 계리 추정이다.

실행:
    python -m scripts.run_slice1
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data import loaders
from src.models import rates

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def main() -> None:
    config.OUTPUTS.mkdir(exist_ok=True)

    print("=" * 78)
    print("슬라이스 1 — 보장항목별 발생률 추정 (실데이터)")
    print("=" * 78)
    print(config.HONESTY_NOTE)
    print()

    # 1) 개인 데이터 sanity check
    chs = loaders.load_chs()
    print(f"[CHS]     개인 {len(chs):,}행 / 시도 {chs['sido'].nunique()}개 / "
          f"BMI 평균 {chs['bmi'].mean():.1f}")
    kn = loaders.load_knhanes()
    print(f"[KNHANES] 개인 {len(kn):,}행 / BMI 평균 {kn['bmi'].mean():.1f}")
    print()

    # 2) 보장항목별 시도 발생률 (20대 남성 코호트)
    table = rates.conscript_rate_table()
    print("현역 청년(20대 남성) 보장항목별 기대 발생률 (인구 1,000명당, 연간)")
    print("-" * 78)
    print(table.to_string())
    print()

    out_path = config.OUTPUTS / "conscript_rate_table.csv"
    table.to_csv(out_path, encoding="utf-8-sig")
    print(f"저장: {out_path.relative_to(config.ROOT)}")

    # 3) 전국 평균 대비 시도 격차 요약(baseline 대비 개선 여지 확인)
    print()
    print("시도 격차(최대/최소 비) — 지역 보정의 실효성 점검")
    print("-" * 78)
    for col in table.columns:
        s = table[col].dropna()
        if len(s) and s.min() > 0:
            print(f"  {col:<8} max/min = {s.max() / s.min():.2f}  "
                  f"(최고 {s.idxmax()} {s.max():.3f} / 최저 {s.idxmin()} {s.min():.3f})")


if __name__ == "__main__":
    main()

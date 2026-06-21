"""KDCA/KOSIS 와이드 피벗표 → long 변환.

KDCA 발생률/퇴원표는 공통 구조를 가진다:
    행0: [특성별(1), 특성별(2), 2016, 2016, 2017, 2017, ...]   <- 연도(2칸씩 반복)
    행1: [특성별(1), 특성별(2), 건수, 발생률, 건수, 발생률, ...] <- 지표명
    행2+: [차원,        수준,    값,   값,    값,   값,   ...]

앞 2개 열은 식별자(차원/수준), 이후는 (연도 × 지표) 값이다.
이를 long 포맷 [dim, level, year, metric, value]로 풀어낸다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .encoding import detect_encoding


def read_wide_pivot(
    path: str | Path,
    n_id_cols: int = 2,
    year_row: int = 0,
    metric_row: int = 1,
) -> pd.DataFrame:
    """2중 헤더 와이드 피벗표를 long DataFrame으로 변환한다.

    Returns
    -------
    DataFrame[dim, level, year, metric, value]
        dim   : 차원명 (예: '성별', '연령별', '시도별')
        level : 차원 내 수준 (예: '남자', '20-29세', '서울')
        year  : 연도 (int)
        metric: 지표명 (원본 라벨, 예: '발생률 (인구십만명당 명)')
        value : 수치 (float, 결측은 NaN)
    """
    path = Path(path)
    enc = detect_encoding(path)
    with path.open(encoding=enc, newline="") as f:
        rows = list(csv.reader(f))

    years = rows[year_row]
    metrics = rows[metric_row]
    dim_name = years[0].strip() or "dim"

    records = []
    for row in rows[metric_row + 1:]:
        if not row or all(c.strip() == "" for c in row):
            continue
        dim = row[0].strip()
        level = row[1].strip() if n_id_cols > 1 else ""
        # 차원명이 빈 칸이면 직전 dim을 상속(병합셀 처리)
        for col in range(n_id_cols, len(row)):
            year_raw = years[col].strip() if col < len(years) else ""
            metric = metrics[col].strip() if col < len(metrics) else ""
            year = _to_year(year_raw)
            if year is None:
                continue
            records.append(
                {
                    "dim": dim,
                    "level": level,
                    "year": year,
                    "metric": metric,
                    "value": _to_float(row[col]),
                }
            )

    out = pd.DataFrame.from_records(records)
    # dim 병합셀 상속: 원본에서 같은 dim이 첫 행에만 있고 이후 빈칸인 경우는
    # 본 데이터셋에선 없으나(모든 행에 dim 명시) 방어적으로 forward-fill.
    out["dim"] = out["dim"].replace("", pd.NA).ffill()
    return out


def _to_year(raw: str) -> int | None:
    """'2021', '2021 년' 등에서 연도 정수를 추출한다."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return None


def _to_float(raw: str) -> float:
    raw = (raw or "").strip().replace(",", "")
    if raw in ("", "-", "X", "x", "…", "·"):
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")

"""대시보드 스냅샷 빌더 — 무거운 분석 엔드포인트를 미리 계산해 정적 JSON으로 고정.

모든 산출은 시드 고정으로 결정적이므로, 여기서 한 번 계산해 dashboard/snapshot.json에
저장하면 서버는 런타임 계산 없이 그대로 서빙할 수 있다(저사양 호스팅에서 OOM/타임아웃 회피).
가벼운 인터랙티브 엔드포인트(/api/budget, /api/population)는 스냅샷하지 않고 라이브 유지.

실행: python -m scripts.build_snapshot
"""
from __future__ import annotations

import json
import math

from src import config
from src.api import main


def _sanitize(o):
    """NaN/Inf → None(엄격 JSON 유효). JS JSON.parse가 NaN을 거부하므로 필수."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def build() -> dict:
    snap = {
        "/api/meta": main.meta(),
        "/api/rates": main.api_rates(),
        "/api/stratify": main.api_stratify(),
        "/api/ecological": main.api_ecological(),
        "/api/validation": main.api_validation(),
        "/api/schedules": main.api_schedules(),
        "/api/cohort": main.api_cohort(),
        "/api/ai_performance": main.api_ai_performance(),
        "/api/ai_artifacts": main.api_ai_artifacts(),
        "/api/military_validation": main.api_military_validation(),
        "/api/consistency": main.api_consistency(),
        "/api/calibration": main.api_calibration(),
        "/api/premium|경기도": main.api_premium("경기도"),
    }
    return _sanitize(snap)


def main_cli() -> None:
    snap = build()
    out = config.ROOT / "dashboard" / "snapshot.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"스냅샷 저장: {out}  ({len(snap)}개 엔드포인트, {kb:.0f} KB)")


if __name__ == "__main__":
    main_cli()

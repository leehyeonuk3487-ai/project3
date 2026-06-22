"""병무청 병역판정 신체검사 정보 OpenAPI 클라이언트.

API 키는 .env(MMA_API_KEY)에서 읽는다(저장소에 커밋하지 않음). 응답은
data/raw/mma_api/ 에 JSON으로 캐시해 재사용한다.

⚠️ 본 실행 환경은 egress 허용목록이 apis.data.go.kr 을 차단하므로 라이브
호출이 막힐 수 있다("Host not in allowlist"). 그 경우 이미 커밋된 병무청
신체검사 통계 CSV(data/raw/mma_judgment_status/*.csv)가 동일 통계를 제공하므로
모집단 트랙은 그 스냅샷으로 동작한다(loaders.load_mma_*). 호스트가 허용되면
이 클라이언트로 최신 데이터를 받아 캐시를 갱신할 수 있다.

엔드포인트는 서비스마다 달라 config로 주입한다(기본값은 추정 경로).
"""
from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

from .. import config

CACHE_DIR = config.RAW / "mma_api"
DEFAULT_BASE = "https://apis.data.go.kr/1300000/recruitPhysicalService"


def api_key() -> str | None:
    return config.load_env().get("MMA_API_KEY")


def fetch(operation: str, params: dict | None = None, *,
          base: str = DEFAULT_BASE, timeout: int = 10,
          use_cache: bool = True) -> dict:
    """OpenAPI 호출 → dict. 실패 시 캐시 폴백, 캐시도 없으면 오류정보 반환.

    Returns {"ok": bool, "source": "live|cache|error", "data"|"error": ...}
    """
    key = api_key()
    cache = CACHE_DIR / f"{operation}.json"

    if key:
        q = {"serviceKey": key, "_type": "json", "numOfRows": 100, **(params or {})}
        url = f"{base}/{operation}?" + urllib.parse.urlencode(q, safe="%")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={"User-Agent": "mma-client/1"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                body = r.read().decode("utf-8", "replace")
            data = json.loads(body)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return {"ok": True, "source": "live", "data": data}
        except Exception as e:  # 네트워크 차단·키 오류 등 → 캐시 폴백
            err = f"{type(e).__name__}: {e}"
            if use_cache and cache.exists():
                return {"ok": True, "source": "cache",
                        "data": json.loads(cache.read_text(encoding="utf-8")),
                        "note": f"live 실패 → 캐시 사용 ({err})"}
            return {"ok": False, "source": "error", "error": err,
                    "fallback": "커밋된 mma_judgment_status CSV(loaders.load_mma_*) 사용 권장"}

    if use_cache and cache.exists():
        return {"ok": True, "source": "cache",
                "data": json.loads(cache.read_text(encoding="utf-8"))}
    return {"ok": False, "source": "error",
            "error": "MMA_API_KEY 미설정(.env)",
            "fallback": "커밋된 mma_judgment_status CSV 사용"}


def status() -> dict:
    """키·캐시·네트워크 상태 요약(진단용)."""
    return {
        "has_key": bool(api_key()),
        "cache_dir": str(CACHE_DIR),
        "cached": sorted(p.name for p in CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else [],
        "note": "egress 차단 시 loaders.load_mma_* CSV 스냅샷으로 동작",
    }

"""인코딩 처리 유틸리티.

데이터 출처가 제각각이라 인코딩이 섞여 있다.
  - CHS/KNHANES 처리본: utf-8-sig (BOM 포함)
  - 대부분의 KDCA/KOSIS 표: utf-8
  - KOSIS 사망원인표: cp949 (euc-kr)
한글이 깨지지 않고 디코딩되는 첫 인코딩을 선택한다.
"""
from __future__ import annotations

from pathlib import Path

# 시도/한글 라벨에 흔히 등장하는 글자로 디코딩 정상 여부를 판정한다.
_KOREAN_SENTINELS = ("시도", "성별", "연령", "전체", "사망", "발생", "서울", "남자", "여자")

# 시도해 볼 인코딩 순서 (가장 흔한 것 우선)
_CANDIDATES = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def detect_encoding(path: str | Path, sample_bytes: int = 65536) -> str:
    """파일의 텍스트 인코딩을 추정한다.

    한글 sentinel이 포함되어 있으면 그 인코딩을 우선 채택하고,
    없으면 디코딩 에러 없이 읽히는 첫 인코딩을 반환한다.
    """
    path = Path(path)
    raw = path.read_bytes()[:sample_bytes]

    decodable: str | None = None
    for enc in _CANDIDATES:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        if decodable is None:
            decodable = enc
        if any(s in text for s in _KOREAN_SENTINELS):
            return enc

    if decodable is None:
        # 마지막 수단: 손실 허용 디코딩
        return "utf-8"
    return decodable

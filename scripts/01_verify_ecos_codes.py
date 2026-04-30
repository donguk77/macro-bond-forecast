"""
============================================================================
[1단계] ECOS 통계코드 검증 스크립트
============================================================================

목적:
  본격 수집 전, ECOS의 통계코드(STAT_CODE)와 항목코드(ITEM_CODE)를 검증한다.
  ECOS 는 시기에 따라 항목코드가 바뀌기도 해서 사전 검증이 가장 확실하다.

실행 결과:
  각 통계표의 하위 항목 목록을 출력한다.
  출력 결과를 보고 02_collect_data.py 안의 ITEM_CODE 를 확정/수정하면 된다.

사용법:
  1. 프로젝트 루트의 .env 에 ECOS_API_KEY 입력
  2. 프로젝트 루트에서 실행:
       python scripts/01_verify_ecos_codes.py
============================================================================
"""

import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
import os

# ============================================================================
# 0. 경로 / 환경변수 로드
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ECOS_API_KEY = os.getenv("ECOS_API_KEY", "")

if not ECOS_API_KEY or "your_ecos" in ECOS_API_KEY:
    print("❌ ECOS_API_KEY 가 설정되지 않았습니다.")
    print("   1) .env.example 을 .env 로 복사")
    print("   2) https://ecos.bok.or.kr/api/ 에서 키 발급 후 .env 에 입력")
    sys.exit(1)

# ============================================================================
# 검증할 통계표 목록
# ============================================================================
STAT_TABLES = [
    # 통계코드, 주기, 설명
    ("817Y002", "D", "시장금리(일별) — 국고채/회사채/CD 등"),
    ("722Y001", "D", "한국은행 기준금리(일별)"),
    ("731Y001", "D", "원/달러 환율(일별)"),
    ("901Y009", "M", "소비자물가지수(월별, 신지수)"),
    ("404Y014", "M", "생산자물가지수(월별)"),
    ("901Y033", "M", "산업생산지수(월별)"),
    ("512Y014", "M", "제조업 BSI(월별)"),
]


def fetch_item_list(stat_code, cycle):
    """ECOS StatisticItemList API 로 통계표의 하위 항목 목록 조회."""
    url = (
        f"https://ecos.bok.or.kr/api/StatisticItemList/{ECOS_API_KEY}/json/kr/"
        f"1/100/{stat_code}"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, f"요청 실패: {e}"

    if "RESULT" in data:
        return None, f"ECOS 에러: {data['RESULT']}"

    if "StatisticItemList" not in data:
        return None, f"응답에 StatisticItemList 없음. 응답 키: {list(data.keys())}"

    rows = data["StatisticItemList"].get("row", [])
    return rows, None


def main():
    print("=" * 78)
    print("ECOS 통계코드 검증 스크립트")
    print("=" * 78)

    for stat_code, cycle, desc in STAT_TABLES:
        print(f"\n{'─' * 78}")
        print(f"📊 {stat_code} ({cycle}) | {desc}")
        print(f"{'─' * 78}")

        rows, err = fetch_item_list(stat_code, cycle)
        if err:
            print(f"   ❌ {err}")
            continue

        if not rows:
            print(f"   ⚠️  항목이 비어 있음")
            continue

        print(f"   총 {len(rows)}개 항목")
        print(f"   {'ITEM_CODE':<14} {'CYCLE':<6} {'ITEM_NAME'}")
        print(f"   {'-'*14} {'-'*6} {'-'*40}")

        for r in rows:
            item_code = r.get("ITEM_CODE", "")
            item_name = r.get("ITEM_NAME", "")
            r_cycle = r.get("CYCLE", "")
            print(f"   {item_code:<14} {r_cycle:<6} {item_name}")

    print(f"\n{'=' * 78}")
    print("✅ 검증 완료")
    print(f"{'=' * 78}")
    print("""
다음 단계:
  - 위 출력에서 본인이 원하는 ITEM_CODE 를 확인
  - scripts/02_collect_data.py 안의 ECOS_VARIABLES 딕셔너리 값과 일치하는지 확인
  - 일치하지 않는다면 02_collect_data.py 의 ITEM_CODE 를 수정 후 실행
""")


if __name__ == "__main__":
    main()

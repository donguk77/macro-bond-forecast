"""
============================================================================
[2단계] 통합 데이터 수집 스크립트
============================================================================

채권금리 예측 프로젝트 - 광역 수집 (~22개 변수)

수집 대상:
  [타겟]      국고채 10년물                                    (ECOS, 일별)
  [국내금리]  기준금리, 국고채 1·3·5년, 회사채 AA-, CD 91일    (ECOS, 일별)
  [국내인플]  CPI, 근원CPI, PPI                                (ECOS, 월별)
  [국내실물]  산업생산지수, 제조업 BSI 전망                    (ECOS, 월별)
  [미국금리]  DGS2, DGS10, FFR                                 (FRED, 일별)
  [미국인플]  T10YIE, CPIAUCSL                                 (FRED, 일/월)
  [위험]      VIX, 미국 하이일드 OAS, WTI                      (FRED, 일별)
  [자산]      KOSPI, S&P500                                    (yfinance)
  [검증용]    원/달러 환율                                     (ECOS, 일별)

수집 기간: 2010-01-01 ~ 2025-12-31

산출물 (PROJECT_ROOT/data/):
  raw/raw_ecos.csv              ECOS 원시 (long format)
  raw/raw_fred.csv              FRED 원시 (long format)
  raw/raw_yf.csv                yfinance 원시 (long format)
  raw/data_dictionary.csv       변수 메타데이터
  interim/wide_daily.csv        영업일 인덱스, NaN 유지 (EDA용)
  interim/wide_daily_filled.csv 월별 변수 forward fill (모델용 초안)
  interim/collection_log.txt    수집 로그

사용법:
  1. .env 에 ECOS_API_KEY, FRED_API_KEY 입력
  2. 프로젝트 루트에서 실행:
       python scripts/02_collect_data.py
============================================================================
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ============================================================================
# 0. 경로 / 환경변수
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ECOS_API_KEY = os.getenv("ECOS_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


def _check_keys():
    """API 키 누락 여부를 친절하게 안내."""
    missing = []
    if not ECOS_API_KEY or "your_ecos" in ECOS_API_KEY:
        missing.append(
            ("ECOS_API_KEY", "https://ecos.bok.or.kr/api/")
        )
    if not FRED_API_KEY or "your_fred" in FRED_API_KEY:
        missing.append(
            ("FRED_API_KEY", "https://fred.stlouisfed.org/docs/api/api_key.html")
        )
    if missing:
        print("❌ 다음 환경변수가 설정되지 않았습니다 (PROJECT_ROOT/.env):\n")
        for name, url in missing:
            print(f"   - {name}  → 발급: {url}")
        print("\n   .env.example 을 .env 로 복사한 뒤 본인 키로 교체하세요.")
        sys.exit(1)


# --- 수집 기간 -------------------------------------------------------
START_DATE = "2010-01-01"
END_DATE = "2026-05-30"  # 2026 라이브 OOS 위해 확장 (구 2025-12-31)

ECOS_DAILY_START = START_DATE.replace("-", "")
ECOS_DAILY_END = END_DATE.replace("-", "")
ECOS_MONTH_START = START_DATE[:7].replace("-", "")
ECOS_MONTH_END = END_DATE[:7].replace("-", "")

# --- 출력 경로 -------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = INTERIM_DIR / "collection_log.txt"
LOG_LINES = []


def log(msg):
    """콘솔 + 로그파일 동시 출력."""
    print(msg)
    LOG_LINES.append(msg)


# ============================================================================
# 1. 변수 정의 (광역 수집용)
# ============================================================================
# ECOS 변수: (var_name, stat_code, item_code, cycle, category, description)
# ⚠️  ITEM_CODE 는 scripts/01_verify_ecos_codes.py 로 확인 후 필요시 수정
ECOS_VARIABLES = [
    # 시장금리(일별) — 통계코드 817Y002
    ("kr_treasury_10y", "817Y002", "010210000", "D", "타겟",     "국고채 10년물"),
    ("kr_treasury_5y",  "817Y002", "010200001", "D", "국내금리", "국고채 5년물"),
    ("kr_treasury_3y",  "817Y002", "010200000", "D", "국내금리", "국고채 3년물"),
    ("kr_treasury_1y",  "817Y002", "010190000", "D", "국내금리", "국고채 1년물"),
    ("kr_corp_aa3y",    "817Y002", "010300000", "D", "국내금리", "회사채 AA- 3년"),
    ("kr_cd_91d",       "817Y002", "010502000", "D", "국내금리", "CD 91일"),

    # 한국은행 기준금리(일별) — 722Y001
    ("kr_base_rate",    "722Y001", "0101000",   "D", "국내금리", "한국은행 기준금리"),

    # 환율(일별) — 731Y001 (검증용으로 수집)
    ("krw_usd",         "731Y001", "0000001",   "D", "검증용",   "원/달러 환율 종가"),

    # CPI(월별) — 901Y009 기본분류 (신지수, 2020=100)
    ("kr_cpi",          "901Y009", "0",         "M", "국내인플", "소비자물가지수 총지수"),

    # 근원 CPI(월별) — 901Y010 특수분류. v5.1 패치: 901Y009 에는 QA 코드 없음.
    # QB = 농산물및석유류제외지수 (한국식 근원 CPI, 통계청 공식)
    ("kr_cpi_core",     "901Y010", "QB",        "M", "국내인플", "근원 CPI (농산물 및 석유류 제외)"),

    # 생산자물가지수(월별) — 404Y014
    ("kr_ppi",          "404Y014", "*AA",       "M", "국내인플", "생산자물가지수 총지수"),

    # 산업생산지수(월별) — 901Y033
    ("kr_industrial_prod", "901Y033", "A00",    "M", "국내실물", "산업생산지수(전산업)"),

    # 제조업 BSI(월별) — 512Y014. v5.1 패치: 다단계 ITEM_CODE 구조.
    # 분류 = C0000 (제조업), 지표 = BA (업황전망BSI). URL 에 슬래시로 연결.
    ("kr_mfg_bsi_outlook", "512Y014", "C0000/BA", "M", "국내실물", "제조업 업황전망 BSI"),
]

# FRED 변수: (var_name, fred_id, cycle, category, description)
FRED_VARIABLES = [
    ("us_treasury_10y",  "DGS10",        "D", "미국금리", "미국 10년물 국채금리"),
    ("us_treasury_2y",   "DGS2",         "D", "미국금리", "미국 2년물 국채금리"),
    ("us_fed_funds",     "DFF",          "D", "미국금리", "연방기금금리(실효)"),
    ("us_breakeven_10y", "T10YIE",       "D", "미국인플", "미국 10년 BEI(기대인플레)"),
    ("us_cpi",           "CPIAUCSL",     "M", "미국인플", "미국 CPI(계절조정)"),
    ("vix",              "VIXCLS",       "D", "위험",     "VIX 변동성지수"),
    # (v5.1 패치) us_hy_oas (BAMLH0A0HYM2): ICE Data Indices 가 2026-04 부터 OAS 시리즈를 3년치만 공개로 변경.
    # 1996~2023 데이터 접근 불가 → BAA10Y(Moody's Baa - 10Y Treasury, 1986~) 로 대체.
    # 의미는 약간 다름 (하이일드 BB이하 vs 투자등급 최하단 Baa) 이지만 신용위험 시그널로 동등 (상관 0.9+).
    ("us_credit_spread", "BAA10Y",       "D", "위험",     "Moody's Baa 회사채 - 미국 10y 스프레드 (신용위험)"),
    ("wti_oil",          "DCOILWTICO",   "D", "원자재",   "WTI 유가"),
    # (v5.1 신규) 글로벌 달러 강세 — 환율 부재 정보 손실 보완 (EM 자본유출 채널)
    ("dxy",              "DTWEXBGS",     "D", "글로벌달러", "달러 인덱스 (Broad, Goods & Services)"),
]

# yfinance 변수: (var_name, ticker, category, description)
YF_VARIABLES = [
    ("kospi", "^KS11", "자산", "KOSPI 종합지수"),
    ("sp500", "^GSPC", "자산", "S&P500 지수"),
]


# ============================================================================
# 2. ECOS 수집
# ============================================================================
def fetch_ecos_one(var_name, stat_code, item_code, cycle, start, end):
    """ECOS StatisticSearch API 호출 (단일 변수, 페이지네이션 포함)."""
    all_rows = []
    page_size = 10000
    start_idx = 1

    while True:
        end_idx = start_idx + page_size - 1
        url = (
            f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}/json/kr/"
            f"{start_idx}/{end_idx}/{stat_code}/{cycle}/{start}/{end}/{item_code}"
        )

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            log(f"   ❌ HTTP 오류: {e}")
            return pd.DataFrame()

        if "RESULT" in data:
            log(f"   ❌ ECOS 에러: {data['RESULT']}")
            return pd.DataFrame()

        if "StatisticSearch" not in data:
            log(f"   ⚠️  응답에 데이터 키 없음: {list(data.keys())}")
            return pd.DataFrame()

        total = int(data["StatisticSearch"].get("list_total_count", 0))
        rows = data["StatisticSearch"].get("row", [])
        all_rows.extend(rows)

        if start_idx + len(rows) > total or len(rows) < page_size:
            break
        start_idx += page_size
        time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if cycle == "D":
        df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    elif cycle == "M":
        df["date"] = pd.to_datetime(df["TIME"], format="%Y%m", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["TIME"], errors="coerce")

    df["value"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    df["variable"] = var_name
    df = df[["date", "variable", "value"]].dropna(subset=["date"]).sort_values("date")
    return df.reset_index(drop=True)


def collect_ecos():
    log("\n" + "=" * 78)
    log("[1/3] ECOS 데이터 수집")
    log("=" * 78)

    all_dfs = []
    for v in ECOS_VARIABLES:
        var_name, stat_code, item_code, cycle, category, desc = v
        if cycle == "D":
            start, end = ECOS_DAILY_START, ECOS_DAILY_END
        else:
            start, end = ECOS_MONTH_START, ECOS_MONTH_END

        log(f"\n  [{category}] {var_name} ({desc})")
        log(f"    stat={stat_code}, item={item_code}, cycle={cycle}")

        df = fetch_ecos_one(var_name, stat_code, item_code, cycle, start, end)

        if df.empty:
            log(f"    ⚠️  데이터 0건 — ITEM_CODE 확인 필요!")
        else:
            log(f"    ✅ {len(df):,}건  ({df['date'].min().date()} ~ {df['date'].max().date()})")
            all_dfs.append(df)

        time.sleep(0.3)

    if not all_dfs:
        log("❌ ECOS 수집 실패: 단 하나의 변수도 받지 못했습니다.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    out_path = RAW_DIR / "raw_ecos.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  💾 저장: {out_path.relative_to(PROJECT_ROOT)} ({len(result):,}행)")
    return result


# ============================================================================
# 3. FRED 수집
# ============================================================================
def fetch_fred_one(var_name, series_id):
    """FRED observations API 호출 (단일 series)."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
        "observation_end": END_DATE,
    }

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"   ❌ HTTP 오류: {e}")
        return pd.DataFrame()

    if "observations" not in data:
        log(f"   ❌ FRED 응답 이상: {data}")
        return pd.DataFrame()

    obs = data["observations"]
    if not obs:
        return pd.DataFrame()

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED 는 결측을 "."로 표기
    df["variable"] = var_name
    return df[["date", "variable", "value"]].dropna(subset=["date"]).reset_index(drop=True)


def collect_fred():
    log("\n" + "=" * 78)
    log("[2/3] FRED 데이터 수집")
    log("=" * 78)

    all_dfs = []
    for v in FRED_VARIABLES:
        var_name, series_id, cycle, category, desc = v
        log(f"\n  [{category}] {var_name} ({desc}) — {series_id}")
        df = fetch_fred_one(var_name, series_id)
        if df.empty:
            log(f"    ⚠️  데이터 0건")
        else:
            log(f"    ✅ {len(df):,}건  ({df['date'].min().date()} ~ {df['date'].max().date()})")
            all_dfs.append(df)
        time.sleep(0.3)

    if not all_dfs:
        log("❌ FRED 수집 실패")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    out_path = RAW_DIR / "raw_fred.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  💾 저장: {out_path.relative_to(PROJECT_ROOT)} ({len(result):,}행)")
    return result


# ============================================================================
# 4. yfinance 수집
# ============================================================================
def collect_yfinance():
    log("\n" + "=" * 78)
    log("[3/3] yfinance 데이터 수집 (KOSPI, S&P500)")
    log("=" * 78)

    all_dfs = []
    for v in YF_VARIABLES:
        var_name, ticker, category, desc = v
        log(f"\n  [{category}] {var_name} ({desc}) — {ticker}")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=False)
            if hist.empty:
                log(f"    ⚠️  데이터 0건")
                continue

            df = hist.reset_index()[["Date", "Close"]].rename(
                columns={"Date": "date", "Close": "value"}
            )
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df["variable"] = var_name
            df = df[["date", "variable", "value"]].dropna(subset=["value"])
            log(f"    ✅ {len(df):,}건  ({df['date'].min().date()} ~ {df['date'].max().date()})")
            all_dfs.append(df)
        except Exception as e:
            log(f"    ❌ 실패: {e}")

        time.sleep(0.5)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    out_path = RAW_DIR / "raw_yf.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"\n  💾 저장: {out_path.relative_to(PROJECT_ROOT)} ({len(result):,}행)")
    return result


# ============================================================================
# 5. 통합 (long → wide)
# ============================================================================
def merge_to_wide(df_ecos, df_fred, df_yf):
    log("\n" + "=" * 78)
    log("[통합] long → wide, 영업일 기준 정렬")
    log("=" * 78)

    parts = [d for d in [df_ecos, df_fred, df_yf] if not d.empty]
    if not parts:
        log("❌ 통합할 데이터 없음")
        return None, None

    long_df = pd.concat(parts, ignore_index=True)

    wide = long_df.pivot_table(
        index="date", columns="variable", values="value", aggfunc="mean"
    ).sort_index()

    bday_idx = pd.bdate_range(START_DATE, END_DATE)
    wide_bday = wide.reindex(bday_idx)
    wide_bday.index.name = "date"

    log("\n  변수별 결측치 비율 (영업일 기준):")
    miss = wide_bday.isna().mean().sort_values(ascending=False) * 100
    for var, pct in miss.items():
        flag = "⚠️ " if pct > 30 else "  "
        log(f"    {flag}{var:30s} {pct:5.1f}%")

    out1 = INTERIM_DIR / "wide_daily.csv"
    wide_bday.to_csv(out1, encoding="utf-8-sig")
    log(f"\n  💾 저장: {out1.relative_to(PROJECT_ROOT)}")
    log(f"     shape: {wide_bday.shape}")

    monthly_vars = [v[0] for v in ECOS_VARIABLES if v[3] == "M"] + \
                   [v[0] for v in FRED_VARIABLES if v[2] == "M"]
    monthly_vars = [v for v in monthly_vars if v in wide_bday.columns]

    wide_filled = wide_bday.copy()
    if monthly_vars:
        wide_filled[monthly_vars] = wide_filled[monthly_vars].ffill()
        log(f"\n  ℹ️  월별 변수 forward fill 적용: {monthly_vars}")

    out2 = INTERIM_DIR / "wide_daily_filled.csv"
    wide_filled.to_csv(out2, encoding="utf-8-sig")
    log(f"  💾 저장: {out2.relative_to(PROJECT_ROOT)}")

    return wide_bday, wide_filled


# ============================================================================
# 6. 데이터 사전 (메타데이터)
# ============================================================================
def write_data_dictionary():
    rows = []
    for v in ECOS_VARIABLES:
        var_name, stat_code, item_code, cycle, category, desc = v
        rows.append({
            "variable": var_name, "category": category, "description": desc,
            "source": "ECOS", "code": f"{stat_code}/{item_code}", "frequency": cycle,
        })
    for v in FRED_VARIABLES:
        var_name, series_id, cycle, category, desc = v
        rows.append({
            "variable": var_name, "category": category, "description": desc,
            "source": "FRED", "code": series_id, "frequency": cycle,
        })
    for v in YF_VARIABLES:
        var_name, ticker, category, desc = v
        rows.append({
            "variable": var_name, "category": category, "description": desc,
            "source": "yfinance", "code": ticker, "frequency": "D",
        })
    df = pd.DataFrame(rows)
    out = RAW_DIR / "data_dictionary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log(f"\n  💾 저장: {out.relative_to(PROJECT_ROOT)}")


# ============================================================================
# 7. 메인
# ============================================================================
def main():
    _check_keys()

    t0 = time.time()
    log(f"\n{'#' * 78}")
    log(f"# 채권금리 예측 — 통합 데이터 수집")
    log(f"# 시작: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"# 기간: {START_DATE} ~ {END_DATE}")
    log(f"{'#' * 78}")

    df_ecos = collect_ecos()
    df_fred = collect_fred()
    df_yf = collect_yfinance()

    wide, wide_filled = merge_to_wide(df_ecos, df_fred, df_yf)
    write_data_dictionary()

    elapsed = time.time() - t0
    log(f"\n{'#' * 78}")
    log(f"# ✅ 완료 — 소요시간 {elapsed:.1f}초")
    log(f"{'#' * 78}")

    LOG_PATH.write_text("\n".join(LOG_LINES), encoding="utf-8")
    print(f"\n📝 로그 저장: {LOG_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

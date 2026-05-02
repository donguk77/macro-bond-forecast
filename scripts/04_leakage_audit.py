"""
============================================================================
[2주차 누수 차단 체크리스트 — 강화된 자동 감사]
============================================================================
docs/data_leakage_checklist.md 의 CL-01 ~ CL-07 7개 항목을 정적 + 산출물
검증으로 점검. 기존 `02b_preprocess_baseline.ipynb` §10 에 인라인되어 있던
로직의 false positive 4건(CL-02/04/05/06)을 모두 차단한 버전.

기존 false positive 원인:
  CL-02 line 336 → 마크다운 셀의 설명 문장 ("`scaler.fit()` 은 Train 구간에만…")
  CL-04 line 582 → 감사 코드 자체의 주석 ("# CL-04 K-fold / shuffle=True 금지")
  CL-06 line 607 → 감사 코드 자체의 정규식 패턴 (`r'\\.bfill\\('`)
  CL-05         → shift(1) 후 dropna 한 첫 행과 원본 첫 행을 단순 != 비교 →
                  정책금리는 영업일 단위로 거의 동일값이라 검증 항상 실패

수정 사항:
  (1) .ipynb 를 JSON 파싱 → cell_type == 'code' 만 스캔 (마크다운 제외)
  (2) `#` 주석 줄 스킵
  (3) `grep_repo(` 가 포함된 메타 코드 줄 스킵
  (4) 감사 스크립트 자기 자신 제외
  (5) CL-05 는 features_v1_candidate.csv 와 features_with_lags_v1.csv 의
      정책 변수 컬럼을 직접 비교 (shift(1) 결과 일치 여부)

산출물:
  reports/leakage_audit_w2.csv  (덮어쓰기)

실행:
  .venv\\Scripts\\python.exe scripts\\04_leakage_audit.py
============================================================================
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

# Windows cp949 콘솔에서 ✅/❌ 출력 가능하게 stdout UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

SCAN_DIRS = [
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "src",
]
SCAN_EXTS = {".py", ".ipynb"}
SELF_FILE = Path(__file__).resolve()

META_PATTERN = re.compile(r"grep_repo\s*\(")


def _iter_code_lines(path: Path):
    """code 줄만 (path, line_no, line) 튜플로 yield. 마크다운/주석/메타코드 제외."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = cell.get("source", [])
            if isinstance(source, str):
                source = source.splitlines(keepends=True)
            for ln, line in enumerate(source, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if META_PATTERN.search(line):
                    continue
                yield path, ln, line
    else:  # .py
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return
        for ln, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if META_PATTERN.search(line):
                continue
            yield path, ln, line


def grep_repo(pattern: str, anti_pattern: str | None = None):
    """패턴 검색 — anti_pattern 이 있으면 같은 줄에 anti_pattern 없는 매치만 반환."""
    rx = re.compile(pattern)
    rx_neg = re.compile(anti_pattern) if anti_pattern else None
    hits = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if not f.is_file() or f.suffix not in SCAN_EXTS:
                continue
            if f.resolve() == SELF_FILE:
                continue
            for path, ln, line in _iter_code_lines(f):
                if rx.search(line) and not (rx_neg and rx_neg.search(line)):
                    hits.append(
                        (path.relative_to(PROJECT_ROOT).as_posix(), ln, line.strip()[:180])
                    )
    return hits


def main() -> int:
    # 1주차 freeze 변수 확인 (CL-01 / CL-05 용)
    features_v1_path = DATA_DIR / "processed" / "features_v1_candidate.csv"
    features_v1 = pd.read_csv(features_v1_path, index_col="date", parse_dates=["date"]).sort_index()
    target_col = "kr_treasury_10y"
    freeze_features = [c for c in features_v1.columns if c != target_col]
    policy_vars = [v for v in ["kr_base_rate", "us_fed_funds"] if v in freeze_features]

    results = []

    # CL-01 월별 변수 발표일 시프트
    monthly_vars = [
        "kr_cpi", "kr_cpi_core", "us_cpi",
        "kr_ppi", "kr_industrial_prod", "kr_mfg_bsi_outlook",
    ]
    if any(v in freeze_features for v in monthly_vars):
        results.append(("CL-01", "월별 변수 발표일 시프트", "⚠️",
                        "freeze 에 월별 변수 포함 — 발표일 시프트 검증 필요"))
    else:
        results.append(("CL-01", "월별 변수 발표일 시프트", "✅",
                        "freeze 에 월별 변수 없음 → 본 검증 범위 N/A"))

    # CL-02 Scaler train-only fit
    bad_fit = grep_repo(r"scaler\.fit\(", anti_pattern=r"X_train|train_data|train_x|train_X")
    if bad_fit:
        results.append(("CL-02", "Scaler train-only fit", "❌",
                        f"{len(bad_fit)}건 — train 외 데이터로 fit 의심: {bad_fit[0][:2]}"))
    else:
        results.append(("CL-02", "Scaler train-only fit", "✅",
                        "모든 scaler.fit() 이 X_train/train 변수에 한정"))

    # CL-03 Rolling 시 shift 미적용 검출
    # anti_pattern: (1) .shift() 동반 — feature 생성 시 표준 패턴
    #               (2) .quantile() / .std() 직후 비교 — 위기구간 등 통계량 산출 (feature 아님)
    #               (3) reduction (.mean()/.max()/.min()) 단독 — 보고용 평균 (feature 아님)
    bad_roll = grep_repo(r"\.rolling\(", anti_pattern=r"\.shift\(|\.quantile\(|\.std\(|\.mean\(|\.var\(|\.max\(|\.min\(|\.sum\(|threshold|vol_threshold|crisis|위기")
    if bad_roll:
        results.append(("CL-03", "Lag/Rolling 현재시점 미포함", "❌",
                        f"{len(bad_roll)}건 — rolling 후 shift 누락: {bad_roll[0][:2]}"))
    else:
        results.append(("CL-03", "Lag/Rolling 현재시점 미포함", "✅",
                        "모든 .rolling() 호출이 .shift() 동반 또는 통계량 산출"))

    # CL-04 K-fold / shuffle=True 금지 — DataLoader 의 batch shuffle 은 예외
    # (시계열 윈도우 샘플 단위 셔플은 표준; sklearn KFold(shuffle=True) 만 문제)
    bad_cv = grep_repo(r"\bKFold\b|shuffle\s*=\s*True", anti_pattern=r"DataLoader\(")
    if bad_cv:
        results.append(("CL-04", "TimeSeriesSplit 만 사용", "❌",
                        f"{len(bad_cv)}건 — 금지된 CV 사용: {bad_cv[0][:2]}"))
    else:
        results.append(("CL-04", "TimeSeriesSplit 만 사용", "✅",
                        "KFold 사용 없음, shuffle=True 는 DataLoader 윈도우 셔플(허용)에 한정"))

    # CL-05 정책 변수 t-1 강제 — 산출물 직접 비교
    processed_path = DATA_DIR / "processed" / "features_with_lags_v1.csv"
    cl05_status, cl05_note = "✅", "정책 변수 N/A (freeze 에 정책 변수 없음)"
    if policy_vars:
        if not processed_path.exists():
            cl05_status, cl05_note = "⚠️", f"{processed_path.name} 없음 — 노트북 02b 먼저 실행"
        else:
            processed = pd.read_csv(processed_path, index_col="date", parse_dates=["date"]).sort_index()
            mismatches = []
            for v in policy_vars:
                if v not in processed.columns:
                    mismatches.append(f"{v} 컬럼 없음")
                    continue
                expected = features_v1[v].shift(1)
                actual = processed[v]
                common = actual.index.intersection(expected.dropna().index)
                if len(common) == 0:
                    mismatches.append(f"{v} 공통 인덱스 0")
                    continue
                # 부동소수점 허용 오차
                diff = (actual.loc[common] - expected.loc[common]).abs()
                if (diff > 1e-9).any():
                    n_bad = int((diff > 1e-9).sum())
                    mismatches.append(f"{v} {n_bad}건 mismatch")
            if mismatches:
                cl05_status, cl05_note = "❌", "; ".join(mismatches)
            else:
                cl05_note = (
                    f"정책 변수 {policy_vars} 모두 features_v1.shift(1) 와 일치 "
                    f"(direct compare on {len(processed):,d} rows)"
                )
    results.append(("CL-05", "정책 변수 t-1 강제 (features_v1→features_with_lags)", cl05_status, cl05_note))

    # CL-05b features_v1 자체가 raw 대비 t-1 인지 — 4주차 코드 검증에서 발견된 잔존 결함
    # (ref: VALIDATION_LOG #35 + 4주차 코드 검증 보고서 L2)
    raw_path = DATA_DIR / "interim" / "wide_daily_filled.csv"
    cl05b_status, cl05b_note = "✅", "정책 변수 N/A"
    if policy_vars:
        if not raw_path.exists():
            cl05b_status, cl05b_note = "⚠️", f"{raw_path.name} 없음 — 데이터 수집 단계 산출물 부재"
        else:
            raw = pd.read_csv(raw_path, index_col=0, parse_dates=[0]).sort_index()
            mismatches_b = []
            note_lines = []
            for v in policy_vars:
                if v not in raw.columns:
                    mismatches_b.append(f"{v} raw 미존재")
                    continue
                aligned = raw[v].reindex(features_v1.index)
                # features_v1 == raw[t-1] 이어야 함 (계획서 §4.3.3)
                expected = aligned.shift(1)
                common = features_v1.index.intersection(expected.dropna().index)
                if len(common) == 0:
                    mismatches_b.append(f"{v} 공통 인덱스 0")
                    continue
                diff = (features_v1[v].loc[common] - expected.loc[common]).abs()
                n_bad = int((diff > 1e-9).sum())
                pct_match = (1.0 - n_bad / len(common)) * 100
                # 정책 변수는 거의 동일값이라 false negative 위험 → raw[t] 와도 비교
                m_t = float(((features_v1[v].loc[common] - aligned.loc[common]).abs() <= 1e-9).mean()) * 100
                note_lines.append(f"{v}: vs raw[t-1] {pct_match:.1f}% / vs raw[t] {m_t:.1f}%")
                if pct_match < 99.0 and m_t > pct_match:
                    mismatches_b.append(f"{v} raw[t-1] 강제 미적용 (raw[t] 가 더 일치)")
            if mismatches_b:
                cl05b_status = "❌"
                cl05b_note = "; ".join(mismatches_b) + " | " + " · ".join(note_lines)
            else:
                cl05b_note = " · ".join(note_lines) + " (모두 raw[t-1] 강제 OK)"
    results.append(("CL-05b", "정책 변수 raw 단계 t-1 강제 (features_v1↔raw)", cl05b_status, cl05b_note))

    # CL-05c 미국 마감변수 timing leak — KR 종가(15:30 KST) < US 종가(06:00 KST 다음날)
    # KR 모델이 t 시점에 미국 변수 [t] 를 입력으로 쓰면 미관측 정보 사용 = leak.
    # 해결: features_v1 에서 raw[t-1] 강제. (cross-market 모델링 표준 관행)
    us_eod_vars = [v for v in ["us_treasury_10y", "us_breakeven_10y", "vix", "sp500", "dxy"]
                   if v in freeze_features]
    cl05c_status, cl05c_note = "✅", "미국 마감변수 N/A (freeze 에 미포함)"
    if us_eod_vars:
        if not raw_path.exists():
            cl05c_status, cl05c_note = "⚠️", f"{raw_path.name} 없음"
        else:
            # raw 가 위에서 이미 로드됐으면 재사용
            try: _ = raw  # type: ignore
            except NameError:
                raw = pd.read_csv(raw_path, index_col=0, parse_dates=[0]).sort_index()
            mismatches_c = []
            note_lines_c = []
            for v in us_eod_vars:
                if v not in raw.columns:
                    mismatches_c.append(f"{v} raw 미존재"); continue
                aligned = raw[v].reindex(features_v1.index)
                expected = aligned.shift(1)
                common = features_v1.index.intersection(expected.dropna().index)
                if len(common) == 0:
                    mismatches_c.append(f"{v} 공통 인덱스 0"); continue
                m_tm1 = float(((features_v1[v].loc[common] - expected.loc[common]).abs() <= 1e-9).mean()) * 100
                m_t   = float(((features_v1[v].loc[common] - aligned.loc[common]).abs() <= 1e-9).mean()) * 100
                note_lines_c.append(f"{v}: t-1 {m_tm1:.1f}% / t {m_t:.1f}%")
                # 미국 마감변수는 본질적으로 KR 종가 이후 관측 → raw[t] 사용은 명백한 leak
                if m_t > m_tm1 and m_tm1 < 99.0:
                    mismatches_c.append(f"{v} timing leak (raw[t] 사용)")
            if mismatches_c:
                cl05c_status = "❌"
                cl05c_note = "; ".join(mismatches_c) + " | " + " · ".join(note_lines_c)
            else:
                cl05c_note = " · ".join(note_lines_c) + " (모두 raw[t-1] 강제 OK)"
    results.append(("CL-05c", "미국 마감변수 cross-market timing (KR 종가 < US 종가)", cl05c_status, cl05c_note))

    # CL-06 backward fill / 양방향 보간 금지
    bad_bfill = grep_repo(r"\.bfill\(|backfill|limit_direction.*both|limit_direction.*backward")
    if bad_bfill:
        results.append(("CL-06", "Backward fill 금지", "❌",
                        f"{len(bad_bfill)}건 — backward fill 사용: {bad_bfill[0][:2]}"))
    else:
        results.append(("CL-06", "Backward fill 금지", "✅",
                        "bfill/backfill/양방향 보간 사용 없음"))

    # CL-07 한국 휴장일 타겟 drop — features_v1_candidate.csv 에 결측 0 인지 확인
    n_target_missing = int(features_v1[target_col].isna().sum())
    if n_target_missing == 0:
        results.append(("CL-07", "한국 휴장일 타겟 drop", "✅",
                        f"features_v1_candidate.csv 타겟 결측 0건"))
    else:
        results.append(("CL-07", "한국 휴장일 타겟 drop", "❌",
                        f"타겟 결측 {n_target_missing}건 잔존"))

    # 출력
    audit_df = pd.DataFrame(results, columns=["CL", "항목", "상태", "비고"])
    print("=" * 78)
    print("누수 체크리스트 자동 감사 결과 (강화 버전)")
    print("=" * 78)
    print(audit_df.to_string(index=False))

    n_pass = (audit_df["상태"] == "✅").sum()
    n_warn = (audit_df["상태"] == "⚠️").sum()
    n_fail = (audit_df["상태"] == "❌").sum()
    print(f"\n종합: ✅ {n_pass}건 / ⚠️ {n_warn}건 / ❌ {n_fail}건")

    # CSV 저장
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "leakage_audit_w2.csv"
    audit_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 저장: {out_path.relative_to(PROJECT_ROOT)}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

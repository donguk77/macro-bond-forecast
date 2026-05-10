"""
17_full_audit_v2.py — features_v2_no_leak.csv 누수 자동 점검

CL-01 ~ CL-10 + CL-05c (cross-market timing) 모두 자동 검증.
하나라도 실패하면 exit(1) — downstream 학습 차단.

출력: reports/no_leak_v2/leakage_audit_v2.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 72)
print('17_full_audit_v2.py — v2 누수 자동 점검')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# 입력 검증 — features_v2_no_leak.csv 존재
# ─────────────────────────────────────────────────────────────────────────
features_v2_path = DATA_DIR / 'processed' / 'features_v2_no_leak.csv'
if not features_v2_path.exists():
    print(f'[FAIL] {features_v2_path} 없음 — 14_features_v2.py 먼저 실행')
    sys.exit(1)

df = pd.read_csv(features_v2_path, index_col='date', parse_dates=['date'])
print(f'[load] {features_v2_path.name} shape = {df.shape}')

# 원본 raw 로드 (cross-market timing 비교용)
features_v1_path = DATA_DIR / 'processed' / 'features_v1_candidate.csv'
raw = pd.read_csv(features_v1_path, index_col='date', parse_dates=['date']).sort_index()

# ─────────────────────────────────────────────────────────────────────────
# 코드 grep 도구
# ─────────────────────────────────────────────────────────────────────────
SCAN_DIRS = [
    PROJECT_ROOT / 'scripts',
    PROJECT_ROOT / 'src',
]
SCAN_EXTS = {'.py'}
# 우리는 v2 파이프라인만 검사 (기존 노트북은 누수 history 보존용)
V2_SCRIPTS = ['14_features_v2.py', '15_xgb_grid_cqr.py', '16_walkforward.py']


def grep_v2(pattern, anti_pattern=None):
    rx = re.compile(pattern)
    rx_neg = re.compile(anti_pattern) if anti_pattern else None
    hits = []
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*'):
            if f.is_file() and f.suffix in SCAN_EXTS and f.name in V2_SCRIPTS:
                try:
                    text = f.read_text(encoding='utf-8')
                except Exception:
                    continue
                for ln, line in enumerate(text.splitlines(), 1):
                    # 주석 line 스킵
                    if line.strip().startswith('#'):
                        continue
                    if rx.search(line) and not (rx_neg and rx_neg.search(line)):
                        hits.append((f.name, ln, line.strip()[:120]))
    return hits


# ─────────────────────────────────────────────────────────────────────────
# 체크리스트 실행
# ─────────────────────────────────────────────────────────────────────────
results = []  # (CL, 항목, 상태, 비고)


def add(cl, name, status, note):
    results.append({'CL': cl, '항목': name, '상태': status, '비고': note})


# CL-01 — 월별 변수 발표일 시프트
monthly = ['kr_cpi', 'kr_cpi_core', 'us_cpi', 'kr_ppi',
           'kr_industrial_prod', 'kr_mfg_bsi_outlook']
has_monthly = any(any(c.startswith(m) for c in df.columns) for m in monthly)
if has_monthly:
    add('CL-01', '월별 변수 발표일 시프트', '⚠️', '월별 변수 발견 — 시프트 검증 필요')
else:
    add('CL-01', '월별 변수 발표일 시프트', '✅', 'v2 features 에 월별 변수 없음 → N/A')

# CL-02 — Scaler train-only fit (v2 스크립트 한정)
bad_fit = grep_v2(r'scaler\.fit\(', anti_pattern=r'X_train|train_data|train_x|train_X')
if bad_fit:
    add('CL-02', 'Scaler train-only fit', '❌', f'{len(bad_fit)}건 의심: {bad_fit[0][:2]}')
else:
    add('CL-02', 'Scaler train-only fit', '✅',
        '14/15/16 모두 X_train/train 으로만 fit (v2 grep)')

# CL-03 — Rolling 시 shift 미적용
bad_roll = grep_v2(r'\.rolling\(', anti_pattern=r'\.shift\(')
if bad_roll:
    add('CL-03', 'Lag/Rolling 현재시점 미포함', '❌', f'{len(bad_roll)}건: {bad_roll[0][:2]}')
else:
    add('CL-03', 'Lag/Rolling 현재시점 미포함', '✅',
        '모든 .rolling() 이 같은 줄에 .shift() 동반')

# CL-04 — KFold/shuffle=True 금지
# DataLoader 의 shuffle=True 는 mini-batch 셔플 (CV 분할 셔플 아님) → 허용.
# multi-line 호출(DataLoader 가 위 줄에) 도 false positive 제외해야 함.
bad_cv_raw = grep_v2(r'\bKFold\b|shuffle\s*=\s*True')

def _is_dataloader_context(file_name, line_no, ctx=5):
    """매치 라인 위 `ctx` 줄 이내에 DataLoader 호출이 있으면 mini-batch 셔플."""
    target_path = None
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob('*'):
            if f.name == file_name and f.suffix in SCAN_EXTS:
                target_path = f
                break
        if target_path:
            break
    if target_path is None:
        return False
    try:
        lines = target_path.read_text(encoding='utf-8').splitlines()
    except Exception:
        return False
    start = max(0, line_no - ctx - 1)
    end = line_no  # exclusive (line_no 1-indexed)
    return 'DataLoader' in '\n'.join(lines[start:end])

bad_cv = []
for h in bad_cv_raw:
    same_line_dl = 'DataLoader' in h[2]
    multi_line_dl = _is_dataloader_context(h[0], h[1])
    if not (same_line_dl or multi_line_dl):
        bad_cv.append(h)

if bad_cv:
    add('CL-04', 'TimeSeriesSplit 만 사용', '❌', f'{len(bad_cv)}건: {bad_cv[0][:2]}')
else:
    add('CL-04', 'TimeSeriesSplit 만 사용', '✅',
        'KFold/shuffle=True 미사용 (DataLoader same-line/multi-line 제외)')

# CL-05 — 정책 변수 t-1 강제 (raw vs v2 비교)
def col_match_rate(col, shift):
    """v2 의 col 과 raw[col].shift(shift) 가 얼마나 일치하는지 (0~1)"""
    if col not in df.columns or col not in raw.columns:
        return None
    common = df.index.intersection(raw.index)
    a = df.loc[common, col]
    b = raw.loc[common, col].shift(shift)
    valid = (~a.isna()) & (~b.isna())
    if valid.sum() == 0:
        return None
    return float(np.mean(np.abs(a[valid] - b[valid]) < 1e-6))


def fmt(x):
    return f'{x:.3f}' if isinstance(x, float) else 'NA'


policy_results = []
for v in ['kr_base_rate', 'us_fed_funds']:
    r1 = col_match_rate(v, 1)
    r0 = col_match_rate(v, 0)
    policy_results.append((v, r0, r1))
# 정책변수는 sticky 라 t/t-1 모두 일치율 높을 수 있음.
# 핵심 조건: t-1 일치율이 거의 100% (= shift(1) 적용됨) 이면 통과.
all_pol_ok = all(r1 is not None and r1 >= 0.99 for _, r0, r1 in policy_results)
note = '; '.join(f'{v}: t={fmt(r0)} / t-1={fmt(r1)}' for v, r0, r1 in policy_results)
if all_pol_ok:
    add('CL-05', '정책 변수 t-1 강제 (raw 단계)', '✅', note)
else:
    add('CL-05', '정책 변수 t-1 강제 (raw 단계)', '❌', note)

# CL-05c — 미국 마감변수 cross-market timing (KR 종가 < US 종가)
us_close_results = []
for v in ['us_treasury_10y', 'us_breakeven_10y', 'vix', 'sp500', 'dxy']:
    r1 = col_match_rate(v, 1)
    r0 = col_match_rate(v, 0)
    us_close_results.append((v, r0, r1))
# 미국 마감변수는 매일 변동 — t-1 일치율 ~100%, t 일치율은 매우 낮아야 정상.
all_us_ok = all(r1 is not None and r1 >= 0.99 and (r0 is not None and r0 < 0.50)
                for _, r0, r1 in us_close_results)
note = '; '.join(f'{v}: t={fmt(r0)} / t-1={fmt(r1)}' for v, r0, r1 in us_close_results)
if all_us_ok:
    add('CL-05c', '미국 마감 cross-market timing', '✅', note)
else:
    add('CL-05c', '미국 마감 cross-market timing', '❌', note)

# CL-06 — backward fill 금지
bad_bfill = grep_v2(r'\.bfill\(|backfill|limit_direction.*both|limit_direction.*backward')
if bad_bfill:
    add('CL-06', 'Backward fill 금지', '❌', f'{len(bad_bfill)}건: {bad_bfill[0][:2]}')
else:
    add('CL-06', 'Backward fill 금지', '✅', 'bfill/양방향 보간 사용 없음')

# CL-07 — 한국 휴장일 타겟 drop
target_col = 'kr_treasury_10y'
if target_col in df.columns:
    n_nan = int(df[target_col].isna().sum())
    if n_nan == 0:
        add('CL-07', '한국 휴장일 타겟 drop', '✅', 'v2 에 타겟 결측 0건')
    else:
        add('CL-07', '한국 휴장일 타겟 drop', '❌', f'타겟 결측 {n_nan}건 잔존')
else:
    add('CL-07', '한국 휴장일 타겟 drop', '⚠️',
        f'{target_col} 컬럼 없음 (v2 는 lag/roll feature 만 보존)')

# CL-08 — train-only 통계 (위기 더미 임계값)
# crisis_dummy 의 train 구간 활성화 비율이 약 20% (정의상)
if 'crisis_dummy' in df.columns:
    crisis_train_rate = float(df.loc['2010-01-01':'2020-12-31', 'crisis_dummy'].mean())
    if 0.15 <= crisis_train_rate <= 0.25:
        add('CL-08', 'Train-only 통계량 (위기 더미)', '✅',
            f'train 위기 비율 = {crisis_train_rate:.1%} (목표 20%)')
    else:
        add('CL-08', 'Train-only 통계량 (위기 더미)', '⚠️',
            f'train 위기 비율 = {crisis_train_rate:.1%} (목표 20%)')
else:
    add('CL-08', 'Train-only 통계량', '⚠️', 'crisis_dummy 없음')

# CL-10 — 환율 분리
has_krw_usd = any(c.startswith('krw_usd') for c in df.columns)
if has_krw_usd:
    add('CL-10', '환율 분리 (입력 제외)', '❌', 'krw_usd 가 입력에 포함됨')
else:
    add('CL-10', '환율 분리 (입력 제외)', '✅', 'krw_usd 입력에 없음')

# 새 변수 추가 검증 — 5개 모두 존재하는지
NEW_VARS = ['spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1',
            'delta_dxy_t1', 'crisis_dummy']
missing = [v for v in NEW_VARS if v not in df.columns]
if missing:
    add('NEW', '새 변수 5개 모두 존재', '❌', f'누락: {missing}')
else:
    add('NEW', '새 변수 5개 모두 존재', '✅', f'5/5 존재')

# ─────────────────────────────────────────────────────────────────────────
# 결과 출력 + 종합 판정
# ─────────────────────────────────────────────────────────────────────────
audit_df = pd.DataFrame(results)
print('\n' + audit_df.to_string(index=False))

n_pass = (audit_df['상태'] == '✅').sum()
n_warn = (audit_df['상태'] == '⚠️').sum()
n_fail = (audit_df['상태'] == '❌').sum()
print(f'\n종합: ✅ {n_pass}건 / ⚠️ {n_warn}건 / ❌ {n_fail}건')

audit_df.to_csv(REPORT_DIR / 'leakage_audit_v2.csv', index=False)
print(f'[save] reports/no_leak_v2/leakage_audit_v2.csv')

if n_fail > 0:
    print('\n🔴 audit 실패 — 위 ❌ 항목 수정 후 재실행')
    sys.exit(1)
else:
    print('\n🟢 audit 통과 — downstream 학습 진행 가능')
    sys.exit(0)

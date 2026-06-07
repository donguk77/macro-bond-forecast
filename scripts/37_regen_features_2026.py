"""
37_regen_features_2026.py — 2026 라이브 OOS용 피처 파이프라인 재생성

체인 (기존 nb01 §8 + script 14 + nb09 SET_A 와 동일 로직):
  wide_daily_filled.csv (2026 확장)
    → v1: wide_filled[[TARGET]+FROZEN_FEATURES(9)].dropna()   (nb01 §8)
    → v2: shift(1)+파생5+lag/roll                              (scripts/14 그대로 호출)
    → v3: v2에서 kospi 컬럼 전부 제거, 백업 v3 컬럼 순서 적용  (nb09 SET_A)

검증: 재생성 v3를 2025-12-31 이하로 자른 것이 백업 v3와 값까지 동일한지 대조.
누수 가드: crisis_dummy 임계값은 train-only(2010-2020) quantile → 2026 확장이 과거를 바꾸지 않음.

사용: python scripts/37_regen_features_2026.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
PROC = DATA / 'processed'
TARGET = 'kr_treasury_10y'

# nb01 §8 옵션 C+ 9변수 (순서 동일)
FROZEN_FEATURES = [
    'kr_treasury_3y', 'kr_base_rate',
    'us_treasury_10y', 'us_fed_funds', 'us_breakeven_10y',
    'vix', 'kospi', 'sp500', 'dxy',
]

# 백업 위치 (검증 대조용) — 가장 최근 _backup_2025end_*
BK = sorted(DATA.glob('_backup_2025end_*'))[-1]

print('=' * 72)
print('37_regen_features_2026 — v1 → v2 → v3 재생성')
print(f'백업 대조: {BK.name}')
print('=' * 72)

# ─── 1. v1 재생성 (nb01 §8 로직) ─────────────────────────────────
wide_filled = pd.read_csv(DATA / 'interim' / 'wide_daily_filled.csv',
                          index_col='date', parse_dates=['date'])
v1 = wide_filled[[TARGET] + FROZEN_FEATURES].dropna()
v1.to_csv(PROC / 'features_v1_candidate.csv', index_label='date')
print(f'[v1] {v1.shape}  {v1.index.min().date()} ~ {v1.index.max().date()}')

# ─── 2. v2 재생성 (검증된 scripts/14 그대로 실행) ────────────────
print('[v2] scripts/14_features_v2.py 실행 ...')
r = subprocess.run([sys.executable, str(ROOT / 'scripts' / '14_features_v2.py')],
                   capture_output=True, text=True, encoding='utf-8')
if r.returncode != 0:
    print(r.stdout); print(r.stderr); sys.exit('script 14 실패')
# 마지막 몇 줄만 출력
print('   ' + '\n   '.join(r.stdout.strip().splitlines()[-6:]))

# ─── 3. v3 구성: v2 − kospi, 백업 v3 컬럼 순서 적용 ───────────────
v2 = pd.read_csv(PROC / 'features_v2_no_leak.csv', index_col='date', parse_dates=['date'])
bk_v3_cols = list(pd.read_csv(BK / 'features_v3_candidate.csv', nrows=1).columns)
bk_v3_cols = [c for c in bk_v3_cols if c != 'date']  # index 제외

missing = [c for c in bk_v3_cols if c not in v2.columns]
assert not missing, f'v2에 없는 v3 컬럼: {missing}'
v3 = v2[bk_v3_cols].copy()
v3.to_csv(PROC / 'features_v3_candidate.csv', index_label='date')
print(f'[v3] {v3.shape}  {v3.index.min().date()} ~ {v3.index.max().date()}')

# ─── 4. 검증: ≤2025-12-31 구간이 백업과 값까지 동일한가 ───────────
print('-' * 72)
print('[검증] 재생성 v3(≤2025-12-31) vs 백업 v3')
bk_v3 = pd.read_csv(BK / 'features_v3_candidate.csv', index_col='date', parse_dates=['date'])
new_hist = v3.loc[:'2025-12-31']
# 동일 인덱스/컬럼으로 정렬
common_idx = bk_v3.index.intersection(new_hist.index)
print(f'  백업 행수={len(bk_v3)}  재생성≤2025 행수={len(new_hist)}  공통={len(common_idx)}')
a = bk_v3.loc[common_idx, bk_v3_cols]
b = new_hist.loc[common_idx, bk_v3_cols]
diff = (a - b).abs()
max_abs = float(np.nanmax(diff.values))
n_mismatch = int((diff > 1e-9).sum().sum())
print(f'  최대 절대차={max_abs:.3e}  | 불일치 셀(>1e-9)={n_mismatch}')
idx_match = list(bk_v3.index) == list(new_hist.index)
print(f'  인덱스 완전일치={idx_match}')
if max_abs < 1e-6 and idx_match:
    print('  ✅ 검증 통과 — 과거 구간 재현 동일')
else:
    print('  ⚠️ 불일치 존재 — 점검 필요')
    # 컬럼별 최대차 top
    cmax = diff.max().sort_values(ascending=False).head(8)
    print(cmax)

# 2026 신규 구간 요약
new_2026 = v3.loc['2026-01-01':]
print('-' * 72)
print(f'[2026 신규] 행수={len(new_2026)}  {new_2026.index.min().date()} ~ {new_2026.index.max().date()}')
print(f'  delta_y_bp NaN={int(new_2026["delta_y_bp"].isna().sum())}')

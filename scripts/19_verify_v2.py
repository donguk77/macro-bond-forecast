"""
19_verify_v2.py — v2 작업 모든 주장 fresh 재검증

verification-before-completion 가이드라인:
  - 모든 주장에 대해 실제 명령 실행
  - 출력 직접 확인 후 PASS/FAIL 표기
  - 한 건이라도 FAIL 시 exit 1
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_V2 = PROJECT_ROOT / 'reports' / 'no_leak_v2'
MODELS_DIR = PROJECT_ROOT / 'models'

print('=' * 72)
print('19_verify_v2.py — v2 모든 주장 fresh 재검증')
print('=' * 72)

results = []  # (check_name, status, evidence)


def add(name, ok, evidence):
    status = '✅ PASS' if ok else '❌ FAIL'
    results.append((name, status, evidence))
    print(f'\n[{status}] {name}')
    print(f'  → {evidence}')


# ─────────────────────────────────────────────────────────────────────────
# 검증 1: 17_full_audit_v2.py 재실행 → exit 0
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('1. 누수 audit 재실행')
print('=' * 72)
proc = subprocess.run(
    [sys.executable, '-X', 'utf8', str(PROJECT_ROOT / 'scripts' / '17_full_audit_v2.py')],
    capture_output=True, text=True, encoding='utf-8',
    env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUTF8': '1'},
)
exit_code = proc.returncode
last_lines = proc.stdout.strip().split('\n')[-5:]
add(
    '1. leakage audit re-run exit 0',
    exit_code == 0,
    f'exit={exit_code}, last lines: {last_lines[-1] if last_lines else "EMPTY"}',
)

# ─────────────────────────────────────────────────────────────────────────
# 검증 2: features v2 미국 마감변수 spot check (직접 raw vs v2 비교)
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('2. features v2 미국 마감변수 t-1 spot check')
print('=' * 72)

raw = pd.read_csv(DATA_DIR / 'processed' / 'features_v1_candidate.csv',
                  index_col='date', parse_dates=['date']).sort_index()
v2 = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

US_CLOSE = ['us_treasury_10y', 'us_breakeven_10y', 'vix', 'sp500', 'dxy']
spot_check_pass = True
spot_evidence = []
for v_col in US_CLOSE:
    if v_col not in v2.columns:
        spot_check_pass = False
        spot_evidence.append(f'{v_col}: MISSING')
        continue
    common = raw.index.intersection(v2.index)
    a = v2.loc[common, v_col]
    b_t = raw.loc[common, v_col]
    b_t1 = raw.loc[common, v_col].shift(1)
    valid = ~(a.isna() | b_t.isna())
    rate_t = float((np.abs(a[valid] - b_t[valid]) < 1e-6).mean())
    valid1 = ~(a.isna() | b_t1.isna())
    rate_t1 = float((np.abs(a[valid1] - b_t1[valid1]) < 1e-6).mean())
    is_t1 = (rate_t1 >= 0.99 and rate_t < 0.50)
    spot_evidence.append(f'{v_col}: t-1={rate_t1:.3f} t={rate_t:.3f} {"OK" if is_t1 else "FAIL"}')
    if not is_t1:
        spot_check_pass = False

add(
    '2. US close vars all shift(1) applied (raw vs v2 spot check)',
    spot_check_pass,
    '; '.join(spot_evidence),
)

# ─────────────────────────────────────────────────────────────────────────
# 검증 3: XGB v2 single-split dir_acc 61.13% claim
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('3. XGB v2 single-split test dir_acc')
print('=' * 72)

xgb_eval = pd.read_csv(REPORT_V2 / 'xgb_v2_eval.csv')
xgb_test_raw = xgb_eval[(xgb_eval['split'] == 'test') & (xgb_eval['stage'] == 'raw')]
xgb_test_cqr = xgb_eval[(xgb_eval['split'] == 'test') & (xgb_eval['stage'] == 'CQR')]

claimed_dir = 0.6113
actual_dir_raw = float(xgb_test_raw['dir_acc_q50'].iloc[0])
actual_dir_cqr = float(xgb_test_cqr['dir_acc_q50'].iloc[0])
actual_cov_raw = float(xgb_test_raw['coverage_90'].iloc[0])
actual_cov_cqr = float(xgb_test_cqr['coverage_90'].iloc[0])

add(
    '3. XGB v2 single dir_acc 0.6113 claim',
    abs(actual_dir_cqr - claimed_dir) < 1e-3,
    f'claimed=0.6113, actual_raw={actual_dir_raw:.4f}, actual_CQR={actual_dir_cqr:.4f}',
)

claimed_cov_cqr = 0.8573
add(
    '3b. XGB v2 single Coverage CQR 0.8573 claim',
    abs(actual_cov_cqr - claimed_cov_cqr) < 1e-3,
    f'claimed=0.8573, actual_raw={actual_cov_raw:.4f}, actual_CQR={actual_cov_cqr:.4f}',
)

# ─────────────────────────────────────────────────────────────────────────
# 검증 4: 3-fold 평균 dir_acc 60.99%±2.69% claim
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('4. Walk-forward 3-fold 평균 dir_acc 재계산')
print('=' * 72)

wf_xgb = pd.read_csv(REPORT_V2 / 'walkforward_xgb_v2.csv')
wf_xgb_cqr = wf_xgb[wf_xgb['stage'] == 'CQR']

actual_mean = float(wf_xgb_cqr['dir_acc_q50'].mean())
actual_std = float(wf_xgb_cqr['dir_acc_q50'].std())
claimed_mean = 0.6099
claimed_std = 0.0269

add(
    '4. 3-fold 평균 dir_acc 0.6099 claim',
    abs(actual_mean - claimed_mean) < 1e-3,
    f'claimed=0.6099±0.0269, actual={actual_mean:.4f}±{actual_std:.4f}',
)

# fold별 수치도 spot check
print('  fold별 수치 spot check:')
for _, r in wf_xgb_cqr.iterrows():
    print(f'    {r["fold"]}: dir_acc={r["dir_acc_q50"]:.4f}, coverage={r["coverage_90"]:.4f}, Q_hat={r["cqr_Q_hat"]:+.3f}')

# ─────────────────────────────────────────────────────────────────────────
# 검증 5: DM test (Pooled) DM=-8.78, p=0 claim
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('5. DM test Pooled XGB vs Naive')
print('=' * 72)

dm = pd.read_csv(REPORT_V2 / 'walkforward_dm_v2.csv')
pooled = dm[dm['fold'] == 'POOLED'].iloc[0]
claimed_dm = -8.782
claimed_p = 0.0
actual_dm = float(pooled['DM_HLN'])
actual_p = float(pooled['p_value'])

add(
    '5. Pooled DM_HLN -8.78 claim',
    abs(actual_dm - claimed_dm) < 0.01,
    f'claimed=-8.782, actual={actual_dm:.3f}',
)

add(
    '5b. Pooled p-value < 0.0167 (Bonferroni) claim',
    actual_p < 0.0167,
    f'p_value={actual_p:.4f}, Bonferroni α=0.0167',
)

# fold별 결과 spot check
print('  fold별 DM 결과:')
for _, r in dm.iterrows():
    print(f'    {r["fold"]}: DM={r["DM_HLN"]:.3f}, p={r["p_value"]:.4f}, winner={r["winner"]}')

# ─────────────────────────────────────────────────────────────────────────
# 검증 6: XGB 모델 저장 + 예측 재현 (single-split test)
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('6. XGB v2 모델 로드 + 예측 재현 (test set)')
print('=' * 72)

# 데이터 재구성 (15_xgb_grid_cqr 와 동일)
SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}
FEATURE_COLS = [c for c in v2.columns if c != 'delta_y_bp']

X_train_raw = v2.loc[SPLIT['train'][0]:SPLIT['train'][1], FEATURE_COLS]
X_test_raw = v2.loc[SPLIT['test'][0]:SPLIT['test'][1], FEATURE_COLS]
y_test = v2.loc[SPLIT['test'][0]:SPLIT['test'][1], 'delta_y_bp']

scaler = RobustScaler().fit(X_train_raw)
X_test = pd.DataFrame(scaler.transform(X_test_raw),
                      index=X_test_raw.index, columns=FEATURE_COLS)

# Load 모델 + 예측
preds = {}
for q in [0.05, 0.5, 0.95]:
    m_path = MODELS_DIR / f'xgb_v2_q{int(q*100):02d}.json'
    if not m_path.exists():
        add(f'6. xgb_v2_q{int(q*100):02d}.json model load',
            False, f'MISSING: {m_path}')
        continue
    m = xgb.XGBRegressor()
    m.load_model(str(m_path))
    preds[q] = m.predict(X_test.values)

# Sort
arr = np.column_stack([preds[q] for q in [0.05, 0.5, 0.95]])
arr_s = np.sort(arr, axis=1)
preds_sorted = {q: arr_s[:, i] for i, q in enumerate([0.05, 0.5, 0.95])}

# dir_acc 재계산
y_arr = y_test.values
p_arr = preds_sorted[0.5]
mask = (np.sign(p_arr) != 0) & (np.sign(y_arr) != 0)
reproduced_dir = float((np.sign(p_arr[mask]) == np.sign(y_arr[mask])).mean())

add(
    '6. XGB 모델 로드 후 dir_acc 재현 = csv 값과 일치',
    abs(reproduced_dir - actual_dir_raw) < 1e-3,
    f'csv dir_acc raw={actual_dir_raw:.4f}, 재현={reproduced_dir:.4f}',
)

# Coverage 재계산
reproduced_cov = float(np.mean((y_arr >= preds_sorted[0.05]) & (y_arr <= preds_sorted[0.95])))
add(
    '6b. XGB 모델 로드 후 Coverage raw 재현',
    abs(reproduced_cov - actual_cov_raw) < 1e-3,
    f'csv cov raw={actual_cov_raw:.4f}, 재현={reproduced_cov:.4f}',
)

# ─────────────────────────────────────────────────────────────────────────
# 검증 7: 노트북 08 셀 출력 + figures 존재
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('7. 노트북 08 + figures 존재 검증')
print('=' * 72)

nb_path = PROJECT_ROOT / 'notebooks' / '08_v2_no_leak_pipeline.ipynb'
if nb_path.exists():
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    n_cells = len(nb['cells'])
    n_code = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    n_executed = sum(1 for c in nb['cells']
                     if c['cell_type'] == 'code' and c.get('execution_count') is not None)
    add(
        '7. 노트북 08 cells 모두 실행됨',
        n_executed == n_code,
        f'total cells={n_cells}, code cells={n_code}, executed={n_executed}',
    )
else:
    add('7. 노트북 08 존재', False, f'MISSING: {nb_path}')

# Figures 존재
figs = [
    '01_dir_acc_3stages.png',
    '02_cqr_effect.png',
    '03_walkforward_dir_acc.png',
    '04_dm_test.png',
    '05_summary_4panel.png',
]
fig_dir = REPORT_V2 / 'figures'
missing = [f for f in figs if not (fig_dir / f).exists()]
add(
    '7b. 5 figures 모두 생성됨',
    len(missing) == 0,
    f'missing={missing}' if missing else f'all 5 present, sizes: ' + ', '.join(
        f'{f}={(fig_dir/f).stat().st_size}B' for f in figs),
)

# ─────────────────────────────────────────────────────────────────────────
# 검증 8: 새 변수 5개 모두 v2 features 에 존재 + 통계
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('8. 새 변수 5개 + 위기 더미 train 비율')
print('=' * 72)

NEW_VARS = ['spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1',
            'delta_dxy_t1', 'crisis_dummy']
present = [v_ for v_ in NEW_VARS if v_ in v2.columns]
add(
    '8. 새 변수 5개 모두 존재',
    len(present) == 5,
    f'present={len(present)}/5: {present}',
)

# 위기 더미 train-only 비율 ≈ 20%
train_crisis = float(v2.loc['2010-01-01':'2020-12-31', 'crisis_dummy'].mean())
add(
    '8b. 위기 더미 train 비율 ≈ 20%',
    0.18 <= train_crisis <= 0.22,
    f'train crisis rate = {train_crisis:.4f} (목표 0.20)',
)

# ─────────────────────────────────────────────────────────────────────────
# 종합 결과
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('종합 검증 결과')
print('=' * 72)
n_pass = sum(1 for _, s, _ in results if s.startswith('✅'))
n_fail = sum(1 for _, s, _ in results if s.startswith('❌'))
print(f'\n  PASS: {n_pass} / FAIL: {n_fail}')
print(f'  Total checks: {len(results)}')
print()
for name, status, _ in results:
    print(f'  {status}  {name}')

# Save 검증 결과
df_v = pd.DataFrame(results, columns=['check', 'status', 'evidence'])
df_v.to_csv(REPORT_V2 / 'verification_v2.csv', index=False)
print(f'\n[save] reports/no_leak_v2/verification_v2.csv')

if n_fail > 0:
    print(f'\n🔴 {n_fail} 건 FAIL — 위 항목 확인 필요')
    sys.exit(1)
else:
    print('\n🟢 모든 검증 PASS — v2 작업 신뢰 가능')
    sys.exit(0)

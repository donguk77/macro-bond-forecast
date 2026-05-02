# -*- coding: utf-8 -*-
"""
W5 메타-검증 스크립트
========================
W4 검증 패턴 (#30 → #36 → #37) 의 5주차 적용.
W5 multi-seed + grid 5×5 + ablation 결과의 잔존 결함 점검.
"""
import sys, re, json, subprocess, csv
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def header(title):
    print('='*78)
    print(title)
    print('='*78)

# V1
header('V1: A0 multi-seed CV — 모든 metric 정량 점검')
ms = pd.read_csv(ROOT/'reports/lstm_a0_multiseed_w5.csv')
final = pd.read_csv(ROOT/'reports/lstm_a0_final_eval_w5.csv')
metrics = ['pinball_q05','pinball_q50','pinball_q95','coverage_90','sharpness_bp','rmse_q50_bp','dir_acc_q50']
for name, df in [('default config (h=64,lr=1e-3)', ms), ('best HP (h=128,lr=5e-4)', final)]:
    print(f'\n  {name} — test split:')
    test = df[df['split']=='test']
    for m in metrics:
        mean, std = test[m].mean(), test[m].std()
        cv = std/abs(mean)*100 if mean else float('nan')
        flag = '+' if cv < 5 else ('~' if cv < 10 else 'X')
        print(f'    {m:18s} mean={mean:.4f} std={std:.4f} CV={cv:.1f}% [{flag}]')

# V2
print()
header('V2: Grid 5x5 — val 1위 vs test 1위 일치 (val-overfit 점검)')
grid = pd.read_csv(ROOT/'reports/grid_5x5_w5.csv')
val_top = grid.nsmallest(5, 'best_val_pinball')[['lr','hidden','best_val_pinball','test_pinball_q50','test_rmse_q50_bp','test_dir_acc_q50']]
test_top = grid.nsmallest(5, 'test_pinball_q50')[['lr','hidden','best_val_pinball','test_pinball_q50','test_rmse_q50_bp','test_dir_acc_q50']]
print('  val pinball 상위 5:')
print(val_top.to_string(index=False))
print('\n  test pinball 상위 5:')
print(test_top.to_string(index=False))
val_best = val_top.iloc[0]
test_best = test_top.iloc[0]
match = (val_best['lr']==test_best['lr']) and (val_best['hidden']==test_best['hidden'])
print(f'\n  val 1위 (lr={val_best["lr"]}, h={int(val_best["hidden"])}) vs test 1위 (lr={test_best["lr"]}, h={int(test_best["hidden"])})  {"[+] 일치" if match else "[~] 불일치 - val-overfit 가능성"}')

# V3
print()
header("V3: A0 vs A1' paired difference — 통계적 차이 vs std 범위")
final_test = final[final['split']=='test'].sort_values('seed')
a1 = pd.read_csv(ROOT/'reports/ablation_a1_w5.csv')
a1_test = a1[a1['split']=='test'].sort_values('seed')
print(f'  A0 seeds: {final_test["seed"].tolist()}, A1 seeds: {a1_test["seed"].tolist()}')
for m in ['rmse_q50_bp','coverage_90','dir_acc_q50','pinball_q50']:
    a0_vals = final_test[m].values
    a1_vals = a1_test[m].values
    diff = a1_vals - a0_vals
    mean_diff, std_diff = diff.mean(), diff.std()
    a0_std = a0_vals.std()
    in_range = abs(mean_diff) < a0_std
    print(f'    {m:15s}  A0 {a0_vals.mean():.4f} -> A1 {a1_vals.mean():.4f}  delta={mean_diff:+.4f}+-{std_diff:.4f}  (A0 std {a0_std:.4f})  [{"같은범위" if in_range else "범위밖"}]')

# V4
print()
header("V4: A2'/A3' corr 사전 점검 — 임계값 |r|<0.05 의 적절성")
fv = pd.read_csv(ROOT/'data/processed/features_v1_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
raw = pd.read_csv(ROOT/'data/interim/wide_daily_filled.csv', index_col=0, parse_dates=[0]).sort_index()
dy = (fv['kr_treasury_10y'].diff()*100).dropna()

for v_name, src in [('kospi', raw['kospi'].reindex(fv.index)),
                    ('kr_ppi(announced)', raw['kr_ppi'].shift(freq='30D').reindex(fv.index, method='ffill'))]:
    rs = []
    for k in [1,2,3,5,10,20]:
        d = src.diff().shift(k)
        common = dy.index.intersection(d.dropna().index)
        rs.append((k, d.loc[common].corr(dy.loc[common])))
    max_r = max(rs, key=lambda kr: abs(kr[1]))
    print(f'  {v_name:20s} lag1~20 corr: {[f"k={k}:{r:+.4f}" for k,r in rs]}')
    print(f'    max |r| at lag={max_r[0]}: {max_r[1]:+.4f}  -> 건너뜀 정당성 [{"+" if abs(max_r[1])<0.05 else "~"}]')

# V5
print()
header('V5: krw_usd 결측 — 데이터 일관성')
print(f'  raw["krw_usd"] 결측: {raw["krw_usd"].isna().sum()} / {len(raw)}')
krw_aligned = raw['krw_usd'].reindex(fv.index)
print(f'  fv 인덱스 reindex 후 결측: {krw_aligned.isna().sum()}')
SPLIT = {'train':('2010-01-01','2020-12-31'), 'cal':('2021-01-01','2021-12-31'),
         'val':('2022-01-01','2022-12-31'), 'test':('2023-01-01','2025-12-31')}
for sp, (s,e) in SPLIT.items():
    n_null = krw_aligned.loc[s:e].isna().sum()
    n_total = len(krw_aligned.loc[s:e])
    print(f'    {sp:6s} 결측 {n_null}/{n_total} ({n_null/n_total*100:.1f}%)')

# A1' 시퀀스 크기 비교
print(f'\n  A0 vs A1 시퀀스 size 비교 (이후 dropna 영향):')
fv_a1 = fv[['kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
            'us_breakeven_10y','vix','sp500','dxy']].copy()
fv_a1['krw_usd'] = krw_aligned
diff_a0 = fv_a1.iloc[:, :8].diff().shift(1).dropna()
diff_a1 = fv_a1.diff().shift(1).dropna()
print(f'    A0 (8 vars) dropna 후 행: {len(diff_a0)}')
print(f'    A1 (9 vars) dropna 후 행: {len(diff_a1)}  (차이 {len(diff_a0)-len(diff_a1)} 행)')

# V6
print()
header('V6: CQR Q 산출 + 적용 후 Coverage 정합')
cqr = pd.read_csv(ROOT/'reports/cqr_post_w5.csv')
print(cqr.to_string(index=False))
Q = cqr['Q_bp'].iloc[0]
pre_cov = cqr.iloc[0]['coverage_90']
post_cov = cqr.iloc[1]['coverage_90']
print(f'\n  Q = {Q:.3f} bp')
print(f'  pre 0.9 target {"+" if 0.87<=pre_cov<=0.93 else "X"}: pre {pre_cov:.4f}, post {post_cov:.4f}')
# Q 의 의미 — Q bp 만큼 양 끝 확장 -> sharpness 가 2*Q 만큼 증가
expected_sharp_inc = 2*Q
actual_sharp_inc = cqr.iloc[1]['sharpness_bp'] - cqr.iloc[0]['sharpness_bp']
print(f'  sharpness 증가: 예상 2Q = {expected_sharp_inc:.3f} bp  vs  실제 {actual_sharp_inc:+.3f} bp  [{"+" if abs(expected_sharp_inc-actual_sharp_inc)<0.1 else "~"}]')

# V7
print()
header('V7: 노트북 실행 완전성')
nb = json.loads((ROOT/'notebooks/05_tuning_ablation.ipynb').read_text(encoding='utf-8'))
n_code = sum(1 for c in nb['cells'] if c['cell_type']=='code')
n_with_output = sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs'))
n_errors = sum(1 for c in nb['cells'] if c['cell_type']=='code'
               for o in c.get('outputs', [])
               if o.get('output_type')=='error')
print(f'  code cells: {n_code}, output 있는 cell: {n_with_output}, error: {n_errors}')

# V8
print()
header('V8: Audit 재실행 — CL-05b/c 잔존 결함 동일 보고')
res = subprocess.run([sys.executable, str(ROOT/'scripts/04_leakage_audit.py')], capture_output=True, encoding='utf-8')
with open(ROOT/'reports/leakage_audit_w2.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(f'  {r["CL"]:8s} {r["상태"]} {r["항목"][:55]}')
n_pass = sum(1 for r in rows if r['상태']=='✅')
n_fail = sum(1 for r in rows if r['상태']=='❌')
print(f'  -> ✅ {n_pass}건 / ❌ {n_fail}건  [{"+" if n_pass==7 and n_fail==2 else "~"}]')

# V9: A1' 의 hidden=128 모델 이 A0 default config (h=64) 와 직접 비교 가능한가?
print()
header("V9: A1' 비교 baseline 일관성")
# A1' 는 BEST_HP (lr=5e-4, h=128) + 9 vars 로 학습
# A0 final 은 BEST_HP (lr=5e-4, h=128) + 8 vars 로 학습
# 비교는 동일 HP 에서 변수 변경만 격리 -> [+] 정합
# 다만 9 vars 에서 best HP 가 동일한지 미검증 (grid 재실행 안 함)
print(f'  A1\' 와 A0 final 모두 lr=5e-4, hidden=128 사용 -> [+] HP 동일')
print(f'  단, 9 vars 에서의 best HP 가 8 vars 와 동일한지 별도 grid 안 함')
print(f'  -> A1\' 효과가 미미한 이유로 (1) 신호 부재, (2) HP 부적합 둘 다 가능')
print(f'     만약 (2) 라면 9 vars 전용 grid 필요 -> 학부 7주 일정상 deferred OK')

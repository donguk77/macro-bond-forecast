# -*- coding: utf-8 -*-
"""
W1-W6 전체 정합성 검증 (중간평가용 실제 검증)
============================================
1. 노트북 실행 산출물 vs 보고된 수치
2. cross-reference: 변수·split·모델 일관성
3. LOG 카운트 정합
4. 파일 무결성
5. audit 재실행
"""
import sys, json, subprocess, csv, pickle, re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def H(t):
    print('='*78); print(t); print('='*78)

# 1. 파일 존재 검증 (LOG/docs/notebooks/scripts/reports 가 모두 있는지)
H('1. 핵심 산출물 존재 여부')
required = {
    'notebooks/01_eda.ipynb': 'W1',
    'notebooks/02_feature_selection.ipynb': 'W2',
    'notebooks/02b_preprocess_baseline.ipynb': 'W2',
    'notebooks/03_freeze_xgboost.ipynb': 'W3',
    'notebooks/04_lstm_quantile.ipynb': 'W4',
    'notebooks/05_tuning_ablation.ipynb': 'W5',
    'notebooks/06_shap_error_analysis.ipynb': 'W6',
    'scripts/04_leakage_audit.py': 'audit',
    'scripts/05_lstm_diff_ablation.py': 'A0 ablation',
    'scripts/06_w5_meta_verify.py': 'W5 meta',
    'scripts/07_w6_meta_verify.py': 'W6 meta',
    'docs/project_plan.md': 'plan',
    'docs/freeze_final_w3.md': 'freeze',
    'docs/ablation_plan_w3.md': 'ablation',
    'docs/feature_validation_w2.md': 'W2 validation',
    'VALIDATION_LOG.md': 'LOG',
    'reports/baseline_results_w5.csv': 'W5 baseline',
    'reports/dm_test_w6.csv': 'W6 DM',
    'reports/summary_w6.md': 'W6 summary',
    'reports/lstm_a0_shap_w6.npz': 'W6 SHAP',
}
missing = [p for p,d in required.items() if not (ROOT/p).exists()]
print(f'  required {len(required)}, missing {len(missing)}')
for p in missing: print(f'  X {p}')

# Models — gitignored .pt 는 로컬에만
if (ROOT/'models/lstm_a0_final_w5.pt').exists():
    print(f'  + models/lstm_a0_final_w5.pt (local only, gitignored)')
else:
    print(f'  X models/lstm_a0_final_w5.pt missing locally — W6/W7 SHAP/Streamlit 영향')

# 2. Cross-reference: FROZEN 변수가 W3/W4/W5/W6 모두 일치
print()
H('2. Cross-reference: FROZEN 8 vars 일관성')
ck = pickle.load(open(ROOT/'models/scaler_robust_train.pkl','rb'))
print(f'  scaler split keys: {list(ck["split"].keys())}')

# W3 freeze
fz3 = (ROOT/'docs/freeze_final_w3.md').read_text(encoding='utf-8')
w3_vars = re.findall(r'`(kr_treasury_3y|kr_base_rate|us_treasury_10y|us_fed_funds|us_breakeven_10y|vix|sp500|dxy|kospi)`', fz3)
w3_set = set(w3_vars) - {'kospi'}
print(f'  W3 freeze (kospi 제외): {sorted(w3_set)}')

# W6 SHAP features
shap_npz = np.load(ROOT/'reports/lstm_a0_shap_w6.npz', allow_pickle=True)
w6_vars = set(list(shap_npz['features']))
print(f'  W6 SHAP features:       {sorted(w6_vars)}')
match_3_6 = w3_set == w6_vars
print(f'  W3==W6: {"+" if match_3_6 else "X"}')

# 3. 핵심 수치 cross-validation
print()
H('3. 핵심 수치 cross-validation')

# A0 final RMSE: W5 final eval vs W6 predictions
w5_final = pd.read_csv(ROOT/'reports/lstm_a0_final_eval_w5.csv')
import torch
ckpt = torch.load(ROOT/'models/lstm_a0_final_w5.pt', map_location='cpu', weights_only=False)
best_seed = ckpt['config']['seed']
w5_test_best = w5_final[(w5_final['split']=='test') & (w5_final['seed']==best_seed)].iloc[0]
w6_pred = pd.read_csv(ROOT/'data/processed/lstm_a0_predictions_w6.csv')
w6_test = w6_pred[w6_pred['split']=='test']
rmse_w6 = float(np.sqrt(np.mean((w6_test['y_true_bp'] - w6_test['q50'])**2)))
diff_rmse = abs(w5_test_best['rmse_q50_bp'] - rmse_w6)
print(f'  W5 best_seed={best_seed} test RMSE: {w5_test_best["rmse_q50_bp"]:.4f}')
print(f'  W6 predictions test RMSE:  {rmse_w6:.4f}')
print(f'  diff: {diff_rmse:.4f}  [{"+" if diff_rmse<0.001 else "X"}]')

# DM test 결과 확인
dm = pd.read_csv(ROOT/'reports/dm_test_w6.csv')
print(f'\n  DM test 결과 ({len(dm)} comparisons):')
all_sig = True
for _, r in dm.iterrows():
    sig = r.get('bonf','?') if 'bonf' in dm.columns else r.get('sig_α*=0.0167 (Bonf)','?')
    print(f'    {r["comparison"]:18s}  DM_HLN={r["DM_HLN"]:+.3f}  p={r["p_value"]:.4f}  {sig}')
    if 'OK' not in str(sig) and '✅' not in str(sig): all_sig = False
print(f'  -> all 3 Bonferroni 유의: {"+" if all_sig else "X"}')

# baseline_results_w5 의 A0 vs W6 일치
bw5 = pd.read_csv(ROOT/'reports/baseline_results_w5.csv')
a0_w5 = bw5[(bw5['model']=='A0_Δfeat[t-1]') & (bw5['split']=='test')]
if len(a0_w5):
    a0_rmse_w5 = float(a0_w5.iloc[0]['RMSE_bp'])
    final_test_mean_rmse = w5_final[w5_final['split']=='test']['rmse_q50_bp'].mean()
    print(f'\n  A0 RMSE 일관성:')
    print(f'    baseline_results_w5 A0 test: {a0_rmse_w5:.3f}')
    print(f'    final_eval_w5 test mean:    {final_test_mean_rmse:.3f}')
    diff_a0 = abs(a0_rmse_w5 - final_test_mean_rmse)
    print(f'    diff: {diff_a0:.3f}  [{"+" if diff_a0<0.01 else "X"}]')

# 4. LOG 카운트
print()
H('4. VALIDATION_LOG 카운트')
log = (ROOT/'VALIDATION_LOG.md').read_text(encoding='utf-8')
entries = re.findall(r'^### #(\d+) \|', log, re.MULTILINE)
header = re.search(r'현재 (\d+)건 기록', log)
match = int(header.group(1)) == len(entries)
print(f'  헤더 {header.group(1)} = 실제 {len(entries)}  [{"+" if match else "X"}]')

# 5. Audit 재실행
print()
H('5. Audit 재실행')
res = subprocess.run([sys.executable, str(ROOT/'scripts/04_leakage_audit.py')], capture_output=True, encoding='utf-8')
with open(ROOT/'reports/leakage_audit_w2.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
n_pass = sum(1 for r in rows if r['상태']=='✅')
n_fail = sum(1 for r in rows if r['상태']=='❌')
print(f'  ✅ {n_pass} / ❌ {n_fail}')
for r in rows:
    print(f'  {r["CL"]:8s} {r["상태"]} {r["항목"][:50]}')
expected_state = '7 ✅ + 2 ❌ (CL-05b/c 잔존)'
actual_state = f'{n_pass} ✅ + {n_fail} ❌'
print(f'  expected: {expected_state}, actual: {actual_state}  [{"+" if (n_pass==7 and n_fail==2) else "X"}]')

# 6. 노트북 실행 완전성 (모든 노트북 error 0?)
print()
H('6. 노트북 실행 완전성')
for nb_name in ['01_eda','02_feature_selection','02b_preprocess_baseline','03_freeze_xgboost',
                '04_lstm_quantile','05_tuning_ablation','06_shap_error_analysis']:
    p = ROOT/f'notebooks/{nb_name}.ipynb'
    if not p.exists(): continue
    nb = json.loads(p.read_text(encoding='utf-8'))
    n_code = sum(1 for c in nb['cells'] if c['cell_type']=='code')
    n_with_out = sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs'))
    n_err = sum(1 for c in nb['cells'] if c['cell_type']=='code'
                for o in c.get('outputs', []) if o.get('output_type')=='error')
    flag = '+' if n_err==0 and n_with_out==n_code else 'X'
    print(f'  {nb_name:30s}  cells {n_code}, output {n_with_out}, error {n_err}  [{flag}]')

# 7. 파일 정합성 (.gitignore 가 .env, .venv, models, data 차단하는지)
print()
H('7. .gitignore 차단 검증')
gi = (ROOT/'.gitignore').read_text(encoding='utf-8')
checks = ['.env', '.venv/', 'data/raw/*', 'data/interim/*', 'data/processed/*', 'models/*', '*.pt', '*.pkl']
for pat in checks:
    if pat in gi:
        print(f'  + {pat} 차단')
    else:
        print(f'  X {pat} 미차단')

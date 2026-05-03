# -*- coding: utf-8 -*-
"""
W1-W7 전체 코드 점검 (최종 audit)
==================================
사용자 요청: "지금까지의 모든 코드를 점검을 해보는 게 좋을 것 같아"

점검 항목:
A. 파일 구조 + 산출물 (40+ files)
B. 변수/SPLIT 일관성 (cross-reference)
C. 핵심 수치 cross-validation (W2/W3/W4/W5/W6/W7 모두)
D. 누수 audit (CL-01~07 + CL-05b/c)
E. trivial bias 추가 검증 (LOG #44)
F. 모델 reproducibility (seed)
G. 노트북 cells/outputs/errors
H. Streamlit 앱 syntax
I. 문서 정합성 (project_plan §2.1·§2.2, LOG count, README)
J. 메타-검증 흔적 (#30→#44)
"""
import sys, json, subprocess, csv, pickle, re, ast
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
issues = []  # (severity, category, msg)

def H(t):
    print('\n' + '='*78)
    print(t)
    print('='*78)

def check(cond, ok, ng, severity='M', cat=''):
    flag = '+' if cond else 'X'
    print(f'  [{flag}] {ok if cond else ng}')
    if not cond:
        issues.append((severity, cat, ng))

# ====================================================================
H('A. 파일 구조 + 산출물')
# ====================================================================
required = {
    # notebooks
    'notebooks/01_eda.ipynb': 'W1', 'notebooks/02_feature_selection.ipynb': 'W2',
    'notebooks/02b_preprocess_baseline.ipynb': 'W2', 'notebooks/03_freeze_xgboost.ipynb': 'W3',
    'notebooks/04_lstm_quantile.ipynb': 'W4', 'notebooks/05_tuning_ablation.ipynb': 'W5',
    'notebooks/06_shap_error_analysis.ipynb': 'W6', 'notebooks/07_final_demo.ipynb': 'W7',
    # scripts
    'scripts/01_verify_ecos_codes.py': 'collect', 'scripts/02_collect_data.py': 'collect',
    'scripts/03_eda_check.py': 'W1', 'scripts/04_leakage_audit.py': 'audit',
    'scripts/05_lstm_diff_ablation.py': 'A0', 'scripts/06_w5_meta_verify.py': 'W5 meta',
    'scripts/07_w6_meta_verify.py': 'W6 meta', 'scripts/08_full_audit_w1w6.py': 'audit',
    'scripts/09_full_audit_w1w7.py': 'this',
    # streamlit
    'app/streamlit_app.py': 'W7',
    # docs
    'docs/project_plan.md': 'plan', 'docs/freeze_final_w3.md': 'W3',
    'docs/ablation_plan_w3.md': 'W3', 'docs/feature_validation_w2.md': 'W2',
    'docs/team_roles.md': 'team', 'README.md': 'readme', 'VALIDATION_LOG.md': 'LOG',
    # reports
    'reports/baseline_results_w2.csv': 'W2', 'reports/baseline_results_w3.csv': 'W3',
    'reports/baseline_results_w4.csv': 'W4', 'reports/baseline_results_w5.csv': 'W5',
    'reports/lstm_a0_final_eval_w5.csv': 'W5', 'reports/lstm_a0_multiseed_w5.csv': 'W5',
    'reports/grid_5x5_w5.csv': 'W5', 'reports/ablation_a1_w5.csv': 'W5',
    'reports/cqr_post_w5.csv': 'W5',
    'reports/dm_test_w6.csv': 'W6', 'reports/error_analysis_w6.csv': 'W6',
    'reports/crisis_labels_w6.csv': 'W6', 'reports/channel_validation_w6.csv': 'W6',
    'reports/summary_w6.md': 'W6', 'reports/lstm_a0_shap_w6.npz': 'W6',
    'reports/leakage_audit_w2.csv': 'audit',
}
missing = [p for p in required if not (ROOT/p).exists()]
print(f'  required: {len(required)}, missing: {len(missing)}')
for p in missing[:5]: print(f'    X {p}')
check(len(missing)==0, f'all {len(required)} files present', f'{len(missing)} missing', 'H', 'A')

# Local-only (gitignored): models/lstm_a0_final_w5.pt
ck = ROOT/'models/lstm_a0_final_w5.pt'
print(f'  [+] models/lstm_a0_final_w5.pt {"(local OK)" if ck.exists() else "(missing — W6 SHAP/Streamlit 영향)"}')

# ====================================================================
H('B. 변수/SPLIT 일관성 (cross-reference)')
# ====================================================================
fv = pd.read_csv(ROOT/'data/processed/features_v1_candidate.csv', index_col='date', parse_dates=['date'])
SPLIT = pickle.load(open(ROOT/'models/scaler_robust_train.pkl','rb'))['split']
print(f'  SPLIT: train {SPLIT["train"]}, cal {SPLIT["cal"]}, val {SPLIT["val"]}, test {SPLIT["test"]}')
check(list(SPLIT.keys())==['train','cal','val','test'], 'SPLIT 4-way OK', 'SPLIT not 4-way', 'H', 'B')

# FROZEN 8 across W3, W4, W5, W6
fz3 = (ROOT/'docs/freeze_final_w3.md').read_text(encoding='utf-8')
w3_set = set(re.findall(r'`(kr_treasury_3y|kr_base_rate|us_treasury_10y|us_fed_funds|us_breakeven_10y|vix|sp500|dxy)`', fz3))
shap_npz = np.load(ROOT/'reports/lstm_a0_shap_w6.npz', allow_pickle=True)
w6_set = set(str(f) for f in shap_npz['features'])
check(w3_set==w6_set, f'W3 freeze == W6 SHAP features ({len(w3_set)} vars)', 'W3/W6 vars mismatch', 'H', 'B')
check(len(w3_set)==8, '8 vars', f'{len(w3_set)} vars', 'H', 'B')

# ====================================================================
H('C. 핵심 수치 cross-validation')
# ====================================================================
import torch
ckpt = torch.load(ROOT/'models/lstm_a0_final_w5.pt', map_location='cpu', weights_only=False)
best_seed = ckpt['config']['seed']
print(f'  W5 best_seed (saved in ckpt): {best_seed}')

# C1: W5 final eval test RMSE == W6 predictions test RMSE
w5 = pd.read_csv(ROOT/'reports/lstm_a0_final_eval_w5.csv')
w5_best = w5[(w5['split']=='test') & (w5['seed']==best_seed)].iloc[0]
w6 = pd.read_csv(ROOT/'data/processed/lstm_a0_predictions_w6.csv', parse_dates=['date'])
w6_test = w6[w6['split']=='test']
rmse_w6 = float(np.sqrt(np.mean((w6_test['y_true_bp'] - w6_test['q50'])**2)))
check(abs(w5_best['rmse_q50_bp']-rmse_w6) < 0.001,
      f'W5 best_seed test RMSE {w5_best["rmse_q50_bp"]:.4f} == W6 predictions {rmse_w6:.4f}',
      f'mismatch (diff {abs(w5_best["rmse_q50_bp"]-rmse_w6):.4f})', 'H', 'C')

# C2: A0 baseline_results_w5 == final_eval_w5 mean
bw5 = pd.read_csv(ROOT/'reports/baseline_results_w5.csv')
a0_bw5 = bw5[(bw5['model']=='A0_Δfeat[t-1]') & (bw5['split']=='test')]
if len(a0_bw5):
    a0_rmse = float(a0_bw5.iloc[0]['RMSE_bp'])
    final_mean = w5[w5['split']=='test']['rmse_q50_bp'].mean()
    check(abs(a0_rmse-final_mean) < 0.01,
          f'baseline_w5 A0 {a0_rmse:.3f} == final_eval mean {final_mean:.3f}',
          'mismatch', 'H', 'C')

# C3: DM test reproducibility
from scipy import stats as scistats
xgb = pd.read_csv(ROOT/'data/processed/xgb_predictions_w3.csv', parse_dates=['date'])
lstm_w4 = pd.read_csv(ROOT/'data/processed/lstm_predictions_w4.csv', parse_dates=['date'])
m = w6_test[['date','y_true_bp','q50']].rename(columns={'q50':'a0'}).merge(
    xgb[xgb['split']=='test'][['date','q50']].rename(columns={'q50':'xgb'}), on='date'
).merge(lstm_w4[lstm_w4['split']=='test'][['date','q50']].rename(columns={'q50':'lstm_raw'}), on='date')
m['naive'] = 0.0

def dm_re(e1, e2, h=1):
    d = e1**2 - e2**2; T = len(d); dm = d.mean()
    L = max(1, int(np.floor(4*(T/100)**(2/9))))
    var = ((d-dm)**2).mean()
    for k in range(1, L+1):
        var += 2*(1-k/(L+1))*((d[:-k]-dm)*(d[k:]-dm)).mean()
    var = max(var, 1e-12)
    s = dm/np.sqrt(var/T); corr = np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    return corr*s, 2*(1-scistats.t.cdf(abs(corr*s), df=T-1))

dm_saved = pd.read_csv(ROOT/'reports/dm_test_w6.csv')
all_match = True
for c in ['naive','xgb','lstm_raw']:
    e1 = (m['y_true_bp']-m['a0']).values
    e2 = (m['y_true_bp']-m[c]).values
    dh, p = dm_re(e1, e2)
    saved = dm_saved[dm_saved['comparison']==f'A0_vs_{ {"naive":"Naive","xgb":"XGBoost","lstm_raw":"LSTM_raw"}[c] }'].iloc[0]
    if abs(dh - saved['DM_HLN']) > 0.01: all_match = False
    print(f'    A0 vs {c}: DM_HLN re={dh:+.3f} saved={saved["DM_HLN"]:+.3f}, p={p:.4f}')
check(all_match, 'DM test 3/3 재계산 일치', 'DM mismatch', 'H', 'C')

# ====================================================================
H('D. 누수 audit (CL-01~07 + CL-05b/c)')
# ====================================================================
res = subprocess.run([sys.executable, str(ROOT/'scripts/04_leakage_audit.py')],
                     capture_output=True, encoding='utf-8')
with open(ROOT/'reports/leakage_audit_w2.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
n_pass = sum(1 for r in rows if r['상태']=='✅')
n_fail = sum(1 for r in rows if r['상태']=='❌')
for r in rows:
    print(f'    {r["CL"]:8s} {r["상태"]} {r["항목"][:50]}')
check(n_pass==7 and n_fail==2, f'audit 7 ✅ / 2 ❌ (CL-05b/c 잔존)', f'{n_pass} ✅ / {n_fail} ❌', 'H', 'D')

# ====================================================================
H('E. Trivial bias 검증 (LOG #44)')
# ====================================================================
test = w6_test.copy()
n = len(test); n_up = (test['y_true_bp']>0).sum(); n_down = (test['y_true_bp']<0).sum()
mean_dy = test['y_true_bp'].mean()
mask = (test['y_true_bp']!=0) & (test['q50']!=0)
acc = (np.sign(test.loc[mask,'y_true_bp'])==np.sign(test.loc[mask,'q50'])).mean()
trivial_up = (test.loc[test['y_true_bp']!=0, 'y_true_bp']>0).mean()
edge = acc - max(trivial_up, 1-trivial_up)
print(f'  test N={n}, mean Δy {mean_dy:+.3f} bp, up {n_up/n*100:.1f}% / down {n_down/n*100:.1f}%')
print(f'  trivial best {max(trivial_up,1-trivial_up)*100:.1f}%, A0 LSTM {acc*100:.1f}%, edge {edge*100:+.1f}%p')
check(edge > 0.05, f'trivial 대비 +{edge*100:.1f}%p (>5%p, 진짜 신호)', 'edge < 5%p (trivial 위험)', 'H', 'E')
check(abs(mean_dy) < 0.1, f'mean Δy {mean_dy:+.3f} bp (random walk 가까움)', 'mean Δy 편향 의심', 'M', 'E')

# ====================================================================
H('F. 모델 reproducibility')
# ====================================================================
# multi-seed 결과 CV 확인
ms = pd.read_csv(ROOT/'reports/lstm_a0_multiseed_w5.csv')
ms_test = ms[ms['split']=='test']
cv_rmse = ms_test['rmse_q50_bp'].std()/ms_test['rmse_q50_bp'].mean()*100
print(f'  multi-seed RMSE CV: {cv_rmse:.1f}%')
check(cv_rmse < 5, f'multi-seed CV {cv_rmse:.1f}% (<5%)', f'CV {cv_rmse:.1f}% 큼', 'M', 'F')

# A0 ablation script 의 seed 결정성 확인
script_text = (ROOT/'scripts/05_lstm_diff_ablation.py').read_text(encoding='utf-8')
has_seed = 'torch.manual_seed' in script_text and 'np.random.seed' in script_text
check(has_seed, 'seed 결정성 OK', 'seed 결정성 누락', 'M', 'F')

# ====================================================================
H('G. 노트북 cells/outputs/errors')
# ====================================================================
nb_dir = ROOT/'notebooks'
nb_status = []
for nb_path in sorted(nb_dir.glob('*.ipynb')):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    n_code = sum(1 for c in nb['cells'] if c['cell_type']=='code')
    n_out = sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs'))
    n_err = sum(1 for c in nb['cells'] if c['cell_type']=='code'
                for o in c.get('outputs', []) if o.get('output_type')=='error')
    nb_status.append((nb_path.name, n_code, n_out, n_err))
    flag = '+' if n_err==0 and n_out>0 else ('~' if n_err==0 else 'X')
    print(f'    [{flag}] {nb_path.name:40s} cells {n_code}, output {n_out}, error {n_err}')
n_with_output = sum(1 for _,_,o,_ in nb_status if o>0)
n_total = len(nb_status)
print(f'  → {n_with_output}/{n_total} 노트북에 output 있음')
check(all(e==0 for _,_,_,e in nb_status), '모든 노트북 error 0', 'error 있음', 'H', 'G')

# ====================================================================
H('H. Streamlit 앱 syntax + 의존성')
# ====================================================================
sapp = (ROOT/'app/streamlit_app.py').read_text(encoding='utf-8')
try:
    ast.parse(sapp)
    print('  [+] Python AST parse OK')
    syntax_ok = True
except SyntaxError as e:
    print(f'  [X] SyntaxError: {e}')
    syntax_ok = False
check(syntax_ok, 'streamlit_app.py syntax OK', 'syntax error', 'H', 'H')

# 필수 import 들어 있는지
required_imports = ['streamlit', 'plotly', 'pandas', 'numpy']
all_imp = all(imp in sapp for imp in required_imports)
check(all_imp, f'imports OK ({required_imports})', 'imports 누락', 'M', 'H')

# 데이터 로드 경로 존재
required_paths = ['lstm_a0_predictions_w6.csv', 'crisis_labels_w6.csv', 'dm_test_w6.csv',
                  'baseline_results_w5.csv', 'channel_validation_w6.csv',
                  'lstm_a0_final_eval_w5.csv', 'lstm_a0_shap_w6.npz']
all_paths_in_app = all(p in sapp for p in required_paths)
check(all_paths_in_app, 'app 의 모든 입력 파일 reference OK', 'app 입력 파일 누락', 'H', 'H')

# ====================================================================
H('I. 문서 정합성')
# ====================================================================
plan = (ROOT/'docs/project_plan.md').read_text(encoding='utf-8')
check('Δy_t = (y_t − y_{t-1}) × 100' in plan, 'project_plan §2.1 Δy_t 정의', 'Δy_t 정의 누락', 'H', 'I')
check('1-step ahead' in plan, 'project_plan §2.2 1-step ahead 명시', '1-step ahead 누락', 'H', 'I')

log = (ROOT/'VALIDATION_LOG.md').read_text(encoding='utf-8')
entries = re.findall(r'^### #(\d+) \|', log, re.MULTILINE)
header = re.search(r'현재 (\d+)건 기록', log)
check(int(header.group(1))==len(entries), f'LOG 헤더 {header.group(1)} = 실제 {len(entries)}',
      f'mismatch', 'M', 'I')

readme = (ROOT/'README.md').read_text(encoding='utf-8')
check('A0 LSTM' in readme and 'DM test' in readme, 'README 핵심 키워드 OK', 'README 부족', 'L', 'I')
check('VALIDATION_LOG 43건' in readme or 'VALIDATION_LOG' in readme, 'README LOG 언급', 'LOG 언급 누락', 'L', 'I')

# ====================================================================
H('J. 메타-검증 흔적 (#30 → #44)')
# ====================================================================
expected_meta = [30, 36, 37, 40, 43, 44]
found = [int(e) for e in entries if int(e) in expected_meta]
print(f'  expected meta-verification entries: {expected_meta}')
print(f'  found in LOG:                       {found}')
check(set(found) >= set(expected_meta), f'all 6 meta-verification entries present', 'meta-verification 누락', 'H', 'J')

# 메타-검증 스크립트 존재
meta_scripts = ['scripts/06_w5_meta_verify.py', 'scripts/07_w6_meta_verify.py',
                'scripts/08_full_audit_w1w6.py', 'scripts/09_full_audit_w1w7.py']
all_meta_scripts = all((ROOT/p).exists() for p in meta_scripts)
check(all_meta_scripts, '메타-검증 스크립트 4개 모두 존재', '스크립트 누락', 'M', 'J')

# ====================================================================
H('종합 결과')
# ====================================================================
high = sum(1 for s,_,_ in issues if s=='H')
med = sum(1 for s,_,_ in issues if s=='M')
low = sum(1 for s,_,_ in issues if s=='L')
print(f'  HIGH: {high}, MEDIUM: {med}, LOW: {low}')
if issues:
    print(f'\n  발견된 결함:')
    for s, c, m in issues:
        print(f'    [{s}] [{c}] {m}')
else:
    print(f'\n  ✅ 모든 점검 통과 — 발표/제출 준비 완료')

print(f'\n  최종 audit pass: {len(issues)==0 or high==0}')

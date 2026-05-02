# -*- coding: utf-8 -*-
"""
W6 메타-검증
============================
W4/W5 메타-검증 패턴 (#30 → #36 → #37 → #40) 의 6주차 적용.
"""
import sys, json, subprocess, csv, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats as scistats

ROOT = Path(__file__).resolve().parent.parent

def header(t):
    print('='*78); print(t); print('='*78)

# V1: A0 모델 예측 일관성 (W5 final eval 과 W6 lstm_a0_predictions_w6.csv 정합)
header('V1: A0 모델 예측 일관성 — W5 final eval vs W6 predictions')
w5 = pd.read_csv(ROOT/'reports/lstm_a0_final_eval_w5.csv')
w6_pred = pd.read_csv(ROOT/'data/processed/lstm_a0_predictions_w6.csv', parse_dates=['date'])
# W5 best_seed 의 test RMSE
w5_test = w5[w5['split']=='test']
# Best seed 는 val pinball 최저 → ckpt 의 seed
ckpt = pickle.load(open(ROOT/'models/scaler_robust_train.pkl','rb'))  # not relevant
import torch
ck = torch.load(ROOT/'models/lstm_a0_final_w5.pt', map_location='cpu', weights_only=False)
best_seed = ck['config']['seed']
print(f'  W5 best_seed (ckpt 저장): {best_seed}')
w5_best = w5_test[w5_test['seed']==best_seed].iloc[0]
print(f'  W5 final eval  test RMSE_q50 (seed={best_seed}): {w5_best["rmse_q50_bp"]:.4f}')
w6_test = w6_pred[w6_pred['split']=='test']
rmse_w6 = float(np.sqrt(np.mean((w6_test['y_true_bp'] - w6_test['q50'])**2)))
print(f'  W6 predictions test RMSE_q50 (재계산):           {rmse_w6:.4f}')
match = abs(w5_best['rmse_q50_bp'] - rmse_w6) < 0.001
print(f'  -> [{ "+" if match else "X" }] 일치 ({"<0.001 bp" if match else f"diff={abs(w5_best.rmse_q50_bp-rmse_w6):.4f}"})')

# V2: DM test 산술 검증 — HAC Newey-West + HLN 식
print(); header('V2: DM test 산술 — HAC Newey-West + HLN 보정 공식')
dm = pd.read_csv(ROOT/'reports/dm_test_w6.csv')
print(dm.to_string(index=False))
# 직접 재계산
xgb = pd.read_csv(ROOT/'data/processed/xgb_predictions_w3.csv', parse_dates=['date'])
lstm_w4 = pd.read_csv(ROOT/'data/processed/lstm_predictions_w4.csv', parse_dates=['date'])
a0_test = w6_test[['date','y_true_bp','q50']].rename(columns={'q50':'a0'})
xgb_test = xgb[xgb['split']=='test'][['date','q50']].rename(columns={'q50':'xgb'})
lstm_w4_test = lstm_w4[lstm_w4['split']=='test'][['date','q50']].rename(columns={'q50':'lstm_raw'})
m = a0_test.merge(xgb_test, on='date').merge(lstm_w4_test, on='date')
m['naive'] = 0.0
print(f'\n  DM 비교 sample N (inner join): {len(m)}')

def dm_manual(e1, e2, h=1):
    d = e1**2 - e2**2
    T = len(d); d_mean = d.mean()
    L = max(1, int(np.floor(4*(T/100)**(2/9))))
    var = ((d-d_mean)**2).mean()
    for k in range(1, L+1):
        var += 2*(1-k/(L+1))*((d[:-k]-d_mean)*(d[k:]-d_mean)).mean()
    var = max(var, 1e-12)
    dm_stat = d_mean / np.sqrt(var/T)
    corr = np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    dm_hln = corr*dm_stat
    p = 2*(1-scistats.t.cdf(abs(dm_hln), df=T-1))
    return dm_stat, dm_hln, p, L, T

print(f'\n  재계산 (T, NW lag, DM_HLN, p):')
for c in ['naive','xgb','lstm_raw']:
    e1 = (m['y_true_bp'] - m['a0']).values
    e2 = (m['y_true_bp'] - m[c]).values
    dm_raw, dm_hln, p, L, T = dm_manual(e1, e2)
    saved = dm[dm['comparison']==f'A0_vs_{ {"naive":"Naive","xgb":"XGBoost","lstm_raw":"LSTM_raw"}[c] }'].iloc[0]
    match_dm = abs(dm_hln - saved['DM_HLN']) < 0.01
    match_p = abs(p - saved['p_value (2-sided)']) < 0.01
    print(f'    A0 vs {c:10s}  DM_HLN={dm_hln:+.3f} (저장 {saved["DM_HLN"]:+.3f}) {"[+]" if match_dm else "[X]"}  p={p:.4f} (저장 {saved["p_value (2-sided)"]:.4f}) {"[+]" if match_p else "[X]"}')

# V3: 위기구간 정의의 lookahead 위험
print(); header('V3: 위기구간 정의 — lookahead bias 점검')
fv = pd.read_csv(ROOT/'data/processed/features_v1_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
y = fv['kr_treasury_10y']
dy = (y.diff()*100)
roll_vol_20d = dy.rolling(20).std()
# Threshold 산출 시점
th_full = roll_vol_20d.quantile(0.8)
th_train = roll_vol_20d.loc['2010-01-01':'2020-12-31'].quantile(0.8)
print(f'  Threshold (rolling vol 80% quantile):')
print(f'    전 기간 (2010-2025): {th_full:.4f} bp')
print(f'    train only (2010-2020): {th_train:.4f} bp')
diff_pct = (th_full - th_train) / th_train * 100
print(f'    difference: {diff_pct:+.1f}%')
flag = '[~]' if abs(diff_pct) > 5 else '[+]'
print(f'    {flag} W6 노트북은 전 기간 quantile 사용 → {"lookahead 미세 영향" if abs(diff_pct)>5 else "차이 미미, 무시 가능"}')

# 진짜 leak 인지: test 의 위기 라벨이 test 자체 분포에 의존하는가?
crisis_full = pd.read_csv(ROOT/'reports/crisis_labels_w6.csv', parse_dates=['date'])
test_dates = crisis_full['date']
ratio_full = crisis_full['is_crisis'].mean()
# train threshold 로 다시 라벨링
crisis_alt = (roll_vol_20d.reindex(test_dates) > th_train)
ratio_alt = crisis_alt.mean()
print(f'  위기 비율 (test) — full quantile 기반: {ratio_full*100:.1f}%, train quantile 기반: {ratio_alt*100:.1f}%')

# V4: SHAP 산술 — eval_idx 결정성 + mean|SHAP| 재계산
print(); header('V4: SHAP 산술 — eval_idx 결정성 + 산출물 일관성')
shap_npz = np.load(ROOT/'reports/lstm_a0_shap_w6.npz', allow_pickle=True)
print(f'  npz keys: {list(shap_npz.keys())}')
shap_q05 = shap_npz['shap_q05']; shap_q50 = shap_npz['shap_q50']; shap_q95 = shap_npz['shap_q95']
features = list(shap_npz['features'])
print(f'  shape: q05 {shap_q05.shape}, q50 {shap_q50.shape}, q95 {shap_q95.shape}')
print(f'  mean|SHAP| q50 — top 3:')
abs_q50 = np.abs(shap_q50).mean(axis=(0,1))
ranked = sorted(zip(features, abs_q50), key=lambda x: -x[1])
for f, v in ranked[:3]:
    print(f'    {f:20s} {v:.5f}')
# Reproducibility check — re-run SHAP with same seed should give same result
print(f'  -> SHAP 결과는 npz 로 보존 (재실행 결정성 확인은 비용 큼)')

# V5: 오류 분석 4축 — 산술 재검증
print(); header('V5: 오류 분석 4축 산술 재검증')
df_test = w6_test.copy()
df_test['err_q50'] = df_test['y_true_bp'] - df_test['q50']
df_test['in_band'] = (df_test['y_true_bp']>=df_test['q05']) & (df_test['y_true_bp']<=df_test['q95'])
df_test['sign_correct'] = np.sign(df_test['q50']) == np.sign(df_test['y_true_bp'])
crisis_test = crisis_full.set_index('date')['is_crisis'].reindex(df_test['date']).values
# (a)
da_full = df_test['sign_correct'].mean()
# Dir_Acc 는 sign(p)!=0 & sign(y)!=0 인 경우만
mask_nonzero = (df_test['q50']!=0) & (df_test['y_true_bp']!=0)
da_nonzero = df_test.loc[mask_nonzero, 'sign_correct'].mean()
print(f'  (a) 방향성 정확도: 전체 {da_full*100:.1f}%, sign≠0 mask {da_nonzero*100:.1f}%')
# (b)
big_miss = ((df_test['y_true_bp'].abs()>5) & (df_test['q50'].abs()<1)).sum()
print(f'  (b) 큰 변동 미예측 |Δy|>5 & |q50|<1: {big_miss}건 ({big_miss/len(df_test)*100:.1f}%)')
# (c)
cov = df_test['in_band'].mean()
miss_crisis = (~df_test['in_band'] & crisis_test).sum() / max(crisis_test.sum(),1)
miss_normal = (~df_test['in_band'] & ~crisis_test).sum() / max((~crisis_test).sum(),1)
print(f'  (c) Coverage {cov*100:.1f}%, 위기 Miss {miss_crisis*100:.1f}%, 정상 Miss {miss_normal*100:.1f}%, 비 {miss_crisis/miss_normal:.2f}x')
# (d)
ea = pd.read_csv(ROOT/'reports/error_analysis_w6.csv')
us10 = ea[ea['feature']=='us_treasury_10y'].iloc[0]
print(f'  (d) us_treasury_10y SHAP 위기 {us10["shap_crisis"]:.5f} - 정상 {us10["shap_normal"]:.5f} = {us10["diff (위기-정상)"]:+.5f}')

# V6: 채널 부합 검증의 noise 영역 식별
print(); header('V6: 채널 부합 — signed mean / mean|SHAP| 비율로 noise 영역 식별')
ch = pd.read_csv(ROOT/'reports/channel_validation_w6.csv')
abs_q50_dict = dict(zip(features, abs_q50))
print(f'  {"feature":20s} {"signed":>10s} {"|SHAP|":>10s} {"signed/|SHAP|":>15s}  영역')
for _, r in ch.iterrows():
    f = r['feature']
    s = r['shap_signed_mean']
    a = abs_q50_dict[f]
    ratio = abs(s)/a if a else 0
    region = 'noise (<20%)' if ratio<0.2 else ('weak (20-50%)' if ratio<0.5 else 'strong')
    print(f'  {f:20s} {s:>+10.5f} {a:>10.5f} {ratio*100:>14.1f}%  {region}')

# V7: Audit 재실행
print(); header('V7: Audit 재실행 — W6 산출물 추가 후 회귀 점검')
res = subprocess.run([sys.executable, str(ROOT/'scripts/04_leakage_audit.py')], capture_output=True, encoding='utf-8')
with open(ROOT/'reports/leakage_audit_w2.csv', encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
n_pass = sum(1 for r in rows if r['상태']=='✅')
n_fail = sum(1 for r in rows if r['상태']=='❌')
print(f'  audit: ✅ {n_pass} / ❌ {n_fail}')
for r in rows:
    print(f'  {r["CL"]:8s} {r["상태"]} {r["항목"][:55]}')

# V8: 노트북 실행 완전성
print(); header('V8: 노트북 실행 완전성')
nb = json.loads((ROOT/'notebooks/06_shap_error_analysis.ipynb').read_text(encoding='utf-8'))
n_code = sum(1 for c in nb['cells'] if c['cell_type']=='code')
n_with_out = sum(1 for c in nb['cells'] if c['cell_type']=='code' and c.get('outputs'))
n_err = sum(1 for c in nb['cells'] if c['cell_type']=='code'
            for o in c.get('outputs', []) if o.get('output_type')=='error')
print(f'  code cells: {n_code}, output 있는 cell: {n_with_out}, error: {n_err}')

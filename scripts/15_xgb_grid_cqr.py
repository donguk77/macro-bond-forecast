"""
15_xgb_grid_cqr.py — XGBoost grid search (분위수별) + CQR 후처리

Day 2 작업:
  1. features_v2_no_leak.csv 로드 + train-only scaler
  2. XGBoost 분위수 grid search:
     - max_depth ∈ [4, 5, 6, 8]
     - learning_rate ∈ [0.01, 0.03, 0.05, 0.1]
     - n_estimators: early stopping (val pinball)
     - q05 / q50 / q95 별도 best 선택 (val pinball 기준)
  3. CQR (Conformalized Quantile Regression) 후처리:
     - Cal 2021 conformity score: s_i = max(q05(x_i) - y_i, y_i - q95(x_i))
     - Q = quantile(s, ceil((n+1) * 0.9) / n)
     - Test PI = [q05 - Q, q95 + Q]
  4. 평가: Pinball, Coverage, Sharpness, dir_acc, RMSE
  5. DM test vs Naive (q50 squared error, Bonferroni)

출력:
  reports/no_leak_v2/xgb_grid_v2.csv
  reports/no_leak_v2/xgb_v2_eval.csv (raw vs CQR)
  reports/no_leak_v2/dm_test_xgb_v2.csv
  models/xgb_v2_q05.json / q50.json / q95.json
"""
from __future__ import annotations

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import t as t_dist
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10  # 90% prediction interval

SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}

GRID = {
    'max_depth':     [4, 5, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'n_estimators':  [800],   # early stopping 으로 실제 자동 결정
}

print('=' * 72)
print('15_xgb_grid_cqr.py — XGBoost grid + CQR')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 + 분할
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
print(f'\n[load] shape = {df.shape}')

FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']

def slice_p(p):
    s, e = SPLIT[p]
    return df.loc[s:e]


X_train_raw = slice_p('train')[FEATURE_COLS]
X_cal_raw = slice_p('cal')[FEATURE_COLS]
X_val_raw = slice_p('val')[FEATURE_COLS]
X_test_raw = slice_p('test')[FEATURE_COLS]

scaler = RobustScaler().fit(X_train_raw)


def to_scaled(X_raw):
    return pd.DataFrame(scaler.transform(X_raw), index=X_raw.index, columns=FEATURE_COLS)


X_train = to_scaled(X_train_raw)
X_cal = to_scaled(X_cal_raw)
X_val = to_scaled(X_val_raw)
X_test = to_scaled(X_test_raw)

y_train = slice_p('train')['delta_y_bp']
y_cal = slice_p('cal')['delta_y_bp']
y_val = slice_p('val')['delta_y_bp']
y_test = slice_p('test')['delta_y_bp']

for nm, Xs, ys in [('train', X_train, y_train), ('cal', X_cal, y_cal),
                    ('val', X_val, y_val), ('test', X_test, y_test)]:
    print(f'  {nm:6s}  X.shape={Xs.shape}  y.shape={ys.shape}')


# ─────────────────────────────────────────────────────────────────────────
# 2. 평가 함수
# ─────────────────────────────────────────────────────────────────────────
def pinball(y, p, q):
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def dir_acc(y, p):
    mask = (np.sign(p) != 0) & (np.sign(y) != 0)
    if mask.sum() == 0:
        return float('nan')
    return float((np.sign(p[mask]) == np.sign(y[mask])).mean())


def eval_q(p, y, label):
    out = {'split': label}
    for q in QUANTILES:
        out[f'pinball_q{int(q*100):02d}'] = pinball(y, p[q], q)
    out['coverage_90'] = float(np.mean((y >= p[0.05]) & (y <= p[0.95])))
    out['sharpness_bp'] = float(np.mean(p[0.95] - p[0.05]))
    err = y - p[0.5]
    out['rmse_q50_bp'] = float(np.sqrt(np.mean(err ** 2)))
    out['mae_q50_bp'] = float(np.mean(np.abs(err)))
    out['dir_acc_q50'] = dir_acc(y, p[0.5])
    return out


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


def dm_test_hln(e1, e2, lag=6):
    d = e1 - e2
    n = len(d)
    d_mean = float(d.mean())
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, lag + 1):
        wk = 1.0 - k / (lag + 1)
        cov = float(np.mean((d[:-k] - d_mean) * (d[k:] - d_mean)))
        var += 2.0 * wk * cov
    var = max(var, 1e-12)
    dm = d_mean / np.sqrt(var / n)
    h = 1
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln_factor
    p_val = float(2.0 * (1.0 - t_dist.cdf(abs(dm_hln), df=n - 1)))
    return float(dm_hln), p_val


# ─────────────────────────────────────────────────────────────────────────
# 3. XGBoost grid search (분위수별)
# ─────────────────────────────────────────────────────────────────────────
print('\n=== XGBoost grid search (분위수별, val pinball 최소화) ===')

grid_results = []
best_per_q = {}  # q → (best_params, best_model, best_val_pinball)

for q in QUANTILES:
    best_score = float('inf')
    best_params = None
    best_model = None
    for max_d, lr in product(GRID['max_depth'], GRID['learning_rate']):
        m = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=GRID['n_estimators'][0],
            max_depth=max_d,
            learning_rate=lr,
            early_stopping_rounds=50,
            verbosity=0,
            tree_method='hist',
            random_state=42,
        )
        m.fit(
            X_train.values, y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )
        val_pred = m.predict(X_val.values)
        val_pin = pinball(y_val.values, val_pred, q)
        n_iter = int(m.best_iteration) + 1 if m.best_iteration is not None else GRID['n_estimators'][0]
        grid_results.append({
            'q': q, 'max_depth': max_d, 'learning_rate': lr,
            'n_iter': n_iter, 'val_pinball': val_pin,
        })
        if val_pin < best_score:
            best_score = val_pin
            best_params = {'max_depth': max_d, 'learning_rate': lr, 'n_iter': n_iter}
            best_model = m
    best_per_q[q] = (best_params, best_model, best_score)
    print(f'  q={q}  best params={best_params}  val_pinball={best_score:.4f}')

grid_df = pd.DataFrame(grid_results)
grid_df.to_csv(REPORT_DIR / 'xgb_grid_v2.csv', index=False)
print(f'[save] reports/no_leak_v2/xgb_grid_v2.csv')

# ─────────────────────────────────────────────────────────────────────────
# 4. Best model 로 train/cal/val/test 예측
# ─────────────────────────────────────────────────────────────────────────
preds = {sp: {} for sp in ['train', 'cal', 'val', 'test']}
for q in QUANTILES:
    _, m, _ = best_per_q[q]
    preds['train'][q] = m.predict(X_train.values)
    preds['cal'][q]   = m.predict(X_cal.values)
    preds['val'][q]   = m.predict(X_val.values)
    preds['test'][q]  = m.predict(X_test.values)
    # Save model
    m.save_model(MODELS_DIR / f'xgb_v2_q{int(q*100):02d}.json')

# Sort post-process
preds_sorted = {sp: sort_qs(p) for sp, p in preds.items()}

# Raw 평가
raw_eval = []
for sp, ys in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    raw_eval.append({**eval_q(preds_sorted[sp], ys.values, sp), 'stage': 'raw'})
raw_eval_df = pd.DataFrame(raw_eval)
print('\n=== XGBoost raw 평가 ===')
print(raw_eval_df.round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
# 5. CQR 후처리 (Conformalized Quantile Regression)
# ─────────────────────────────────────────────────────────────────────────
print('\n=== CQR 후처리 (Cal 2021 conformity score) ===')

# Cal conformity score
cal_q05 = preds_sorted['cal'][0.05]
cal_q95 = preds_sorted['cal'][0.95]
cal_y = y_cal.values
cal_score = np.maximum(cal_q05 - cal_y, cal_y - cal_q95)
n_cal = len(cal_score)
# (1 - α) 분위수, finite-sample 보정
k = int(np.ceil((n_cal + 1) * (1 - ALPHA)))
k = min(k, n_cal)
Q_hat = float(np.sort(cal_score)[k - 1])
print(f'  n_cal = {n_cal}, ceil((n+1)*(1-α))/n quantile k={k}')
print(f'  Q_hat = {Q_hat:.4f} bp')

# Test 보정
preds_cqr_test = {
    0.05: preds_sorted['test'][0.05] - Q_hat,
    0.5:  preds_sorted['test'][0.5],
    0.95: preds_sorted['test'][0.95] + Q_hat,
}

# Cal·Val 도 같은 보정 (sanity check 용)
preds_cqr_cal = {
    0.05: preds_sorted['cal'][0.05] - Q_hat,
    0.5:  preds_sorted['cal'][0.5],
    0.95: preds_sorted['cal'][0.95] + Q_hat,
}
preds_cqr_val = {
    0.05: preds_sorted['val'][0.05] - Q_hat,
    0.5:  preds_sorted['val'][0.5],
    0.95: preds_sorted['val'][0.95] + Q_hat,
}

cqr_eval = []
cqr_eval.append({**eval_q(preds_cqr_cal, y_cal.values, 'cal'),
                 'stage': 'CQR'})
cqr_eval.append({**eval_q(preds_cqr_val, y_val.values, 'val'),
                 'stage': 'CQR'})
cqr_eval.append({**eval_q(preds_cqr_test, y_test.values, 'test'),
                 'stage': 'CQR'})
cqr_eval_df = pd.DataFrame(cqr_eval)
print('\n=== XGBoost CQR 평가 ===')
print(cqr_eval_df.round(4).to_string(index=False))

# 결합 + 저장
all_eval = pd.concat([raw_eval_df, cqr_eval_df], ignore_index=True)
all_eval.insert(0, 'model', 'XGBoost(q,v2)')
all_eval.to_csv(REPORT_DIR / 'xgb_v2_eval.csv', index=False)
print(f'\n[save] reports/no_leak_v2/xgb_v2_eval.csv')

# ─────────────────────────────────────────────────────────────────────────
# 6. DM test (Test set, q50 squared error)
# ─────────────────────────────────────────────────────────────────────────
print('\n=== DM test (XGBoost vs Naive · ARIMA, Test set, q50 SE) ===')

# Naive
err_naive = (y_test.values - 0.0) ** 2

# ARIMA (간단: Train-only fit, apply 후 test 예측)
from statsmodels.tsa.arima.model import ARIMA as ARIMA_M
y_full = pd.concat([y_train, y_cal, y_val, y_test]).sort_index()
fit_t = ARIMA_M(y_train, order=(1, 0, 1), trend='c').fit()
fit_f = fit_t.apply(y_full, refit=False)
arima_test = fit_f.predict().reindex(y_test.index)
err_arima = (y_test.values - arima_test.values) ** 2

err_xgb = (y_test.values - preds_sorted['test'][0.5]) ** 2

dm_rows = []
n_compare = 2  # vs Naive · vs ARIMA (Bonferroni alpha = 0.05/2 = 0.025)
for opp_name, err_opp in [('Naive', err_naive), ('ARIMA', err_arima)]:
    dm, p_val = dm_test_hln(err_xgb, err_opp, lag=6)
    bonf_pass = p_val < 0.025
    winner = ('XGB' if dm < 0 and bonf_pass else
              'OPP' if dm > 0 and bonf_pass else 'tie')
    dm_rows.append({
        'comparison': f'XGBv2_vs_{opp_name}',
        'mean_se_xgb': float(err_xgb.mean()),
        'mean_se_opp': float(err_opp.mean()),
        'DM_HLN': dm, 'p_value': p_val,
        'bonf_alpha_0.025': 'OK' if bonf_pass else 'NO',
        'winner': winner,
    })
dm_df = pd.DataFrame(dm_rows)
print(dm_df.round(4).to_string(index=False))
dm_df.to_csv(REPORT_DIR / 'dm_test_xgb_v2.csv', index=False)
print(f'\n[save] reports/no_leak_v2/dm_test_xgb_v2.csv')

# ─────────────────────────────────────────────────────────────────────────
# 7. 요약
# ─────────────────────────────────────────────────────────────────────────
test_raw = raw_eval_df[raw_eval_df['split'] == 'test'].iloc[0]
test_cqr = cqr_eval_df[cqr_eval_df['split'] == 'test'].iloc[0]

print('\n' + '=' * 72)
print('XGBoost v2 결과 요약 (Test set)')
print('=' * 72)
print(f"{'지표':25s}{'raw':>12s}{'CQR':>12s}{'목표':>15s}")
print(f"{'dir_acc':25s}{test_raw['dir_acc_q50']:>12.4f}{test_cqr['dir_acc_q50']:>12.4f}{'≥0.55':>15s}")
print(f"{'coverage_90':25s}{test_raw['coverage_90']:>12.4f}{test_cqr['coverage_90']:>12.4f}{'0.87~0.93':>15s}")
print(f"{'sharpness_bp':25s}{test_raw['sharpness_bp']:>12.4f}{test_cqr['sharpness_bp']:>12.4f}{'좁을수록':>15s}")
print(f"{'rmse_q50_bp':25s}{test_raw['rmse_q50_bp']:>12.4f}{test_cqr['rmse_q50_bp']:>12.4f}{'<Naive 4.65':>15s}")

print('\n=== Day 2 완료 ===')

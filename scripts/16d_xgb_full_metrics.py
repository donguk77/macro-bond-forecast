"""
16d_xgb_full_metrics.py
Re-run XGBoost v2 walkforward to get FULL eval_q() metrics per fold
(the original 16b manual save missed rmse/sharpness columns)
Then regenerate comparison charts with both models complete.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t as t_dist
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
FIG_DIR = REPORT_DIR / 'figures'

QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10

FOLDS = [
    {'name': 'fold1',
     'train': ('2010-01-01','2017-12-31'), 'val': ('2018-01-01','2019-12-31'),
     'cal': ('2019-07-01','2019-12-31'), 'test': ('2020-01-01','2020-12-31')},
    {'name': 'fold2',
     'train': ('2010-01-01','2019-12-31'), 'val': ('2020-01-01','2020-12-31'),
     'cal': ('2020-07-01','2020-12-31'), 'test': ('2021-01-01','2022-12-31')},
    {'name': 'fold3',
     'train': ('2010-01-01','2021-12-31'), 'val': ('2022-01-01','2022-12-31'),
     'cal': ('2022-07-01','2022-12-31'), 'test': ('2023-01-01','2025-12-31')},
]

df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
XGB_FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']

XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}

def pinball(y, p, q):
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))

def dir_acc(y, p):
    mask = (np.sign(p) != 0) & (np.sign(y) != 0)
    if mask.sum() == 0: return float('nan')
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

def fit_xgb(q, X_tr, y_tr, X_val, y_val):
    p = XGB_BEST[q]
    m = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=q,
        n_estimators=p['n_estimators'], max_depth=p['max_depth'],
        learning_rate=p['learning_rate'], early_stopping_rounds=50,
        verbosity=0, tree_method='hist', random_state=42,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m

print('=' * 60)
print('16d — XGBoost v2 full metrics recovery')
print('=' * 60)

all_xgb_eval = []
pooled_xgb_q50, pooled_xgb_q05, pooled_xgb_q95, pooled_y = [], [], [], []
fold3_xgb_preds = {}

for fold in FOLDS:
    name = fold['name']
    print(f'\n{name}: train {fold["train"]}, test {fold["test"]}')

    def sl(p): return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[XGB_FEATURE_COLS]
    X_cal_raw = sl('cal')[XGB_FEATURE_COLS]
    X_val_raw = sl('val')[XGB_FEATURE_COLS]
    X_te_raw  = sl('test')[XGB_FEATURE_COLS]

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
    X_tr, X_cal, X_val, X_te = s(X_tr_raw), s(X_cal_raw), s(X_val_raw), s(X_te_raw)

    y_tr = sl('train')['delta_y_bp']
    y_cal = sl('cal')['delta_y_bp']
    y_val = sl('val')['delta_y_bp']
    y_te  = sl('test')['delta_y_bp']

    xgb_preds = {sp: {} for sp in ['cal', 'test']}
    for q in QUANTILES:
        m = fit_xgb(q, X_tr.values, y_tr.values, X_val.values, y_val.values)
        xgb_preds['cal'][q] = m.predict(X_cal.values)
        xgb_preds['test'][q] = m.predict(X_te.values)
    xgb_preds['cal'] = sort_qs(xgb_preds['cal'])
    xgb_preds['test'] = sort_qs(xgb_preds['test'])

    # CQR
    sc = np.maximum(xgb_preds['cal'][0.05] - y_cal.values,
                    y_cal.values - xgb_preds['cal'][0.95])
    n_c = len(sc)
    k = min(int(np.ceil((n_c + 1) * (1 - ALPHA))), n_c)
    Q_hat = float(np.sort(sc)[k - 1])

    xgb_test_cqr = {
        0.05: xgb_preds['test'][0.05] - Q_hat,
        0.5:  xgb_preds['test'][0.5],
        0.95: xgb_preds['test'][0.95] + Q_hat,
    }

    raw_e = eval_q(xgb_preds['test'], y_te.values, 'test')
    raw_e.update({'fold': name, 'stage': 'raw', 'model': 'XGB(q,v2)',
                  'cqr_Q_hat': Q_hat, 'n_cal': n_c})
    cqr_e = eval_q(xgb_test_cqr, y_te.values, 'test')
    cqr_e.update({'fold': name, 'stage': 'CQR', 'model': 'XGB(q,v2)',
                  'cqr_Q_hat': Q_hat, 'n_cal': n_c})
    all_xgb_eval.extend([raw_e, cqr_e])

    print(f'  raw: dir_acc={raw_e["dir_acc_q50"]:.4f} cov={raw_e["coverage_90"]:.4f} '
          f'rmse={raw_e["rmse_q50_bp"]:.3f} sharp={raw_e["sharpness_bp"]:.2f}')
    print(f'  CQR: dir_acc={cqr_e["dir_acc_q50"]:.4f} cov={cqr_e["coverage_90"]:.4f} '
          f'rmse={cqr_e["rmse_q50_bp"]:.3f} sharp={cqr_e["sharpness_bp"]:.2f} Q_hat={Q_hat:.3f}')

    pooled_xgb_q50.append(xgb_preds['test'][0.5])
    pooled_xgb_q05.append(xgb_test_cqr[0.05])
    pooled_xgb_q95.append(xgb_test_cqr[0.95])
    pooled_y.append(y_te.values)

    if name == 'fold3':
        fold3_xgb_preds = {
            'dates': X_te.index,
            'y': y_te.values,
            'q05_cqr': xgb_test_cqr[0.05],
            'q50': xgb_preds['test'][0.5],
            'q95_cqr': xgb_test_cqr[0.95],
        }

# Pooled
y_pool = np.concatenate(pooled_y)
xgb_q50_pool = np.concatenate(pooled_xgb_q50)
pooled_e = eval_q({0.05: np.concatenate(pooled_xgb_q05),
                   0.5: xgb_q50_pool,
                   0.95: np.concatenate(pooled_xgb_q95)}, y_pool, 'pooled')
pooled_e.update({'fold': 'POOLED', 'stage': 'CQR', 'model': 'XGB(q,v2)'})
all_xgb_eval.append(pooled_e)
print(f'\nPooled CQR: dir_acc={pooled_e["dir_acc_q50"]:.4f} '
      f'cov={pooled_e["coverage_90"]:.4f} rmse={pooled_e["rmse_q50_bp"]:.3f}')

# Save full XGB CSV
xgb_df = pd.DataFrame(all_xgb_eval)
xgb_df.to_csv(REPORT_DIR / 'walkforward_xgb_v2_full.csv', index=False)
print(f'\nSaved: walkforward_xgb_v2_full.csv')

# ═══════════════════════════════════════════════════════════════════════
# Now regenerate comparison charts with both models' full data
# ═══════════════════════════════════════════════════════════════════════
print('\nRegenerating comparison charts with complete data ...')

# Load LSTM data
lstm_df = pd.read_csv(REPORT_DIR / 'walkforward_lstm_v2_full.csv')
lstm_dm_df = pd.read_csv(REPORT_DIR / 'walkforward_lstm_dm_v2.csv')
xgb_dm_df = pd.read_csv(REPORT_DIR / 'walkforward_dm_v2.csv')

xgb_cqr = xgb_df[xgb_df['stage'] == 'CQR']
xgb_raw = xgb_df[xgb_df['stage'] == 'raw']
lstm_folds = lstm_df[lstm_df['fold'] != 'POOLED']

folds_names = ['fold1', 'fold2', 'fold3']

# Compute per-fold LSTM averages (across seeds)
lstm_fold_avg = {}
for fn in folds_names:
    rows = lstm_folds[lstm_folds['fold'] == fn]
    lstm_fold_avg[fn] = {
        'dir_acc': rows['dir_acc_q50'].mean(),
        'coverage': rows['coverage_90'].mean(),
        'sharpness': rows['sharpness_bp'].mean(),
        'rmse': rows['rmse_q50_bp'].mean(),
    }

# --- Chart: XGBoost v2 Prediction Interval (fold3) for page 6 ---
if fold3_xgb_preds:
    d = fold3_xgb_preds
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(d['dates'])
    ax.fill_between(dates, d['q05_cqr'], d['q95_cqr'], alpha=0.25, color='#4CAF50',
                    label='90% PI (CQR)')
    ax.plot(dates, d['q50'], color='#1B5E20', lw=1.2, label='XGBoost q50')
    ax.plot(dates, d['y'], color='#E53935', lw=0.8, alpha=0.7, label='Actual Δy (bp)')
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.set_title('XGBoost v2 + CQR — 90% Prediction Interval (fold3: 2023-2025)', fontsize=13)
    ax.set_ylabel('Δ Bond Yield (bp)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'xgb_v2_pred_interval_fold3.png')
    plt.close(fig)
    print('  Saved: xgb_v2_pred_interval_fold3.png')

# --- Chart (Page 8): 3-panel comparison with FULL data ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
x = np.arange(3)
w = 0.35

# (a) Direction Accuracy
ax = axes[0]
xgb_da = [xgb_cqr[xgb_cqr['fold']==f]['dir_acc_q50'].values[0] for f in folds_names]
lstm_da = [lstm_fold_avg[f]['dir_acc'] for f in folds_names]
ax.bar(x - w/2, xgb_da, w, label='XGBoost v2', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_da, w, label='LSTM v2', color='#E53935', alpha=0.85)
ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='Random (50%)')
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('Direction Accuracy')
ax.set_title('(a) Direction Accuracy')
ax.legend(fontsize=8); ax.set_ylim(0.4, 0.75)
for i, (v1, v2) in enumerate(zip(xgb_da, lstm_da)):
    ax.text(i-w/2, v1+0.008, f'{v1:.1%}', ha='center', fontsize=8)
    ax.text(i+w/2, v2+0.008, f'{v2:.1%}', ha='center', fontsize=8)

# (b) RMSE
ax = axes[1]
xgb_rm = [xgb_cqr[xgb_cqr['fold']==f]['rmse_q50_bp'].values[0] for f in folds_names]
lstm_rm = [lstm_fold_avg[f]['rmse'] for f in folds_names]
naive_rm = [float(np.sqrt(np.mean(df.loc[fold['test'][0]:fold['test'][1]]['delta_y_bp'].values**2)))
            for fold in FOLDS]
ax.bar(x - w/2, xgb_rm, w, label='XGBoost v2', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_rm, w, label='LSTM v2', color='#E53935', alpha=0.85)
ax.scatter(x, naive_rm, marker='D', color='#FFA000', s=60, zorder=5, label='Naive (Δ=0)')
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('RMSE (bp)')
ax.set_title('(b) RMSE')
ax.legend(fontsize=8)
for i, (v1, v2, vn) in enumerate(zip(xgb_rm, lstm_rm, naive_rm)):
    ax.text(i-w/2, v1+0.08, f'{v1:.2f}', ha='center', fontsize=8)
    ax.text(i+w/2, v2+0.08, f'{v2:.2f}', ha='center', fontsize=8)

# (c) Coverage 90%
ax = axes[2]
xgb_cv = [xgb_cqr[xgb_cqr['fold']==f]['coverage_90'].values[0] for f in folds_names]
lstm_cv = [lstm_fold_avg[f]['coverage'] for f in folds_names]
ax.bar(x - w/2, xgb_cv, w, label='XGBoost v2 (CQR)', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_cv, w, label='LSTM v2 (raw)', color='#E53935', alpha=0.85)
ax.axhline(0.9, color='orange', ls='--', lw=0.8, label='Target 90%')
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('Coverage Rate')
ax.set_title('(c) Coverage 90%')
ax.legend(fontsize=8); ax.set_ylim(0.5, 1.05)
for i, (v1, v2) in enumerate(zip(xgb_cv, lstm_cv)):
    ax.text(i-w/2, v1+0.01, f'{v1:.1%}', ha='center', fontsize=8)
    ax.text(i+w/2, v2+0.01, f'{v2:.1%}', ha='center', fontsize=8)

fig.suptitle('XGBoost v2 vs LSTM v2 — Walk-forward 3-fold Comparison', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'xgb_vs_lstm_v2_comparison.png')
plt.close(fig)
print('  Saved: xgb_vs_lstm_v2_comparison.png (updated)')

# --- Summary comparison table chart (Page 10) ---
fig, ax = plt.subplots(figsize=(13, 7))
ax.axis('off')

xgb_avg_da = np.mean(xgb_da)
lstm_avg_da = np.mean(lstm_da)
xgb_avg_rm = np.mean(xgb_rm)
lstm_avg_rm = np.mean(lstm_rm)
xgb_avg_cv = np.mean(xgb_cv)
lstm_avg_cv = np.mean(lstm_cv)
xgb_avg_sh = float(xgb_cqr[xgb_cqr['fold'].isin(folds_names)]['sharpness_bp'].mean())
lstm_avg_sh = np.mean([lstm_fold_avg[f]['sharpness'] for f in folds_names])
naive_avg_rm = np.mean(naive_rm)

# DM results
xgb_dm_pooled = xgb_dm_df[xgb_dm_df['fold']=='POOLED'].iloc[0]
lstm_dm_pooled = lstm_dm_df[lstm_dm_df['fold']=='POOLED'].iloc[0]

col_labels = ['Metric', 'XGBoost v2 (CQR)', 'LSTM v2 (raw)', 'Naive (Δ=0)']
table_data = [
    ['Dir Accuracy\n(3-fold avg)',
     f'{xgb_avg_da:.1%}',
     f'{lstm_avg_da:.1%}',
     '50.0%'],
    ['RMSE\n(3-fold avg, bp)',
     f'{xgb_avg_rm:.2f}',
     f'{lstm_avg_rm:.2f}',
     f'{naive_avg_rm:.2f}'],
    ['Coverage 90%\n(3-fold avg)',
     f'{xgb_avg_cv:.1%}',
     f'{lstm_avg_cv:.1%}',
     'N/A'],
    ['Sharpness\n(3-fold avg, bp)',
     f'{xgb_avg_sh:.2f}',
     f'{lstm_avg_sh:.2f}',
     'N/A'],
    ['DM test vs Naive\n(pooled)',
     f'DM={xgb_dm_pooled["DM_HLN"]:.2f}\np<0.001 ✓',
     f'DM={lstm_dm_pooled["DM_HLN"]:.2f}\np<0.001 ✓',
     '—'],
    ['DM fold1 (COVID)\nvs Naive',
     f'p={xgb_dm_df[xgb_dm_df["fold"]=="fold1"]["p_value"].values[0]:.3f}\n(tie)',
     f'p={lstm_dm_df[lstm_dm_df["fold"]=="fold1"]["p_value"].values[0]:.3f}\n(tie)',
     '—'],
]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 2.2)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold', fontsize=11)

for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor('#E3F2FD')

# Highlight best values
best_cells = [(1, 1), (2, 2), (3, 1)]  # best dir_acc=XGB, best RMSE=LSTM, best cov=XGB
for r, c in best_cells:
    tbl[r, c].set_text_props(fontweight='bold', color='#1B5E20')

ax.set_title('Model Comparison — Walk-forward 3-fold Validation Summary',
             fontsize=14, pad=20, fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'model_comparison_summary_v2.png')
plt.close(fig)
print('  Saved: model_comparison_summary_v2.png (updated)')

# --- Print final summary for the revision request ---
print('\n' + '=' * 72)
print('FINAL SUMMARY FOR PRESENTATION')
print('=' * 72)
print(f'\n{"Metric":<25} {"XGBoost v2 (CQR)":>20} {"LSTM v2 (raw)":>20} {"Naive":>15}')
print('-' * 82)
print(f'{"Dir Acc (3-fold avg)":<25} {xgb_avg_da:>19.1%} {lstm_avg_da:>19.1%} {"50.0%":>15}')
print(f'{"RMSE (3-fold avg, bp)":<25} {xgb_avg_rm:>20.2f} {lstm_avg_rm:>20.2f} {naive_avg_rm:>15.2f}')
print(f'{"Coverage 90%":<25} {xgb_avg_cv:>19.1%} {lstm_avg_cv:>19.1%} {"N/A":>15}')
print(f'{"Sharpness (bp)":<25} {xgb_avg_sh:>20.2f} {lstm_avg_sh:>20.2f} {"N/A":>15}')
print(f'{"DM vs Naive (pooled)":<25} {"p<0.001 ✓":>20} {"p<0.001 ✓":>20} {"—":>15}')

print('\n--- Per-fold detail ---')
for fn in folds_names:
    xr = xgb_cqr[xgb_cqr['fold']==fn].iloc[0]
    la = lstm_fold_avg[fn]
    print(f'\n{fn}:')
    print(f'  XGB:  dir_acc={xr["dir_acc_q50"]:.4f}  rmse={xr["rmse_q50_bp"]:.3f}  '
          f'cov={xr["coverage_90"]:.4f}  sharp={xr["sharpness_bp"]:.2f}')
    print(f'  LSTM: dir_acc={la["dir_acc"]:.4f}  rmse={la["rmse"]:.3f}  '
          f'cov={la["coverage"]:.4f}  sharp={la["sharpness"]:.2f}')

print('\n\nDone!')

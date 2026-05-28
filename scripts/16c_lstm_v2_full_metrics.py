"""
16c_lstm_v2_full_metrics.py
Re-run LSTM v2 walkforward 3-fold to collect ALL metrics
(coverage_90, sharpness_bp, rmse_q50_bp, dir_acc_q50, pinball)
+ DM test (LSTM vs Naive)
+ Generate charts for presentation pages 7, 8, 10
"""
from __future__ import annotations
import warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import t as t_dist
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
FIG_DIR = REPORT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
SEEDS = [42, 123, 2024]

print('=' * 72)
print('16c — LSTM v2 full metrics + charts')
print(f'Device: {DEVICE}')
print('=' * 72)

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

# ── Data ──
df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

RAW_INPUT_FOR_LSTM = [
    'kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
    'us_breakeven_10y','vix','kospi','sp500','dxy',
    'spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy',
]
RAW_INPUT_FOR_LSTM = [c for c in RAW_INPUT_FOR_LSTM if c in df.columns]
XGB_FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']
print(f'LSTM inputs: {len(RAW_INPUT_FOR_LSTM)}, XGB features: {len(XGB_FEATURE_COLS)}')

# ── Eval helpers ──
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

def dm_test_hln(e1, e2, lag=6):
    d = e1 - e2
    n = len(d)
    if n < 10: return float('nan'), float('nan')
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

# ── LSTM model ──
class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_q)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def pinball_loss_torch(pred, target, qs=QUANTILES):
    target = target.unsqueeze(1)
    q = torch.tensor(qs, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    diff = target - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_seq(X_df, y_ser, lookback):
    idx = X_df.index.intersection(y_ser.index)
    X_arr = X_df.loc[idx].to_numpy(dtype=np.float32)
    y_arr = y_ser.loc[idx].to_numpy(dtype=np.float32)
    valid = ~np.isnan(y_arr)
    seqs, tgts, dates = [], [], []
    date_index = X_df.loc[idx].index
    for t in range(lookback - 1, len(X_arr)):
        if not valid[t]: continue
        win = X_arr[t - lookback + 1: t + 1]
        if np.isnan(win).any(): continue
        seqs.append(win)
        tgts.append(y_arr[t])
        dates.append(date_index[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), dates

def train_lstm_one(seed, Xs_tr, ys_tr, Xs_val, ys_val):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = QuantileLSTM(input_dim=Xs_tr.shape[2],
                     hidden=LSTM_CFG['hidden_units'],
                     num_layers=LSTM_CFG['num_layers'],
                     dropout=LSTM_CFG['dropout'],
                     n_q=len(QUANTILES)).to(DEVICE)
    tr_ld = DataLoader(SeqDS(Xs_tr, ys_tr), batch_size=LSTM_CFG['batch_size'],
                       shuffle=True, drop_last=False)
    vl_ld = DataLoader(SeqDS(Xs_val, ys_val), batch_size=LSTM_CFG['batch_size'],
                       shuffle=False, drop_last=False)
    opt = torch.optim.Adam(m.parameters(), lr=LSTM_CFG['learning_rate'])
    best, best_st, wait = float('inf'), None, 0
    for ep in range(1, LSTM_CFG['epochs'] + 1):
        m.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pinball_loss_torch(m(xb), yb).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = [float(pinball_loss_torch(m(xb.to(DEVICE)), yb.to(DEVICE)).item())
                  for xb, yb in vl_ld]
        vl_loss = float(np.mean(vl))
        if vl_loss < best - 1e-6:
            best, wait = vl_loss, 0
            best_st = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= LSTM_CFG['early_stopping_patience']:
            break
    if best_st is not None:
        m.load_state_dict(best_st)
    return m, best, ep

@torch.no_grad()
def predict_lstm(m, Xs):
    m.eval()
    pred = m(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════
all_lstm_eval = []
all_lstm_dm = []
pooled_lstm_preds = {seed: {'q50': [], 'q05': [], 'q95': []} for seed in SEEDS}
pooled_lstm_y = []
pooled_lstm_dates = []

# For charts: store fold3 (test 2023-2025) predictions for page 7
fold3_preds = {}

for fold in FOLDS:
    name = fold['name']
    print(f'\n{"="*60}')
    print(f'{name}: train {fold["train"]}, test {fold["test"]}')
    print(f'{"="*60}')

    def sl(p): return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[XGB_FEATURE_COLS]
    X_val_raw = sl('val')[XGB_FEATURE_COLS]
    X_te_raw  = sl('test')[XGB_FEATURE_COLS]

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
    X_tr = s(X_tr_raw); X_val = s(X_val_raw); X_te = s(X_te_raw)

    y_tr = sl('train')['delta_y_bp']
    y_val = sl('val')['delta_y_bp']
    y_te  = sl('test')['delta_y_bp']

    X_tr_lstm = X_tr[RAW_INPUT_FOR_LSTM]
    X_val_lstm = X_val[RAW_INPUT_FOR_LSTM]
    X_te_lstm  = X_te[RAW_INPUT_FOR_LSTM]

    Xs_tr, ys_tr, _ = make_seq(X_tr_lstm, y_tr, LOOKBACK)
    Xs_val, ys_val, _ = make_seq(X_val_lstm, y_val, LOOKBACK)
    Xs_te, ys_te, dates_te = make_seq(X_te_lstm, y_te, LOOKBACK)
    print(f'  Xs_tr={Xs_tr.shape}, Xs_val={Xs_val.shape}, Xs_te={Xs_te.shape}')

    fold_q50_preds = []
    for seed in SEEDS:
        print(f'  Training seed={seed} ...', end=' ', flush=True)
        m, bv, ne = train_lstm_one(seed, Xs_tr, ys_tr, Xs_val, ys_val)
        p_te = sort_qs(predict_lstm(m, Xs_te))
        e = eval_q(p_te, ys_te, 'test')
        e.update({'fold': name, 'seed': seed, 'n_epochs': ne,
                  'best_val_pinball': bv, 'model': 'LSTM(q,v2)'})
        all_lstm_eval.append(e)
        fold_q50_preds.append(p_te[0.5])

        for qk in ['q50', 'q05', 'q95']:
            q_val = {0.5: 'q50', 0.05: 'q05', 0.95: 'q95'}
            pooled_lstm_preds[seed][qk].append(p_te[{v:k for k,v in q_val.items()}[qk]])

        print(f'epochs={ne} dir_acc={e["dir_acc_q50"]:.4f} '
              f'cov={e["coverage_90"]:.4f} sharp={e["sharpness_bp"]:.2f} '
              f'rmse={e["rmse_q50_bp"]:.3f}')

        if name == 'fold3':
            fold3_preds[seed] = {
                'dates': dates_te,
                'y': ys_te,
                'q05': p_te[0.05],
                'q50': p_te[0.5],
                'q95': p_te[0.95],
            }

    pooled_lstm_y.append(ys_te)
    pooled_lstm_dates.append(dates_te)

    # DM test: LSTM (seed avg) vs Naive per fold
    avg_q50 = np.mean(fold_q50_preds, axis=0)
    err_lstm = (ys_te - avg_q50) ** 2
    err_naive = ys_te ** 2
    dm_val, p_val = dm_test_hln(err_lstm, err_naive, lag=6)
    all_lstm_dm.append({
        'fold': name, 'comparison': 'LSTMv2_vs_Naive',
        'DM_HLN': dm_val, 'p_value': p_val,
        'winner': ('LSTM' if dm_val < 0 and p_val < 0.0167 else
                   'OPP' if dm_val > 0 and p_val < 0.0167 else 'tie')
    })
    print(f'  DM LSTM vs Naive: DM={dm_val:.3f}, p={p_val:.4f}')

# ── Pooled LSTM metrics ──
print('\n' + '=' * 60)
print('LSTM Pooled metrics')
y_pool = np.concatenate(pooled_lstm_y)
for seed in SEEDS:
    q50_pool = np.concatenate(pooled_lstm_preds[seed]['q50'])
    q05_pool = np.concatenate(pooled_lstm_preds[seed]['q05'])
    q95_pool = np.concatenate(pooled_lstm_preds[seed]['q95'])
    p_pool = {0.05: q05_pool, 0.5: q50_pool, 0.95: q95_pool}
    e = eval_q(p_pool, y_pool, 'pooled')
    e.update({'fold': 'POOLED', 'seed': seed, 'model': 'LSTM(q,v2)'})
    all_lstm_eval.append(e)
    print(f'  seed={seed}: dir_acc={e["dir_acc_q50"]:.4f} cov={e["coverage_90"]:.4f} '
          f'sharp={e["sharpness_bp"]:.2f} rmse={e["rmse_q50_bp"]:.3f}')

# Pooled DM (seed average)
avg_pool_q50 = np.mean([np.concatenate(pooled_lstm_preds[s]['q50']) for s in SEEDS], axis=0)
err_lstm_pool = (y_pool - avg_pool_q50) ** 2
err_naive_pool = y_pool ** 2
dm_p, p_p = dm_test_hln(err_lstm_pool, err_naive_pool, lag=6)
all_lstm_dm.append({
    'fold': 'POOLED', 'comparison': 'LSTMv2_vs_Naive',
    'DM_HLN': dm_p, 'p_value': p_p,
    'winner': ('LSTM' if dm_p < 0 and p_p < 0.0167 else
               'OPP' if dm_p > 0 and p_p < 0.0167 else 'tie')
})
print(f'  DM LSTM vs Naive (pooled): DM={dm_p:.3f}, p={p_p:.4f}')

# ═══════════════════════════════════════════════════════════════════════
# SAVE CSVs
# ═══════════════════════════════════════════════════════════════════════
lstm_df = pd.DataFrame(all_lstm_eval)
lstm_dm_df = pd.DataFrame(all_lstm_dm)

lstm_df.to_csv(REPORT_DIR / 'walkforward_lstm_v2_full.csv', index=False)
lstm_dm_df.to_csv(REPORT_DIR / 'walkforward_lstm_dm_v2.csv', index=False)
print(f'\nSaved: walkforward_lstm_v2_full.csv, walkforward_lstm_dm_v2.csv')

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '=' * 72)
print('LSTM v2 Walkforward Full Results')
print('=' * 72)

fold_metrics = lstm_df[lstm_df['fold'] != 'POOLED']
for fname in ['fold1', 'fold2', 'fold3']:
    rows = fold_metrics[fold_metrics['fold'] == fname]
    print(f'\n{fname}:')
    for _, r in rows.iterrows():
        print(f'  seed={r["seed"]}  dir_acc={r["dir_acc_q50"]:.4f}  '
              f'cov={r["coverage_90"]:.4f}  sharp={r["sharpness_bp"]:.2f}  '
              f'rmse={r["rmse_q50_bp"]:.3f}')
    print(f'  Mean: dir_acc={rows["dir_acc_q50"].mean():.4f}  '
          f'cov={rows["coverage_90"].mean():.4f}  '
          f'sharp={rows["sharpness_bp"].mean():.2f}  '
          f'rmse={rows["rmse_q50_bp"].mean():.3f}')

# Overall avg across folds (seed avg per fold, then fold avg)
fold_avgs = []
for fname in ['fold1', 'fold2', 'fold3']:
    rows = fold_metrics[fold_metrics['fold'] == fname]
    fold_avgs.append({
        'fold': fname,
        'dir_acc': rows['dir_acc_q50'].mean(),
        'coverage': rows['coverage_90'].mean(),
        'sharpness': rows['sharpness_bp'].mean(),
        'rmse': rows['rmse_q50_bp'].mean(),
    })
fa = pd.DataFrame(fold_avgs)
print(f'\n3-fold avg (seed-avg per fold):')
print(f'  dir_acc  = {fa["dir_acc"].mean():.4f} +/- {fa["dir_acc"].std():.4f}')
print(f'  coverage = {fa["coverage"].mean():.4f} +/- {fa["coverage"].std():.4f}')
print(f'  sharpness= {fa["sharpness"].mean():.2f} +/- {fa["sharpness"].std():.2f}')
print(f'  rmse     = {fa["rmse"].mean():.3f} +/- {fa["rmse"].std():.3f}')

# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════
print('\nGenerating charts ...')

# --- Chart 1 (Page 7): LSTM v2 prediction interval — fold3 seed=42 ---
if 42 in fold3_preds:
    d = fold3_preds[42]
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = pd.to_datetime(d['dates'])
    ax.fill_between(dates, d['q05'], d['q95'], alpha=0.25, color='#2196F3',
                    label='90% PI (q05–q95)')
    ax.plot(dates, d['q50'], color='#1565C0', lw=1.2, label='LSTM q50 prediction')
    ax.plot(dates, d['y'], color='#E53935', lw=0.8, alpha=0.7, label='Actual Δy (bp)')
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.set_title('LSTM v2 — 90% Prediction Interval (fold3: 2023-2025)', fontsize=13)
    ax.set_ylabel('Δ Bond Yield (bp)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'lstm_v2_pred_interval_fold3.png')
    plt.close(fig)
    print('  Saved: lstm_v2_pred_interval_fold3.png')

# --- Chart 2 (Page 8): XGBoost vs LSTM dir_acc bar chart per fold ---
xgb_csv = pd.read_csv(REPORT_DIR / 'walkforward_xgb_v2.csv')
xgb_cqr = xgb_csv[xgb_csv['stage'] == 'CQR']
xgb_da = xgb_cqr.set_index('fold')['dir_acc_q50']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# (a) Direction Accuracy
ax = axes[0]
folds_names = ['fold1', 'fold2', 'fold3']
xgb_vals = [xgb_da.get(f, 0) for f in folds_names]
lstm_vals = [fa[fa['fold']==f]['dir_acc'].values[0] for f in folds_names]
x = np.arange(3)
w = 0.35
ax.bar(x - w/2, xgb_vals, w, label='XGBoost v2', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_vals, w, label='LSTM v2', color='#E53935', alpha=0.85)
ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='Random (50%)')
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('Direction Accuracy')
ax.set_title('Direction Accuracy')
ax.legend(fontsize=8)
ax.set_ylim(0.4, 0.75)
for i, (xv, lv) in enumerate(zip(xgb_vals, lstm_vals)):
    ax.text(i-w/2, xv+0.005, f'{xv:.1%}', ha='center', fontsize=8)
    ax.text(i+w/2, lv+0.005, f'{lv:.1%}', ha='center', fontsize=8)

# (b) RMSE
ax = axes[1]
xgb_rmse_vals = []
for f in folds_names:
    row = xgb_cqr[xgb_cqr['fold']==f]
    if 'rmse_q50_bp' in row.columns and len(row) > 0:
        xgb_rmse_vals.append(row['rmse_q50_bp'].values[0])
    else:
        xgb_rmse_vals.append(0)
lstm_rmse_vals = [fa[fa['fold']==f]['rmse'].values[0] for f in folds_names]
ax.bar(x - w/2, xgb_rmse_vals, w, label='XGBoost v2', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_rmse_vals, w, label='LSTM v2', color='#E53935', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('RMSE (bp)')
ax.set_title('RMSE')
ax.legend(fontsize=8)
for i, (xv, lv) in enumerate(zip(xgb_rmse_vals, lstm_rmse_vals)):
    ax.text(i-w/2, xv+0.05, f'{xv:.2f}', ha='center', fontsize=8)
    ax.text(i+w/2, lv+0.05, f'{lv:.2f}', ha='center', fontsize=8)

# (c) Coverage 90%
ax = axes[2]
xgb_cov_vals = [xgb_cqr[xgb_cqr['fold']==f]['coverage_90'].values[0] for f in folds_names]
lstm_cov_vals = [fa[fa['fold']==f]['coverage'].values[0] for f in folds_names]
ax.bar(x - w/2, xgb_cov_vals, w, label='XGBoost v2 (CQR)', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, lstm_cov_vals, w, label='LSTM v2 (raw)', color='#E53935', alpha=0.85)
ax.axhline(0.9, color='orange', ls='--', lw=0.8, label='Target 90%')
ax.set_xticks(x); ax.set_xticklabels(folds_names)
ax.set_ylabel('Coverage Rate')
ax.set_title('Coverage 90%')
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.05)
for i, (xv, lv) in enumerate(zip(xgb_cov_vals, lstm_cov_vals)):
    ax.text(i-w/2, xv+0.01, f'{xv:.1%}', ha='center', fontsize=8)
    ax.text(i+w/2, lv+0.01, f'{lv:.1%}', ha='center', fontsize=8)

fig.suptitle('XGBoost v2 vs LSTM v2 — Walk-forward 3-fold Comparison', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'xgb_vs_lstm_v2_comparison.png')
plt.close(fig)
print('  Saved: xgb_vs_lstm_v2_comparison.png')

# --- Chart 3 (Page 10): Summary heatmap-style table ---
# Read XGB DM results
xgb_dm = pd.read_csv(REPORT_DIR / 'walkforward_dm_v2.csv')

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

# Build summary table data
col_labels = ['Metric', 'XGBoost v2 (CQR)', 'LSTM v2', 'Naive']
table_data = [
    ['Dir Accuracy (3-fold avg)',
     f'{fa["dir_acc"].mean():.1%} (XGB pooled {xgb_da.mean():.1%})',
     f'{fa["dir_acc"].mean():.1%}',
     '50% (random)'],
    ['RMSE (3-fold avg, bp)',
     f'{np.mean(xgb_rmse_vals):.2f}' if any(v > 0 for v in xgb_rmse_vals) else 'N/A',
     f'{fa["rmse"].mean():.2f}',
     f'{np.sqrt(np.mean(y_pool**2)):.2f}'],
    ['Coverage 90%',
     f'{np.mean(xgb_cov_vals):.1%}',
     f'{fa["coverage"].mean():.1%}',
     'N/A'],
    ['Sharpness (bp)',
     'CQR adjusted',
     f'{fa["sharpness"].mean():.2f}',
     'N/A'],
    ['DM vs Naive (pooled)',
     f'p<0.001 (XGB wins)',
     f'p={p_p:.4f}' + (' (LSTM wins)' if dm_p < 0 and p_p < 0.05 else ' (tie)'),
     '—'],
]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        if i % 2 == 0:
            tbl[i, j].set_facecolor('#E3F2FD')
        else:
            tbl[i, j].set_facecolor('#FFFFFF')

ax.set_title('Model Comparison Summary — Walk-forward 3-fold Validation',
             fontsize=14, pad=20)
fig.tight_layout()
fig.savefig(FIG_DIR / 'model_comparison_summary_v2.png')
plt.close(fig)
print('  Saved: model_comparison_summary_v2.png')

# --- Chart 4: LSTM v2 3-fold dir_acc by seed (for page 7) ---
fig, ax = plt.subplots(figsize=(8, 5))
colors_seeds = ['#1565C0', '#E53935', '#4CAF50']
for i, seed in enumerate(SEEDS):
    vals = [fold_metrics[(fold_metrics['fold']==f) & (fold_metrics['seed']==seed)]['dir_acc_q50'].values[0]
            for f in folds_names]
    ax.bar(np.arange(3) + (i-1)*0.25, vals, 0.22,
           label=f'seed={seed}', color=colors_seeds[i], alpha=0.85)
ax.axhline(0.5, color='grey', ls='--', lw=0.8, label='Random')
ax.set_xticks(range(3)); ax.set_xticklabels(folds_names)
ax.set_ylabel('Direction Accuracy')
ax.set_title('LSTM v2 Direction Accuracy by Seed (3-fold Walk-forward)')
ax.legend(fontsize=9)
ax.set_ylim(0.4, 0.75)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'lstm_v2_dir_acc_by_seed.png')
plt.close(fig)
print('  Saved: lstm_v2_dir_acc_by_seed.png')

print('\n' + '=' * 72)
print('DONE — All LSTM v2 metrics and charts generated')
print('=' * 72)

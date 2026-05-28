"""
16e_charts_v2.py — 발표용 차트 재생성 (중복 제거, 품질 개선)

생성할 차트:
1. page6_xgb_pred_interval.png  — XGBoost v2 예측 구간 (fold3)
2. page7_lstm_pred_interval.png — LSTM v2 예측 구간 (fold3)
3. page8_model_comparison.png   — XGB vs LSTM 3-fold 비교 (3-panel)
4. page10_summary_table.png     — 최종 요약 테이블 (삭제 → 발표 슬라이드에서 직접 표로)

→ summary table 이미지는 생성하지 않음 (발표 슬라이드에서 직접 표로 넣는 게 깔끔)
→ 예측 구간 차트 2개는 동일 스케일로 비교 가능하게
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
FIG_DIR = REPORT_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']

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

# fold3 only (2023-2025, 발표에서 보여줄 구간)
FOLD3 = {
    'train': ('2010-01-01','2021-12-31'), 'val': ('2022-01-01','2022-12-31'),
    'cal': ('2022-07-01','2022-12-31'), 'test': ('2023-01-01','2025-12-31'),
}

def sl(p): return df.loc[FOLD3[p][0]:FOLD3[p][1]]

XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}

def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}

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
    torch.manual_seed(seed); np.random.seed(seed)
    m = QuantileLSTM(input_dim=Xs_tr.shape[2], hidden=LSTM_CFG['hidden_units'],
                     num_layers=LSTM_CFG['num_layers'], dropout=LSTM_CFG['dropout'],
                     n_q=len(QUANTILES)).to(DEVICE)
    tr_ld = DataLoader(SeqDS(Xs_tr, ys_tr), batch_size=LSTM_CFG['batch_size'], shuffle=True)
    vl_ld = DataLoader(SeqDS(Xs_val, ys_val), batch_size=LSTM_CFG['batch_size'], shuffle=False)
    opt = torch.optim.Adam(m.parameters(), lr=LSTM_CFG['learning_rate'])
    best, best_st, wait = float('inf'), None, 0
    for ep in range(1, LSTM_CFG['epochs'] + 1):
        m.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); pinball_loss_torch(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = [float(pinball_loss_torch(m(xb.to(DEVICE)), yb.to(DEVICE)).item()) for xb, yb in vl_ld]
        vl_loss = float(np.mean(vl))
        if vl_loss < best - 1e-6:
            best, wait = vl_loss, 0
            best_st = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= LSTM_CFG['early_stopping_patience']: break
    if best_st: m.load_state_dict(best_st)
    return m

@torch.no_grad()
def predict_lstm(m, Xs):
    m.eval()
    pred = m(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}

# ═════════════════════════════════════════════════
# Step 1: Train fold3 models
# ═════════════════════════════════════════════════
print('Training fold3 models for charts ...')

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

# XGBoost
print('  XGBoost q05/q50/q95 ...', flush=True)
xgb_preds = {'cal': {}, 'test': {}}
for q in QUANTILES:
    p = XGB_BEST[q]
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q,
                          n_estimators=p['n_estimators'], max_depth=p['max_depth'],
                          learning_rate=p['learning_rate'], early_stopping_rounds=50,
                          verbosity=0, tree_method='hist', random_state=42)
    m.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], verbose=False)
    xgb_preds['cal'][q] = m.predict(X_cal.values)
    xgb_preds['test'][q] = m.predict(X_te.values)
xgb_preds['cal'] = sort_qs(xgb_preds['cal'])
xgb_preds['test'] = sort_qs(xgb_preds['test'])

# CQR
sc = np.maximum(xgb_preds['cal'][0.05] - y_cal.values, y_cal.values - xgb_preds['cal'][0.95])
n_c = len(sc)
k = min(int(np.ceil((n_c + 1) * (1 - 0.10))), n_c)
Q_hat = float(np.sort(sc)[k - 1])
print(f'  CQR Q_hat = {Q_hat:.3f} bp')

xgb_raw = xgb_preds['test']
xgb_cqr = {
    0.05: xgb_preds['test'][0.05] - Q_hat,
    0.5:  xgb_preds['test'][0.5],
    0.95: xgb_preds['test'][0.95] + Q_hat,
}

# LSTM (seed=42 for chart)
print('  LSTM seed=42 ...', flush=True)
X_tr_lstm = X_tr[RAW_INPUT_FOR_LSTM]
X_val_lstm = X_val[RAW_INPUT_FOR_LSTM]
X_te_lstm  = X_te[RAW_INPUT_FOR_LSTM]

Xs_tr, ys_tr, _ = make_seq(X_tr_lstm, y_tr, LOOKBACK)
Xs_val, ys_val, _ = make_seq(X_val_lstm, y_val, LOOKBACK)
Xs_te, ys_te, dates_te = make_seq(X_te_lstm, y_te, LOOKBACK)

lstm_model = train_lstm_one(42, Xs_tr, ys_tr, Xs_val, ys_val)
lstm_preds = sort_qs(predict_lstm(lstm_model, Xs_te))
print('  Done.')

# ═════════════════════════════════════════════════
# Chart 1: Side-by-side prediction intervals (XGB raw vs LSTM)
# ═════════════════════════════════════════════════
print('Generating charts ...')

xgb_dates = pd.to_datetime(X_te.index)
lstm_dates = pd.to_datetime(dates_te)

# Compute interval widths for annotation
xgb_raw_width = xgb_raw[0.95] - xgb_raw[0.05]
xgb_cqr_width = xgb_cqr[0.95] - xgb_cqr[0.05]
lstm_width = lstm_preds[0.95] - lstm_preds[0.05]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# (a) XGBoost — raw intervals (NOT CQR, to show the model's own uncertainty)
ax = axes[0]
ax.fill_between(xgb_dates, xgb_raw[0.05], xgb_raw[0.95],
                alpha=0.3, color='#1565C0', label=f'90% PI (raw, avg width {xgb_raw_width.mean():.1f} bp)')
ax.plot(xgb_dates, xgb_raw[0.5], color='#0D47A1', lw=1.0, label='q50 prediction')
ax.plot(xgb_dates, y_te.values, color='#E53935', lw=0.6, alpha=0.6, label='Actual Δy')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title('(a) XGBoost v2 — Raw 90% Prediction Interval', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-25, 25)
ax.grid(axis='y', alpha=0.2)

# (b) LSTM
ax = axes[1]
ax.fill_between(lstm_dates, lstm_preds[0.05], lstm_preds[0.95],
                alpha=0.3, color='#E53935', label=f'90% PI (avg width {lstm_width.mean():.1f} bp)')
ax.plot(lstm_dates, lstm_preds[0.5], color='#B71C1C', lw=1.0, label='q50 prediction')
ax.plot(lstm_dates, ys_te, color='#1565C0', lw=0.6, alpha=0.6, label='Actual Δy')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title('(b) LSTM v2 — Raw 90% Prediction Interval', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-25, 25)
ax.grid(axis='y', alpha=0.2)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

fig.suptitle('Prediction Interval Comparison — fold3 Test (2023-2025)',
             fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'page6_7_pred_interval_comparison.png', dpi=150)
plt.close(fig)
print('  1. page6_7_pred_interval_comparison.png')

# ═════════════════════════════════════════════════
# Chart 2: XGBoost CQR effect (raw vs CQR)
# ═════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax = axes[0]
ax.fill_between(xgb_dates, xgb_raw[0.05], xgb_raw[0.95],
                alpha=0.3, color='#4CAF50', label=f'Raw PI (width {xgb_raw_width.mean():.1f} bp)')
ax.plot(xgb_dates, xgb_raw[0.5], color='#2E7D32', lw=1.0, label='q50')
ax.plot(xgb_dates, y_te.values, color='#E53935', lw=0.6, alpha=0.6, label='Actual')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title(f'(a) XGBoost v2 Before CQR — Coverage {np.mean((y_te.values >= xgb_raw[0.05]) & (y_te.values <= xgb_raw[0.95])):.1%}', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-30, 30)
ax.grid(axis='y', alpha=0.2)

ax = axes[1]
ax.fill_between(xgb_dates, xgb_cqr[0.05], xgb_cqr[0.95],
                alpha=0.3, color='#1565C0', label=f'CQR PI (width {xgb_cqr_width.mean():.1f} bp, Q̂=+{Q_hat:.1f}bp)')
ax.plot(xgb_dates, xgb_cqr[0.5], color='#0D47A1', lw=1.0, label='q50 (unchanged)')
ax.plot(xgb_dates, y_te.values, color='#E53935', lw=0.6, alpha=0.6, label='Actual')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
cqr_cov = np.mean((y_te.values >= xgb_cqr[0.05]) & (y_te.values <= xgb_cqr[0.95]))
ax.set_title(f'(b) XGBoost v2 After CQR — Coverage {cqr_cov:.1%} (target 90%)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-30, 30)
ax.grid(axis='y', alpha=0.2)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)

fig.suptitle('CQR Conformal Calibration Effect — XGBoost v2 fold3',
             fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'page6_cqr_effect.png', dpi=150)
plt.close(fig)
print('  2. page6_cqr_effect.png')

# ═════════════════════════════════════════════════
# Chart 3: Interval width time series comparison
# ═════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4.5))

ax.plot(xgb_dates, xgb_raw_width, color='#1565C0', lw=0.8, alpha=0.7, label=f'XGBoost raw ({xgb_raw_width.mean():.1f} ± {xgb_raw_width.std():.1f} bp)')
ax.plot(lstm_dates, lstm_width, color='#E53935', lw=0.8, alpha=0.7, label=f'LSTM v2 ({lstm_width.mean():.1f} ± {lstm_width.std():.1f} bp)')
ax.axhline(xgb_cqr_width.mean(), color='#1565C0', ls='--', lw=1.0, alpha=0.5, label=f'XGBoost CQR (≈{xgb_cqr_width.mean():.1f} bp, nearly constant)')
ax.set_ylabel('Interval Width (bp)', fontsize=11)
ax.set_title('90% Prediction Interval Width Over Time — XGBoost vs LSTM', fontsize=13, fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)
plt.xticks(rotation=45)
fig.tight_layout()
fig.savefig(FIG_DIR / 'page8_interval_width_comparison.png', dpi=150)
plt.close(fig)
print('  3. page8_interval_width_comparison.png')

# ═════════════════════════════════════════════════
# Chart 4: 3-fold comparison bar chart (improved)
# ═════════════════════════════════════════════════
lstm_full = pd.read_csv(REPORT_DIR / 'walkforward_lstm_v2_full.csv')
xgb_full = pd.read_csv(REPORT_DIR / 'walkforward_xgb_v2_full.csv')
xgb_cqr_df = xgb_full[(xgb_full['stage']=='CQR') & (xgb_full['fold'].isin(['fold1','fold2','fold3']))]
xgb_raw_df = xgb_full[(xgb_full['stage']=='raw') & (xgb_full['fold'].isin(['fold1','fold2','fold3']))]
lstm_folds = lstm_full[lstm_full['fold'].isin(['fold1','fold2','fold3'])]

fnames = ['fold1', 'fold2', 'fold3']
fold_labels = ['fold1\n(COVID 2020)', 'fold2\n(금리인상 21-22)', 'fold3\n(안정+충격 23-25)']

# Per-fold LSTM seed averages
lstm_avg = {}
for fn in fnames:
    rows = lstm_folds[lstm_folds['fold']==fn]
    lstm_avg[fn] = {c: rows[c].mean() for c in ['dir_acc_q50','rmse_q50_bp','coverage_90','sharpness_bp']}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
x = np.arange(3)
w = 0.32

# (a) Direction Accuracy
ax = axes[0, 0]
xgb_da = [xgb_cqr_df[xgb_cqr_df['fold']==f]['dir_acc_q50'].values[0] for f in fnames]
lstm_da = [lstm_avg[f]['dir_acc_q50'] for f in fnames]
b1 = ax.bar(x - w/2, xgb_da, w, label='XGBoost v2', color='#1565C0', alpha=0.85, edgecolor='white')
b2 = ax.bar(x + w/2, lstm_da, w, label='LSTM v2', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0.5, color='#888', ls=':', lw=1, label='Random 50%')
ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=9)
ax.set_ylabel('Direction Accuracy'); ax.set_title('(a) Direction Accuracy', fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.set_ylim(0.45, 0.72)
for i, (v1, v2) in enumerate(zip(xgb_da, lstm_da)):
    ax.text(i-w/2, v1+0.005, f'{v1:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(i+w/2, v2+0.005, f'{v2:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.grid(axis='y', alpha=0.15)

# (b) RMSE
ax = axes[0, 1]
xgb_rm = [xgb_cqr_df[xgb_cqr_df['fold']==f]['rmse_q50_bp'].values[0] for f in fnames]
lstm_rm = [lstm_avg[f]['rmse_q50_bp'] for f in fnames]
naive_rm = [float(np.sqrt(np.mean(df.loc[FOLD3['test'][0]:FOLD3['test'][1]]['delta_y_bp'].values**2)))] * 3
# Per-fold naive
for i, fn in enumerate(fnames):
    fold_def = [{'test':('2020-01-01','2020-12-31')},{'test':('2021-01-01','2022-12-31')},{'test':('2023-01-01','2025-12-31')}][i]
    naive_rm[i] = float(np.sqrt(np.mean(df.loc[fold_def['test'][0]:fold_def['test'][1]]['delta_y_bp'].values**2)))
ax.bar(x - w/2, xgb_rm, w, label='XGBoost v2', color='#1565C0', alpha=0.85, edgecolor='white')
ax.bar(x + w/2, lstm_rm, w, label='LSTM v2', color='#E53935', alpha=0.85, edgecolor='white')
ax.scatter(x, naive_rm, marker='D', color='#FF8F00', s=50, zorder=5, label='Naive (Δ=0)')
ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=9)
ax.set_ylabel('RMSE (bp)'); ax.set_title('(b) RMSE (lower is better)', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
for i, (v1, v2) in enumerate(zip(xgb_rm, lstm_rm)):
    ax.text(i-w/2, v1+0.05, f'{v1:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(i+w/2, v2+0.05, f'{v2:.2f}', ha='center', va='bottom', fontsize=8)
ax.grid(axis='y', alpha=0.15)

# (c) Coverage 90%
ax = axes[1, 0]
xgb_cv_raw = [xgb_raw_df[xgb_raw_df['fold']==f]['coverage_90'].values[0] for f in fnames]
xgb_cv = [xgb_cqr_df[xgb_cqr_df['fold']==f]['coverage_90'].values[0] for f in fnames]
lstm_cv = [lstm_avg[f]['coverage_90'] for f in fnames]
ax.bar(x - w, xgb_cv_raw, w*0.9, label='XGB raw', color='#64B5F6', alpha=0.85, edgecolor='white')
ax.bar(x, xgb_cv, w*0.9, label='XGB CQR', color='#1565C0', alpha=0.85, edgecolor='white')
ax.bar(x + w, lstm_cv, w*0.9, label='LSTM v2', color='#E53935', alpha=0.85, edgecolor='white')
ax.axhline(0.9, color='#FF8F00', ls='--', lw=1.2, label='Target 90%')
ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=9)
ax.set_ylabel('Coverage Rate'); ax.set_title('(c) Coverage 90%', fontweight='bold')
ax.legend(fontsize=8, loc='lower left'); ax.set_ylim(0.55, 1.02)
for i, (v1, v2, v3) in enumerate(zip(xgb_cv_raw, xgb_cv, lstm_cv)):
    ax.text(i-w, v1+0.008, f'{v1:.1%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.text(i, v2+0.008, f'{v2:.1%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.text(i+w, v3+0.008, f'{v3:.1%}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax.grid(axis='y', alpha=0.15)

# (d) Sharpness
ax = axes[1, 1]
xgb_sh_raw = [xgb_raw_df[xgb_raw_df['fold']==f]['sharpness_bp'].values[0] for f in fnames]
xgb_sh_cqr = [xgb_cqr_df[xgb_cqr_df['fold']==f]['sharpness_bp'].values[0] for f in fnames]
lstm_sh = [lstm_avg[f]['sharpness_bp'] for f in fnames]
ax.bar(x - w, xgb_sh_raw, w*0.9, label='XGB raw', color='#64B5F6', alpha=0.85, edgecolor='white')
ax.bar(x, xgb_sh_cqr, w*0.9, label='XGB CQR', color='#1565C0', alpha=0.85, edgecolor='white')
ax.bar(x + w, lstm_sh, w*0.9, label='LSTM v2', color='#E53935', alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(fold_labels, fontsize=9)
ax.set_ylabel('Sharpness (bp)'); ax.set_title('(d) Sharpness (narrower = better)', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
for i, (v1, v2, v3) in enumerate(zip(xgb_sh_raw, xgb_sh_cqr, lstm_sh)):
    ax.text(i-w, v1+0.2, f'{v1:.1f}', ha='center', va='bottom', fontsize=7)
    ax.text(i, v2+0.2, f'{v2:.1f}', ha='center', va='bottom', fontsize=7)
    ax.text(i+w, v3+0.2, f'{v3:.1f}', ha='center', va='bottom', fontsize=7)
ax.grid(axis='y', alpha=0.15)

fig.suptitle('XGBoost v2 vs LSTM v2 — Walk-forward 3-fold Comparison',
             fontsize=15, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'page8_4panel_comparison.png', dpi=150)
plt.close(fig)
print('  4. page8_4panel_comparison.png')

# ═════════════════════════════════════════════════
# Print summary of generated charts
# ═════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Generated charts:')
print(f'  1. page6_7_pred_interval_comparison.png — XGB vs LSTM 예측 구간 비교')
print(f'  2. page6_cqr_effect.png — XGBoost CQR 보정 전후 비교')
print(f'  3. page8_interval_width_comparison.png — 구간 폭 시계열 비교')
print(f'  4. page8_4panel_comparison.png — 4-panel fold별 비교 (dir_acc/RMSE/Coverage/Sharpness)')
print(f'\n구 차트 (삭제 가능):')
print(f'  - xgb_vs_lstm_v2_comparison.png (→ page8_4panel_comparison.png 으로 대체)')
print(f'  - model_comparison_summary_v2.png (→ 발표 슬라이드에서 직접 표로)')
print(f'  - lstm_v2_pred_interval_fold3.png (→ page6_7_pred_interval_comparison.png 에 통합)')
print(f'  - xgb_v2_pred_interval_fold3.png (→ 통합)')
print(f'  - lstm_v2_dir_acc_by_seed.png (→ page8_4panel 에 dir_acc 포함)')
print('=' * 60)

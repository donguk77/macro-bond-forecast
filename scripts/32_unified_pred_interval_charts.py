"""
32_unified_pred_interval_charts.py — 통일 색상 예측 구간 시각화 (v2)

변경사항:
  - Actual: 선(line) — 회색
  - 색상 통일: XGBoost=파랑, LSTM=빨강
  - v0 split: train 2010-2020, val 2022, test 2023-2025 (원본 13_rerun 동일)
  - v2 split: train 2010-2021, val 2022, test 2023-2025 (fold3 동일)
  - v0 seeds: [42, 123, 2024] (원본 동일)
  - v2 seeds: [42, 123, 7]
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
FIG_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
LAGS = CONFIG['features']['lags']
ROLL_WINDOWS = [5, 10, 20]

# Color palette
C_XGB_BAND = '#1565C0'
C_XGB_Q50  = '#0D47A1'
C_LSTM_BAND = '#E53935'
C_LSTM_Q50  = '#B71C1C'
C_ACTUAL   = '#455A64'
C_CQR_BAND = '#42A5F5'

# Separate splits for v0 and v2
SPLIT_V0 = {
    'train': ('2010-01-01','2020-12-31'),
    'val':   ('2022-01-01','2022-12-31'),
    'test':  ('2023-01-01','2025-12-31'),
}
SPLIT_V2 = {
    'train': ('2010-01-01','2021-12-31'),
    'val':   ('2022-01-01','2022-12-31'),
    'cal':   ('2022-07-01','2022-12-31'),
    'test':  ('2023-01-01','2025-12-31'),
}

SEEDS_V0 = [42, 123, 2024]
SEEDS_V2 = [42, 123, 7]

XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}

# ══════════════════════════════════════════════════════════════
# LSTM model
# ══════════════════════════════════════════════════════════════
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
    return m, ep, best

@torch.no_grad()
def predict_lstm(m, Xs):
    m.eval()
    pred = m(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}

def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


# ══════════════════════════════════════════════════════════════
# Build v0 features (leaked — NO shift on US market vars)
# ══════════════════════════════════════════════════════════════
print('Building v0 features (leaked) ...', flush=True)

v0_raw = pd.read_csv(DATA_DIR / 'processed' / 'features_v1_candidate.csv',
                     index_col='date', parse_dates=['date']).sort_index()
v0_raw = v0_raw.dropna(subset=['kr_treasury_10y'])

POLICY_VARS = ['kr_base_rate', 'us_fed_funds']
V0_FEATURES_RAW = [c for c in v0_raw.columns if c != 'kr_treasury_10y']

v0_safe = v0_raw.copy()
for var in POLICY_VARS:
    if var in v0_safe.columns:
        v0_safe[var] = v0_safe[var].shift(1)
v0_safe = v0_safe.dropna(subset=[v for v in POLICY_VARS if v in v0_safe.columns])

y_bp_v0 = (v0_safe['kr_treasury_10y'].diff() * 100).rename('delta_y_bp')

lag_blocks = [v0_safe[c].shift(k).rename(f'{c}__lag{k}')
              for c in V0_FEATURES_RAW for k in LAGS]
roll_blocks = []
for c in V0_FEATURES_RAW:
    for w in ROLL_WINDOWS:
        roll_blocks.append(v0_safe[c].rolling(w).mean().shift(1).rename(f'{c}__rmean{w}'))
        roll_blocks.append(v0_safe[c].rolling(w).std().shift(1).rename(f'{c}__rstd{w}'))

df_v0 = pd.concat(
    [v0_safe[V0_FEATURES_RAW], pd.concat(lag_blocks, axis=1),
     pd.concat(roll_blocks, axis=1), y_bp_v0.to_frame()],
    axis=1
).dropna()
print(f'  v0 features: {df_v0.shape}')

V0_FEAT_COLS = [c for c in df_v0.columns if c != 'delta_y_bp']

# LSTM v0 uses Δfeature[t-1] (diff+shift(1)) — W5 approach that got 63-65%
FROZEN_W3 = ['kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
             'us_breakeven_10y','vix','sp500','dxy']
df_v0_diff = v0_raw[FROZEN_W3].diff().shift(1)
df_v0_diff.columns = [f'd_{c}' for c in FROZEN_W3]
for c in df_v0_diff.columns:
    df_v0[c] = df_v0_diff.loc[df_v0.index, c]
df_v0 = df_v0.dropna()
V0_FEAT_COLS = [c for c in df_v0.columns if c != 'delta_y_bp']
V0_LSTM_INPUT = [f'd_{c}' for c in FROZEN_W3]  # 8 Δfeature columns

# ══════════════════════════════════════════════════════════════
# Load v2 features
# ══════════════════════════════════════════════════════════════
print('Loading v2 features ...', flush=True)

df_v2 = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                    index_col='date', parse_dates=['date']).sort_index()
V2_FEAT_COLS = [c for c in df_v2.columns if c != 'delta_y_bp']
V2_LSTM_INPUT = [
    'kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
    'us_breakeven_10y','vix','kospi','sp500','dxy',
    'spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy',
]
V2_LSTM_INPUT = [c for c in V2_LSTM_INPUT if c in df_v2.columns]
print(f'  v2 features: {df_v2.shape}')


# ══════════════════════════════════════════════════════════════
# Train function — XGBoost
# ══════════════════════════════════════════════════════════════
def train_xgb(df, feat_cols, split, label):
    print(f'  Training XGBoost {label} ...', flush=True)
    def _sl(p): return df.loc[split[p][0]:split[p][1]]

    X_tr = _sl('train')[feat_cols]; y_tr = _sl('train')['delta_y_bp']
    X_val = _sl('val')[feat_cols]; y_val = _sl('val')['delta_y_bp']
    X_te = _sl('test')[feat_cols]; y_te = _sl('test')['delta_y_bp']

    scaler = RobustScaler().fit(X_tr)
    def _s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=feat_cols)
    X_tr_s, X_val_s, X_te_s = _s(X_tr), _s(X_val), _s(X_te)

    preds_test = {}
    for q in QUANTILES:
        p = XGB_BEST[q]
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q,
                              n_estimators=p['n_estimators'], max_depth=p['max_depth'],
                              learning_rate=p['learning_rate'], early_stopping_rounds=50,
                              verbosity=0, tree_method='hist', random_state=42)
        m.fit(X_tr_s.values, y_tr.values, eval_set=[(X_val_s.values, y_val.values)], verbose=False)
        preds_test[q] = m.predict(X_te_s.values)

    preds_test = sort_qs(preds_test)

    # CQR (use cal if available, else last half of val)
    if 'cal' in split:
        X_cal = _sl('cal')[feat_cols]; y_cal = _sl('cal')['delta_y_bp']
    else:
        X_cal = X_val.iloc[len(X_val)//2:]; y_cal = y_val.iloc[len(y_val)//2:]
    X_cal_s = pd.DataFrame(scaler.transform(X_cal), index=X_cal.index, columns=feat_cols)
    preds_cal = {}
    for q in QUANTILES:
        p = XGB_BEST[q]
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q,
                              n_estimators=p['n_estimators'], max_depth=p['max_depth'],
                              learning_rate=p['learning_rate'], early_stopping_rounds=50,
                              verbosity=0, tree_method='hist', random_state=42)
        m.fit(X_tr_s.values, y_tr.values, eval_set=[(X_val_s.values, y_val.values)], verbose=False)
        preds_cal[q] = m.predict(X_cal_s.values)
    preds_cal = sort_qs(preds_cal)

    sc = np.maximum(preds_cal[0.05] - y_cal.values, y_cal.values - preds_cal[0.95])
    n_c = len(sc)
    k = min(int(np.ceil((n_c + 1) * (1 - 0.10))), n_c)
    Q_hat = float(np.sort(sc)[k - 1])

    preds_cqr = {
        0.05: preds_test[0.05] - Q_hat,
        0.5:  preds_test[0.5],
        0.95: preds_test[0.95] + Q_hat,
    }

    dates = pd.to_datetime(X_te.index)
    y_vals = y_te.values
    return preds_test, preds_cqr, dates, y_vals, Q_hat


def train_lstm_multi(df, lstm_input_cols, feat_cols, split, seeds, label):
    print(f'  Training LSTM {label} ({len(seeds)}-seed avg) ...', flush=True)
    def _sl(p): return df.loc[split[p][0]:split[p][1]]

    X_tr = _sl('train')[feat_cols]; y_tr = _sl('train')['delta_y_bp']
    X_val = _sl('val')[feat_cols]; y_val = _sl('val')['delta_y_bp']
    X_te = _sl('test')[feat_cols]; y_te = _sl('test')['delta_y_bp']

    scaler = RobustScaler().fit(X_tr)
    def _s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=feat_cols)
    X_tr_s, X_val_s, X_te_s = _s(X_tr), _s(X_val), _s(X_te)

    input_cols = [c for c in lstm_input_cols if c in X_tr_s.columns]
    print(f'    LSTM input cols: {len(input_cols)}', flush=True)

    Xs_tr, ys_tr, _ = make_seq(X_tr_s[input_cols], y_tr, LOOKBACK)
    Xs_val, ys_val, _ = make_seq(X_val_s[input_cols], y_val, LOOKBACK)
    Xs_te, ys_te, dates_te = make_seq(X_te_s[input_cols], y_te, LOOKBACK)

    print(f'    Xs_tr={Xs_tr.shape}, Xs_val={Xs_val.shape}, Xs_te={Xs_te.shape}', flush=True)

    all_preds = []
    for seed in seeds:
        m, ep, best_val = train_lstm_one(seed, Xs_tr, ys_tr, Xs_val, ys_val)
        p = sort_qs(predict_lstm(m, Xs_te))
        dir_mask = (np.sign(p[0.5]) != 0) & (np.sign(ys_te) != 0)
        da = float((np.sign(p[0.5][dir_mask]) == np.sign(ys_te[dir_mask])).mean()) if dir_mask.sum() > 0 else 0
        print(f'    seed={seed}: ep={ep}, val_loss={best_val:.4f}, dir_acc={da:.1%}', flush=True)
        all_preds.append(p)

    avg_preds = {}
    for q in QUANTILES:
        avg_preds[q] = np.mean([p[q] for p in all_preds], axis=0)
    avg_preds = sort_qs(avg_preds)

    dates = pd.to_datetime(dates_te)
    return avg_preds, dates, ys_te


# ══════════════════════════════════════════════════════════════
# Train all 4 models
# ══════════════════════════════════════════════════════════════
print('\n=== Training models ===')

xgb_v0_raw, xgb_v0_cqr, xgb_v0_dates, xgb_v0_y, _ = train_xgb(df_v0, V0_FEAT_COLS, SPLIT_V0, 'v0')
xgb_v2_raw, xgb_v2_cqr, xgb_v2_dates, xgb_v2_y, xgb_v2_qhat = train_xgb(df_v2, V2_FEAT_COLS, SPLIT_V2, 'v2')
lstm_v0_preds, lstm_v0_dates, lstm_v0_y = train_lstm_multi(df_v0, V0_LSTM_INPUT, V0_FEAT_COLS, SPLIT_V0, SEEDS_V0, 'v0')
lstm_v2_preds, lstm_v2_dates, lstm_v2_y = train_lstm_multi(df_v2, V2_LSTM_INPUT, V2_FEAT_COLS, SPLIT_V2, SEEDS_V2, 'v2')

print('\n=== Generating charts ===')


# ══════════════════════════════════════════════════════════════
# Plotting helper
# ══════════════════════════════════════════════════════════════
def plot_pred_interval(ax, dates, y_true, preds, color_band, color_q50,
                       ylim=(-25, 25)):
    width = preds[0.95] - preds[0.05]
    cov = np.mean((y_true >= preds[0.05]) & (y_true <= preds[0.95]))
    dir_mask = (np.sign(preds[0.5]) != 0) & (np.sign(y_true) != 0)
    dir_acc = float((np.sign(preds[0.5][dir_mask]) == np.sign(y_true[dir_mask])).mean()) if dir_mask.sum() > 0 else 0.0

    ax.fill_between(dates, preds[0.05], preds[0.95],
                    alpha=0.25, color=color_band,
                    label=f'90% PI (avg width {width.mean():.1f} bp)')
    ax.plot(dates, preds[0.5], color=color_q50, lw=0.9, label='q50 prediction')
    ax.plot(dates, y_true, color=C_ACTUAL, lw=0.5, alpha=0.5, label='Actual Δy')
    ax.axhline(0, color='grey', ls='--', lw=0.4)
    ax.set_ylabel('Δy (bp)', fontsize=11)
    ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.2)

    info = f'Dir={dir_acc:.1%}  Cov={cov:.1%}  Width={width.mean():.1f}bp'
    ax.text(0.98, 0.97, info, transform=ax.transAxes, fontsize=9,
            ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.85, edgecolor='#ccc'))


def finalize_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


# ══════════════════════════════════════════════════════════════
# 1. XGBoost v0
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
plot_pred_interval(ax, xgb_v0_dates, xgb_v0_y, xgb_v0_raw, C_XGB_BAND, C_XGB_Q50)
ax.set_title('XGBoost v0 (누수 포함) — Raw 90% Prediction Interval', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
finalize_time_axis(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pred_interval_xgb_v0.png', dpi=150)
plt.close(fig)
print('  1. pred_interval_xgb_v0.png')


# ══════════════════════════════════════════════════════════════
# 2. LSTM v0
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
plot_pred_interval(ax, lstm_v0_dates, lstm_v0_y, lstm_v0_preds, C_LSTM_BAND, C_LSTM_Q50)
ax.set_title('LSTM v0 (누수 포함) — Raw 90% Prediction Interval', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
finalize_time_axis(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pred_interval_lstm_v0.png', dpi=150)
plt.close(fig)
print('  2. pred_interval_lstm_v0.png')


# ══════════════════════════════════════════════════════════════
# 3. XGBoost v2
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
plot_pred_interval(ax, xgb_v2_dates, xgb_v2_y, xgb_v2_raw, C_XGB_BAND, C_XGB_Q50)
ax.set_title('XGBoost v2 (누수 수정 + 파생변수) — Raw 90% Prediction Interval', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
finalize_time_axis(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pred_interval_xgb_v2.png', dpi=150)
plt.close(fig)
print('  3. pred_interval_xgb_v2.png')


# ══════════════════════════════════════════════════════════════
# 4. LSTM v2
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
plot_pred_interval(ax, lstm_v2_dates, lstm_v2_y, lstm_v2_preds, C_LSTM_BAND, C_LSTM_Q50)
ax.set_title('LSTM v2 (누수 수정 + 파생변수) — Raw 90% Prediction Interval', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
finalize_time_axis(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pred_interval_lstm_v2.png', dpi=150)
plt.close(fig)
print('  4. pred_interval_lstm_v2.png')


# ══════════════════════════════════════════════════════════════
# 5. v2 Combined (XGB + LSTM, 2-panel)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax = axes[0]
plot_pred_interval(ax, xgb_v2_dates, xgb_v2_y, xgb_v2_raw, C_XGB_BAND, C_XGB_Q50)
ax.set_title('(a) XGBoost v2 — Raw 90% Prediction Interval', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)

ax = axes[1]
plot_pred_interval(ax, lstm_v2_dates, lstm_v2_y, lstm_v2_preds, C_LSTM_BAND, C_LSTM_Q50)
ax.set_title('(b) LSTM v2 — Raw 90% Prediction Interval', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
finalize_time_axis(ax)

fig.suptitle('Prediction Interval Comparison — fold3 Test (2023-2025)',
             fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'pred_interval_v2_combined.png', dpi=150)
plt.close(fig)
print('  5. pred_interval_v2_combined.png')


# ══════════════════════════════════════════════════════════════
# 6. CQR effect (색상 통일 — 파랑 계열)
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

xgb_raw_width = xgb_v2_raw[0.95] - xgb_v2_raw[0.05]
xgb_cqr_width = xgb_v2_cqr[0.95] - xgb_v2_cqr[0.05]

# (a) Before CQR
ax = axes[0]
raw_cov = np.mean((xgb_v2_y >= xgb_v2_raw[0.05]) & (xgb_v2_y <= xgb_v2_raw[0.95]))
ax.fill_between(xgb_v2_dates, xgb_v2_raw[0.05], xgb_v2_raw[0.95],
                alpha=0.25, color=C_CQR_BAND,
                label=f'Raw PI (width {xgb_raw_width.mean():.1f} bp)')
ax.plot(xgb_v2_dates, xgb_v2_raw[0.5], color=C_XGB_Q50, lw=0.9, label='q50')
ax.plot(xgb_v2_dates, xgb_v2_y, color=C_ACTUAL, lw=0.5, alpha=0.5, label='Actual')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title(f'(a) XGBoost v2 Before CQR — Coverage {raw_cov:.1%}',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-30, 30); ax.grid(axis='y', alpha=0.2)

# (b) After CQR
ax = axes[1]
cqr_cov = np.mean((xgb_v2_y >= xgb_v2_cqr[0.05]) & (xgb_v2_y <= xgb_v2_cqr[0.95]))
ax.fill_between(xgb_v2_dates, xgb_v2_cqr[0.05], xgb_v2_cqr[0.95],
                alpha=0.25, color=C_XGB_BAND,
                label=f'CQR PI (width {xgb_cqr_width.mean():.1f} bp, Q̂=+{xgb_v2_qhat:.1f}bp)')
ax.plot(xgb_v2_dates, xgb_v2_cqr[0.5], color=C_XGB_Q50, lw=0.9, label='q50 (unchanged)')
ax.plot(xgb_v2_dates, xgb_v2_y, color=C_ACTUAL, lw=0.5, alpha=0.5, label='Actual')
ax.axhline(0, color='grey', ls='--', lw=0.4)
ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title(f'(b) XGBoost v2 After CQR — Coverage {cqr_cov:.1%} (target 90%)',
             fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.set_ylim(-30, 30); ax.grid(axis='y', alpha=0.2)
finalize_time_axis(ax)

fig.suptitle('CQR Conformal Calibration Effect — XGBoost v2 fold3',
             fontsize=14, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'cqr_effect_v2.png', dpi=150)
plt.close(fig)
print('  6. cqr_effect_v2.png')


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Generated charts:')
print('  1. pred_interval_xgb_v0.png  — XGBoost v0 (leaked)')
print('  2. pred_interval_lstm_v0.png — LSTM v0 (leaked)')
print('  3. pred_interval_xgb_v2.png  — XGBoost v2')
print('  4. pred_interval_lstm_v2.png — LSTM v2')
print('  5. pred_interval_v2_combined.png — v2 2-panel')
print('  6. cqr_effect_v2.png         — CQR 전후 비교')
print('=' * 60)

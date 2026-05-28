"""
29_quantile_improvement.py — 분위수 예측 비대칭성 개선

Method A: Multi-Head Quantile LSTM (v3)
  - LSTM → shared dense → 3개 별도 quantile heads (각 2-layer MLP)
  - 손실 = pinball + λ_cross * crossing_penalty + λ_sharp * width_penalty
  - 각 분위수의 독립적 non-linear head → 비대칭 구간 학습 가능

Method B: Asymmetric Variance LSTM (조건부 분산)
  - LSTM → shared dense → (mu, σ_up, σ_down)
  - q05 = mu - σ_down, q95 = mu + σ_up  (softplus로 양수 보장)
  - 구조적으로 crossing 불가능 + 본질적 비대칭

평가: 기존 v2와 동일한 walkforward 3-fold, Interval Score 비교
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'improved'
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
SEEDS = [42, 123, 2024]

LAMBDA_CROSS = 10.0
LAMBDA_SHARP = 0.005

COLORS = {
    'v2': '#6C7A89', 'v3a': '#2ECC71', 'v3b': '#E74C3C',
    'bg_dark': '#1A1A2E', 'bg_card': '#16213E',
    'text': '#EAEAEA', 'grid': '#2C3E50', 'accent': '#F1C40F',
}

FOLDS = [
    {'name': 'fold1',
     'train': ('2010-01-01', '2017-12-31'), 'val': ('2018-01-01', '2019-12-31'),
     'test': ('2020-01-01', '2020-12-31')},
    {'name': 'fold2',
     'train': ('2010-01-01', '2019-12-31'), 'val': ('2020-01-01', '2020-12-31'),
     'test': ('2021-01-01', '2022-12-31')},
    {'name': 'fold3',
     'train': ('2010-01-01', '2021-12-31'), 'val': ('2022-01-01', '2022-12-31'),
     'test': ('2023-01-01', '2025-12-31')},
]


# ─────────────────────────────────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────────────────────────────────

def interval_score(y, q05, q95, alpha=0.10):
    width = q95 - q05
    penalty_low = (2.0 / alpha) * np.maximum(q05 - y, 0)
    penalty_high = (2.0 / alpha) * np.maximum(y - q95, 0)
    return width + penalty_low + penalty_high


def asymmetry_ratio(q05, q50, q95):
    upper = q95 - q50
    lower = q50 - q05
    safe_lower = np.where(np.abs(lower) < 1e-8, 1e-8, lower)
    return upper / safe_lower


def eval_full(y, q05, q50, q95, label):
    err = y - q50
    rmse = float(np.sqrt(np.mean(err ** 2)))

    mask = (np.sign(q50) != 0) & (np.sign(y) != 0)
    da = float((np.sign(q50[mask]) == np.sign(y[mask])).mean()) if mask.sum() > 0 else float('nan')

    cov = float(np.mean((y >= q05) & (y <= q95)))
    sharp = float(np.mean(q95 - q05))

    is_val = float(np.mean(interval_score(y, q05, q95)))
    is_per = interval_score(y, q05, q95)

    ar = asymmetry_ratio(q05, q50, q95)
    ar_mean = float(np.nanmean(ar))

    up_mask = y > 0
    dn_mask = y < 0
    is_up = float(np.mean(is_per[up_mask])) if up_mask.sum() > 0 else float('nan')
    is_dn = float(np.mean(is_per[dn_mask])) if dn_mask.sum() > 0 else float('nan')

    width_std = float(np.std(q95 - q05))

    return {
        'label': label, 'RMSE_bp': round(rmse, 3), 'Dir_Acc': round(da, 4),
        'Coverage_90': round(cov, 4), 'Sharpness_bp': round(sharp, 2),
        'Interval_Score': round(is_val, 3),
        'IS_up': round(is_up, 3), 'IS_down': round(is_dn, 3),
        'Asymmetry_mean': round(ar_mean, 3), 'Width_std': round(width_std, 3),
    }


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


# ─────────────────────────────────────────────────────────────────────────
# 데이터 유틸
# ─────────────────────────────────────────────────────────────────────────

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
        if not valid[t]:
            continue
        win = X_arr[t - lookback + 1: t + 1]
        if np.isnan(win).any():
            continue
        seqs.append(win)
        tgts.append(y_arr[t])
        dates.append(date_index[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), dates


# ─────────────────────────────────────────────────────────────────────────
# 기존 LSTM v2 (baseline 비교용)
# ─────────────────────────────────────────────────────────────────────────

class QuantileLSTM_v2(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True,
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


# ─────────────────────────────────────────────────────────────────────────
# Method A: Multi-Head Quantile LSTM (v3)
# ─────────────────────────────────────────────────────────────────────────

class MultiHeadQuantileLSTM(nn.Module):
    """각 분위수에 독립적인 2-layer MLP head"""
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        head_dim = hidden // 2
        self.head_q05 = nn.Sequential(nn.Linear(hidden, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self.head_q50 = nn.Sequential(nn.Linear(hidden, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))
        self.head_q95 = nn.Sequential(nn.Linear(hidden, head_dim), nn.ReLU(), nn.Linear(head_dim, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.shared(out[:, -1, :])
        return torch.cat([self.head_q05(h), self.head_q50(h), self.head_q95(h)], dim=1)


def enhanced_pinball_loss(pred, target, qs=QUANTILES,
                          lam_cross=LAMBDA_CROSS, lam_sharp=LAMBDA_SHARP):
    pinball = pinball_loss_torch(pred, target, qs)
    crossing = (torch.relu(pred[:, 0] - pred[:, 1]).mean() +
                torch.relu(pred[:, 1] - pred[:, 2]).mean())
    width = (pred[:, 2] - pred[:, 0]).clamp(min=0).mean()
    return pinball + lam_cross * crossing + lam_sharp * width


# ─────────────────────────────────────────────────────────────────────────
# Method B: Asymmetric Variance LSTM (조건부 분산)
# ─────────────────────────────────────────────────────────────────────────

class AsymVarLSTM(nn.Module):
    """mu + σ_up/σ_down 별도 예측 → 구조적 비대칭"""
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_up_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.sigma_dn_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.shared(out[:, -1, :])
        mu = self.mu_head(h)
        sigma_up = F.softplus(self.sigma_up_head(h)) + 0.1
        sigma_dn = F.softplus(self.sigma_dn_head(h)) + 0.1
        q05 = mu - sigma_dn
        q50 = mu
        q95 = mu + sigma_up
        return torch.cat([q05, q50, q95], dim=1)


# ─────────────────────────────────────────────────────────────────────────
# 통합 학습 함수
# ─────────────────────────────────────────────────────────────────────────

def train_model(model, Xs_tr, ys_tr, Xs_val, ys_val, loss_fn, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    tr_ld = DataLoader(SeqDS(Xs_tr, ys_tr), batch_size=LSTM_CFG['batch_size'],
                       shuffle=True, drop_last=False)
    vl_ld = DataLoader(SeqDS(Xs_val, ys_val), batch_size=LSTM_CFG['batch_size'],
                       shuffle=False, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['learning_rate'])
    best, best_st, wait = float('inf'), None, 0

    for ep in range(1, LSTM_CFG['epochs'] + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = [float(loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).item())
                  for xb, yb in vl_ld]
        vl_loss = float(np.mean(vl))
        if vl_loss < best - 1e-6:
            best, wait = vl_loss, 0
            best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= LSTM_CFG['early_stopping_patience']:
            break

    if best_st is not None:
        model.load_state_dict(best_st)
    return model, best, ep


@torch.no_grad()
def predict_model(model, Xs):
    model.eval()
    pred = model(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

print('=' * 72)
print('29 — 분위수 예측 비대칭성 개선')
print(f'Device: {DEVICE}')
print(f'lambda_cross={LAMBDA_CROSS}, lambda_sharp={LAMBDA_SHARP}')
print('=' * 72)

df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

XGB_FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']
RAW_INPUT_FOR_LSTM = [
    'kr_treasury_3y', 'kr_base_rate', 'us_treasury_10y', 'us_fed_funds',
    'us_breakeven_10y', 'vix', 'kospi', 'sp500', 'dxy',
    'spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1', 'delta_dxy_t1', 'crisis_dummy',
]
RAW_INPUT_FOR_LSTM = [c for c in RAW_INPUT_FOR_LSTM if c in df.columns]
N_INPUT = len(RAW_INPUT_FOR_LSTM)
print(f'LSTM inputs: {N_INPUT}')

all_results = []
fold3_intervals = {}  # 시각화용

for fold in FOLDS:
    name = fold['name']
    print(f'\n{"=" * 60}')
    print(f'{name}: train {fold["train"]}, test {fold["test"]}')
    print(f'{"=" * 60}')

    def sl(p):
        return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[XGB_FEATURE_COLS]
    X_val_raw = sl('val')[XGB_FEATURE_COLS]
    X_te_raw = sl('test')[XGB_FEATURE_COLS]

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X):
        return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
    X_tr, X_val, X_te = s(X_tr_raw), s(X_val_raw), s(X_te_raw)

    y_tr = sl('train')['delta_y_bp']
    y_val = sl('val')['delta_y_bp']
    y_te = sl('test')['delta_y_bp']

    X_tr_lstm = X_tr[RAW_INPUT_FOR_LSTM]
    X_val_lstm = X_val[RAW_INPUT_FOR_LSTM]
    X_te_lstm = X_te[RAW_INPUT_FOR_LSTM]

    Xs_tr, ys_tr, _ = make_seq(X_tr_lstm, y_tr, LOOKBACK)
    Xs_val, ys_val, _ = make_seq(X_val_lstm, y_val, LOOKBACK)
    Xs_te, ys_te, dates_te = make_seq(X_te_lstm, y_te, LOOKBACK)
    print(f'  Xs_tr={Xs_tr.shape}, Xs_te={Xs_te.shape}')

    # ── (1) LSTM v2 baseline ──
    print('  [v2] QuantileLSTM baseline...')
    v2_seed_preds = []
    for seed in SEEDS:
        print(f'    seed={seed}', end=' ', flush=True)
        m = QuantileLSTM_v2(N_INPUT, LSTM_CFG['hidden_units'],
                            LSTM_CFG['num_layers'], LSTM_CFG['dropout'], 3)
        m, bv, ne = train_model(m, Xs_tr, ys_tr, Xs_val, ys_val,
                                pinball_loss_torch, seed)
        p = sort_qs(predict_model(m, Xs_te))
        v2_seed_preds.append(p)
        print(f'ep={ne}', end=' ', flush=True)

    v2_q05 = np.mean([p[0.05] for p in v2_seed_preds], axis=0)
    v2_q50 = np.mean([p[0.5] for p in v2_seed_preds], axis=0)
    v2_q95 = np.mean([p[0.95] for p in v2_seed_preds], axis=0)
    e_v2 = eval_full(ys_te, v2_q05, v2_q50, v2_q95, f'LSTM v2 ({name})')
    e_v2.update({'fold': name, 'method': 'v2'})
    all_results.append(e_v2)
    print(f'\n    → IS={e_v2["Interval_Score"]:.2f}  Cov={e_v2["Coverage_90"]:.3f}  '
          f'Asym={e_v2["Asymmetry_mean"]:.3f}  Sharp={e_v2["Sharpness_bp"]:.2f}')

    # ── (2) Method A: Multi-Head Quantile LSTM ──
    print('  [v3a] MultiHead Quantile LSTM...')
    v3a_seed_preds = []
    for seed in SEEDS:
        print(f'    seed={seed}', end=' ', flush=True)
        m = MultiHeadQuantileLSTM(N_INPUT, LSTM_CFG['hidden_units'],
                                  LSTM_CFG['num_layers'], LSTM_CFG['dropout'])
        m, bv, ne = train_model(m, Xs_tr, ys_tr, Xs_val, ys_val,
                                enhanced_pinball_loss, seed)
        p = sort_qs(predict_model(m, Xs_te))
        v3a_seed_preds.append(p)
        print(f'ep={ne}', end=' ', flush=True)

    v3a_q05 = np.mean([p[0.05] for p in v3a_seed_preds], axis=0)
    v3a_q50 = np.mean([p[0.5] for p in v3a_seed_preds], axis=0)
    v3a_q95 = np.mean([p[0.95] for p in v3a_seed_preds], axis=0)
    e_v3a = eval_full(ys_te, v3a_q05, v3a_q50, v3a_q95, f'MultiHead v3 ({name})')
    e_v3a.update({'fold': name, 'method': 'v3a'})
    all_results.append(e_v3a)
    print(f'\n    → IS={e_v3a["Interval_Score"]:.2f}  Cov={e_v3a["Coverage_90"]:.3f}  '
          f'Asym={e_v3a["Asymmetry_mean"]:.3f}  Sharp={e_v3a["Sharpness_bp"]:.2f}')

    # ── (3) Method B: Asymmetric Variance LSTM ──
    print('  [v3b] Asymmetric Variance LSTM...')
    v3b_seed_preds = []
    for seed in SEEDS:
        print(f'    seed={seed}', end=' ', flush=True)
        m = AsymVarLSTM(N_INPUT, LSTM_CFG['hidden_units'],
                        LSTM_CFG['num_layers'], LSTM_CFG['dropout'])
        m, bv, ne = train_model(m, Xs_tr, ys_tr, Xs_val, ys_val,
                                pinball_loss_torch, seed)
        p = predict_model(m, Xs_te)  # no sort needed - crossing impossible by design
        v3b_seed_preds.append(p)
        print(f'ep={ne}', end=' ', flush=True)

    v3b_q05 = np.mean([p[0.05] for p in v3b_seed_preds], axis=0)
    v3b_q50 = np.mean([p[0.5] for p in v3b_seed_preds], axis=0)
    v3b_q95 = np.mean([p[0.95] for p in v3b_seed_preds], axis=0)
    e_v3b = eval_full(ys_te, v3b_q05, v3b_q50, v3b_q95, f'AsymVar v3 ({name})')
    e_v3b.update({'fold': name, 'method': 'v3b'})
    all_results.append(e_v3b)
    print(f'\n    → IS={e_v3b["Interval_Score"]:.2f}  Cov={e_v3b["Coverage_90"]:.3f}  '
          f'Asym={e_v3b["Asymmetry_mean"]:.3f}  Sharp={e_v3b["Sharpness_bp"]:.2f}')

    # fold3 저장 (시각화용)
    if name == 'fold3':
        fold3_intervals = {
            'dates': dates_te, 'y': ys_te,
            'v2': {'q05': v2_q05, 'q50': v2_q50, 'q95': v2_q95},
            'v3a': {'q05': v3a_q05, 'q50': v3a_q50, 'q95': v3a_q95},
            'v3b': {'q05': v3b_q05, 'q50': v3b_q50, 'q95': v3b_q95},
        }


# ═══════════════════════════════════════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════════════════════════════════════

result_df = pd.DataFrame(all_results)
result_df.to_csv(REPORT_DIR / 'quantile_v3_comparison.csv', index=False)

print('\n' + '=' * 72)
print('결과 요약')
print('=' * 72)

cols_show = ['label', 'Interval_Score', 'Coverage_90', 'Sharpness_bp',
             'Dir_Acc', 'Asymmetry_mean', 'Width_std', 'IS_up', 'IS_down']

for fn in ['fold1', 'fold2', 'fold3']:
    subset = result_df[result_df['fold'] == fn][cols_show]
    print(f'\n--- {fn} ---')
    print(subset.to_string(index=False))

# 3-fold 평균
print('\n' + '=' * 72)
print('3-fold 평균')
print('=' * 72)
for method in ['v2', 'v3a', 'v3b']:
    sub = result_df[result_df['method'] == method]
    label = {'v2': 'LSTM v2 (baseline)', 'v3a': 'MultiHead v3 (Method A)',
             'v3b': 'AsymVar v3 (Method B)'}[method]
    print(f'\n{label}:')
    for col in ['Interval_Score', 'Coverage_90', 'Sharpness_bp', 'Dir_Acc',
                'Asymmetry_mean', 'Width_std']:
        vals = sub[col].values
        print(f'  {col:20s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 1: IS / Coverage / Sharpness / Asymmetry 비교 (fold3)
# ═══════════════════════════════════════════════════════════════════════

print('\n시각화 생성 중...')

f3 = result_df[result_df['fold'] == 'fold3']
v2_r = f3[f3['method'] == 'v2'].iloc[0]
v3a_r = f3[f3['method'] == 'v3a'].iloc[0]
v3b_r = f3[f3['method'] == 'v3b'].iloc[0]

fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor=COLORS['bg_dark'])
labels = ['LSTM v2\n(baseline)', 'MultiHead\n(Method A)', 'AsymVar\n(Method B)']
bar_colors = [COLORS['v2'], COLORS['v3a'], COLORS['v3b']]

metrics = [
    ('Interval Score (낮을수록 좋음)', 'Interval_Score', False, '{:.2f}'),
    ('Coverage 90%', 'Coverage_90', True, '{:.1%}'),
    ('Sharpness bp (좁을수록 좋음)', 'Sharpness_bp', False, '{:.2f}'),
    ('비대칭 비율 |1-AR|', None, False, '{:.3f}'),
]

for ax_idx, (title, col, is_pct, fmt) in enumerate(metrics):
    ax = axes[ax_idx]
    ax.set_facecolor(COLORS['bg_card'])

    if col is not None:
        vals = [v2_r[col], v3a_r[col], v3b_r[col]]
        if is_pct:
            plot_vals = [v * 100 for v in vals]
        else:
            plot_vals = vals
    else:
        vals = [abs(1 - v2_r['Asymmetry_mean']),
                abs(1 - v3a_r['Asymmetry_mean']),
                abs(1 - v3b_r['Asymmetry_mean'])]
        plot_vals = vals

    bars = ax.bar(labels, plot_vals, color=bar_colors,
                  edgecolor='white', linewidth=0.5, alpha=0.85)

    for b, v, orig in zip(bars, plot_vals, vals):
        display = fmt.format(orig) if is_pct else fmt.format(v)
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(plot_vals) * 0.02,
                display, ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])

    if col == 'Coverage_90':
        ax.axhline(90, color=COLORS['accent'], ls='--', lw=2, alpha=0.8, label='목표 90%')
        ax.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
                  labelcolor=COLORS['text'])

    ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['text'], pad=12)
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.spines[:].set_color(COLORS['grid'])
    ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

fig.suptitle('분위수 개선 비교 — fold3 (Test: 2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)
fig.text(0.5, -0.03,
         'Method A: 다중 헤드 + crossing penalty + sharpness 정규화  |  '
         'Method B: mu/σ_up/σ_down 분리 예측 (구조적 비대칭)',
         ha='center', fontsize=11, color=COLORS['accent'], fontstyle='italic')
plt.tight_layout()
fig.savefig(FIG_DIR / 'quantile_v3_comparison.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: quantile_v3_comparison.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 2: 비대칭 비율 분포 비교
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg_dark'])

data_sets = [
    ('LSTM v2 (baseline)', fold3_intervals['v2'], COLORS['v2']),
    ('MultiHead v3 (Method A)', fold3_intervals['v3a'], COLORS['v3a']),
    ('AsymVar v3 (Method B)', fold3_intervals['v3b'], COLORS['v3b']),
]

for ax, (title, d, color) in zip(axes, data_sets):
    ax.set_facecolor(COLORS['bg_card'])
    ar = asymmetry_ratio(d['q05'], d['q50'], d['q95'])
    ar_clip = np.clip(ar[np.isfinite(ar)], 0, 3)
    med = np.nanmedian(ar)

    ax.hist(ar_clip, bins=50, color=color, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax.axvline(1.0, color=COLORS['accent'], ls='--', lw=2, label='대칭 (=1.0)')
    ax.axvline(med, color='#E74C3C', ls='-', lw=2, label=f'중앙값 {med:.3f}')

    std_ar = float(np.nanstd(ar_clip))
    ax.set_title(f'{title}\n중앙값={med:.3f}, std={std_ar:.3f}',
                 fontsize=12, fontweight='bold', color=COLORS['text'], pad=12)
    ax.set_xlabel('비대칭 비율 (q95-q50)/(q50-q05)', color=COLORS['text'], fontsize=10)
    ax.set_ylabel('빈도', color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.spines[:].set_color(COLORS['grid'])
    ax.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
              labelcolor=COLORS['text'])

fig.suptitle('비대칭 비율 분포 비교 — fold3 (Test: 2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)
fig.text(0.5, -0.03,
         '비대칭 비율 = 1.0이면 완전 대칭 | >1: 상단 넓음 (상승 불확실) | <1: 하단 넓음 (하락 불확실)',
         ha='center', fontsize=11, color=COLORS['accent'], fontstyle='italic')
plt.tight_layout()
fig.savefig(FIG_DIR / 'quantile_v3_asymmetry.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: quantile_v3_asymmetry.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 3: fold3 예측 구간 시계열 비교
# ═══════════════════════════════════════════════════════════════════════

import matplotlib.dates as mdates

dates_plot = pd.to_datetime(fold3_intervals['dates'])
y_plot = fold3_intervals['y']

fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor=COLORS['bg_dark'],
                         sharex=True)

for ax, (method_key, method_label, color) in zip(axes, [
    ('v2', 'LSTM v2 (baseline)', COLORS['v2']),
    ('v3a', 'MultiHead v3 (Method A)', COLORS['v3a']),
    ('v3b', 'AsymVar v3 (Method B)', COLORS['v3b']),
]):
    ax.set_facecolor(COLORS['bg_card'])
    d = fold3_intervals[method_key]

    ax.fill_between(dates_plot, d['q05'], d['q95'], alpha=0.3, color=color,
                    label='90% PI')
    ax.plot(dates_plot, d['q50'], color=color, lw=1.2, label='q50 예측')
    ax.plot(dates_plot, y_plot, color='white', lw=0.6, alpha=0.7, label='실제값')
    ax.axhline(0, color='grey', ls='--', lw=0.5)

    # 비대칭 구간 강조: 상단/하단 폭이 다른 구간
    upper_w = d['q95'] - d['q50']
    lower_w = d['q50'] - d['q05']
    ar_ts = upper_w / np.maximum(lower_w, 0.01)

    r = f3[f3['method'] == method_key].iloc[0]
    info = (f'IS={r["Interval_Score"]:.2f}  Cov={r["Coverage_90"]:.1%}  '
            f'Sharp={r["Sharpness_bp"]:.1f}bp  Asym={r["Asymmetry_mean"]:.3f}')

    ax.set_title(f'{method_label}  |  {info}',
                 fontsize=12, fontweight='bold', color=COLORS['text'], pad=8)
    ax.set_ylabel('Δ Bond Yield (bp)', color=COLORS['text'])
    ax.tick_params(colors=COLORS['text'])
    ax.spines[:].set_color(COLORS['grid'])
    ax.grid(axis='y', alpha=0.2, color=COLORS['grid'])
    ax.legend(loc='upper right', fontsize=9, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

fig.suptitle('예측 구간 시계열 비교 — fold3 (2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / 'quantile_v3_intervals.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: quantile_v3_intervals.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 4: 방향별 IS + Width 적응성
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg_dark'])

# (1) 방향별 IS
ax = axes[0]
ax.set_facecolor(COLORS['bg_card'])
x_pos = np.arange(3)
width = 0.35
is_up = [v2_r['IS_up'], v3a_r['IS_up'], v3b_r['IS_up']]
is_dn = [v2_r['IS_down'], v3a_r['IS_down'], v3b_r['IS_down']]
bars1 = ax.bar(x_pos - width/2, is_up, width, color='#E74C3C', alpha=0.85,
               label='상승일 IS', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, is_dn, width, color='#3498DB', alpha=0.85,
               label='하락일 IS', edgecolor='white', linewidth=0.5)
for b, v in zip(bars1, is_up):
    ax.text(b.get_x() + b.get_width()/2, v + 0.2, f'{v:.1f}',
            ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
for b, v in zip(bars2, is_dn):
    ax.text(b.get_x() + b.get_width()/2, v + 0.2, f'{v:.1f}',
            ha='center', fontsize=10, fontweight='bold', color='#3498DB')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10, color=COLORS['text'])
ax.set_title('방향별 Interval Score', fontsize=13, fontweight='bold',
             color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
ax.legend(fontsize=10, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'])

# (2) IS 상승-하락 차이 (|IS_up - IS_down|)
ax = axes[1]
ax.set_facecolor(COLORS['bg_card'])
diff_is = [abs(u - d) for u, d in zip(is_up, is_dn)]
bars = ax.bar(labels, diff_is, color=bar_colors, edgecolor='white',
              linewidth=0.5, alpha=0.85)
for b, v in zip(bars, diff_is):
    ax.text(b.get_x() + b.get_width()/2, v + 0.05, f'{v:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('IS 방향 차이 |IS_up - IS_down|\n(클수록 방향별 구분력)', fontsize=12,
             fontweight='bold', color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

# (3) Width 변동성 (적응성)
ax = axes[2]
ax.set_facecolor(COLORS['bg_card'])
w_std = [v2_r['Width_std'], v3a_r['Width_std'], v3b_r['Width_std']]
bars = ax.bar(labels, w_std, color=bar_colors, edgecolor='white',
              linewidth=0.5, alpha=0.85)
for b, v in zip(bars, w_std):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('구간 폭 변동성 (Width std)\n(높을수록 상황 적응적)', fontsize=12,
             fontweight='bold', color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

fig.suptitle('방향별 IS + 구간 적응성 — fold3 (Test: 2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / 'quantile_v3_directional.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: quantile_v3_directional.png')

print('\n' + '=' * 72)
print('완료!')
print('=' * 72)

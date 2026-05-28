"""
30_viz_quantile_improvement.py — 분위수 개선 시각화 (발표용)

LSTM v2 (baseline) vs AsymVar v3 (Method B) 비교
  - 밝은 배경, 시계열 fill_between 스타일
  - AsymVar: 상/하방 구간 다른 색으로 비대칭 시각화
  - fold3 (2023~2025)

산출물:
  reports/figures/improved/quantile_v3_effect.png        (2-panel 시계열)
  reports/figures/improved/quantile_v3_asymmetry_ts.png  (비대칭 비율 시계열)
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'improved'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
SEEDS = [42, 123, 2024]

FOLD3 = {
    'train': ('2010-01-01', '2021-12-31'),
    'val': ('2022-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-12-31'),
}


# ─────────────────────────────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────────────────────────────

def interval_score(y, q05, q95, alpha=0.10):
    width = q95 - q05
    return width + (2/alpha)*np.maximum(q05-y, 0) + (2/alpha)*np.maximum(y-q95, 0)

def asymmetry_ratio(q05, q50, q95):
    upper = q95 - q50
    lower = q50 - q05
    return upper / np.where(np.abs(lower) < 1e-8, 1e-8, lower)


# ─────────────────────────────────────────────────────────────────────
# 데이터 유틸
# ─────────────────────────────────────────────────────────────────────

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


def pinball_loss_torch(pred, target, qs=QUANTILES):
    target = target.unsqueeze(1)
    q = torch.tensor(qs, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    diff = target - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


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
    return model, ep


@torch.no_grad()
def predict_model(model, Xs):
    model.eval()
    pred = model(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


# ─────────────────────────────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────────────────────────────

class QuantileLSTM_v2(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_q)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class AsymVarLSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.shared = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_up_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
        self.sigma_dn_head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.shared(out[:, -1, :])
        mu = self.mu_head(h)
        sigma_up = torchF.softplus(self.sigma_up_head(h)) + 0.1
        sigma_dn = torchF.softplus(self.sigma_dn_head(h)) + 0.1
        q05 = mu - sigma_dn
        q50 = mu
        q95 = mu + sigma_up
        return torch.cat([q05, q50, q95], dim=1)


# ═══════════════════════════════════════════════════════════════════════
# MAIN — fold3만 학습
# ═══════════════════════════════════════════════════════════════════════

print('=' * 60)
print('30 — 분위수 개선 시각화 (fold3)')
print(f'Device: {DEVICE}')
print('=' * 60)

df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

XGB_FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']
RAW_INPUT_FOR_LSTM = [
    'kr_treasury_3y', 'kr_base_rate', 'us_treasury_10y', 'us_fed_funds',
    'us_breakeven_10y', 'vix', 'kospi', 'sp500', 'dxy',
    'spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1', 'delta_dxy_t1', 'crisis_dummy',
]
RAW_INPUT_FOR_LSTM = [c for c in RAW_INPUT_FOR_LSTM if c in df.columns]

X_tr_raw = df.loc[FOLD3['train'][0]:FOLD3['train'][1]][XGB_FEATURE_COLS]
X_val_raw = df.loc[FOLD3['val'][0]:FOLD3['val'][1]][XGB_FEATURE_COLS]
X_te_raw = df.loc[FOLD3['test'][0]:FOLD3['test'][1]][XGB_FEATURE_COLS]

scaler = RobustScaler().fit(X_tr_raw)
def s(X):
    return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
X_tr, X_val, X_te = s(X_tr_raw), s(X_val_raw), s(X_te_raw)

y_tr = df.loc[FOLD3['train'][0]:FOLD3['train'][1]]['delta_y_bp']
y_val = df.loc[FOLD3['val'][0]:FOLD3['val'][1]]['delta_y_bp']
y_te = df.loc[FOLD3['test'][0]:FOLD3['test'][1]]['delta_y_bp']

Xs_tr, ys_tr, _ = make_seq(X_tr[RAW_INPUT_FOR_LSTM], y_tr, LOOKBACK)
Xs_val, ys_val, _ = make_seq(X_val[RAW_INPUT_FOR_LSTM], y_val, LOOKBACK)
Xs_te, ys_te, dates_te = make_seq(X_te[RAW_INPUT_FOR_LSTM], y_te, LOOKBACK)
dates_plot = pd.to_datetime(dates_te)

print(f'Xs_tr={Xs_tr.shape}, Xs_te={Xs_te.shape}')
N_INPUT = Xs_tr.shape[2]

# ── LSTM v2 (3 seeds) ──
print('LSTM v2 학습...')
v2_preds = []
for seed in SEEDS:
    print(f'  seed={seed}', end=' ', flush=True)
    m = QuantileLSTM_v2(N_INPUT, LSTM_CFG['hidden_units'],
                        LSTM_CFG['num_layers'], LSTM_CFG['dropout'], 3)
    m, ne = train_model(m, Xs_tr, ys_tr, Xs_val, ys_val, pinball_loss_torch, seed)
    p = sort_qs(predict_model(m, Xs_te))
    v2_preds.append(p)
    print(f'ep={ne}')
v2_q05 = np.mean([p[0.05] for p in v2_preds], axis=0)
v2_q50 = np.mean([p[0.5] for p in v2_preds], axis=0)
v2_q95 = np.mean([p[0.95] for p in v2_preds], axis=0)

# ── AsymVar v3 (3 seeds) ──
print('AsymVar v3 학습...')
v3b_preds = []
for seed in SEEDS:
    print(f'  seed={seed}', end=' ', flush=True)
    m = AsymVarLSTM(N_INPUT, LSTM_CFG['hidden_units'],
                    LSTM_CFG['num_layers'], LSTM_CFG['dropout'])
    m, ne = train_model(m, Xs_tr, ys_tr, Xs_val, ys_val, pinball_loss_torch, seed)
    p = predict_model(m, Xs_te)
    v3b_preds.append(p)
    print(f'ep={ne}')
v3b_q05 = np.mean([p[0.05] for p in v3b_preds], axis=0)
v3b_q50 = np.mean([p[0.5] for p in v3b_preds], axis=0)
v3b_q95 = np.mean([p[0.95] for p in v3b_preds], axis=0)

# ── 지표 계산 ──
def calc_metrics(y, q05, q50, q95):
    cov = float(np.mean((y >= q05) & (y <= q95)))
    sharp = float(np.mean(q95 - q05))
    is_mean = float(np.mean(interval_score(y, q05, q95)))
    ar = asymmetry_ratio(q05, q50, q95)
    ar_med = float(np.nanmedian(ar))
    ar_std = float(np.nanstd(ar))
    mask = (np.sign(q50) != 0) & (np.sign(y) != 0)
    da = float((np.sign(q50[mask]) == np.sign(y[mask])).mean())
    return cov, sharp, is_mean, ar_med, ar_std, da

v2_cov, v2_sharp, v2_is, v2_ar_med, v2_ar_std, v2_da = calc_metrics(ys_te, v2_q05, v2_q50, v2_q95)
v3_cov, v3_sharp, v3_is, v3_ar_med, v3_ar_std, v3_da = calc_metrics(ys_te, v3b_q05, v3b_q50, v3b_q95)

print(f'\nLSTM v2:  IS={v2_is:.2f}  Cov={v2_cov:.1%}  Sharp={v2_sharp:.1f}bp  '
      f'AR_med={v2_ar_med:.3f}  DA={v2_da:.1%}')
print(f'AsymVar:  IS={v3_is:.2f}  Cov={v3_cov:.1%}  Sharp={v3_sharp:.1f}bp  '
      f'AR_med={v3_ar_med:.3f}  DA={v3_da:.1%}')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 1: 2-panel 시계열 (page6_cqr_effect.png 스타일)
# ═══════════════════════════════════════════════════════════════════════

print('\n시각화 생성 중...')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig.suptitle('분위수 예측 개선 효과 — LSTM v2 vs AsymVar v3 (fold3)',
             fontsize=15, fontweight='bold', y=0.98)

# ── Panel (a): LSTM v2 Baseline ──
ax1.set_title(f'(a) LSTM v2 Baseline — Coverage {v2_cov:.1%}, '
              f'IS={v2_is:.1f}, 비대칭 비율 {v2_ar_med:.3f}',
              fontsize=13, pad=10)

ax1.fill_between(dates_plot, v2_q05, v2_q95, alpha=0.3, color='#4CAF50',
                 label=f'90% PI (width {v2_sharp:.1f} bp, 대칭적)')
ax1.plot(dates_plot, v2_q50, color='#2E7D32', lw=1.2, label='q50')
ax1.plot(dates_plot, ys_te, color='#E53935', lw=0.6, alpha=0.7, label='Actual')
ax1.axhline(0, color='grey', ls='--', lw=0.5)
ax1.set_ylabel('Δy (bp)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(-30, 30)

# ── Panel (b): AsymVar v3 — 상/하방 구간 분리 ──
ax2.set_title(f'(b) AsymVar v3 (조건부 분산) — Coverage {v3_cov:.1%}, '
              f'IS={v3_is:.1f}, 비대칭 비율 {v3_ar_med:.3f}',
              fontsize=13, pad=10)

# 상방 구간 (q50 → q95): 빨간계열
ax2.fill_between(dates_plot, v3b_q50, v3b_q95, alpha=0.3, color='#E53935',
                 label=f'상방 구간 (q50→q95)')
# 하방 구간 (q05 → q50): 파란계열
ax2.fill_between(dates_plot, v3b_q05, v3b_q50, alpha=0.3, color='#1565C0',
                 label=f'하방 구간 (q05→q50)')
ax2.plot(dates_plot, v3b_q50, color='#4A148C', lw=1.2, label='q50')
ax2.plot(dates_plot, ys_te, color='#E53935', lw=0.6, alpha=0.5, label='Actual')
ax2.axhline(0, color='grey', ls='--', lw=0.5)
ax2.set_ylabel('Δy (bp)', fontsize=12)
ax2.set_ylim(-30, 30)

# x축 설정
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax2.grid(axis='y', alpha=0.3)

# 개선 요약 텍스트
delta_is = v2_is - v3_is
delta_cov = (v3_cov - v2_cov) * 100
fig.text(0.5, 0.01,
         f'AsymVar v3 개선: IS {delta_is:+.1f} (낮을수록 좋음)  |  '
         f'Coverage {delta_cov:+.1f}%p  |  '
         f'상/하방 구간 폭이 시장 상황에 따라 독립적으로 변동',
         ha='center', fontsize=11, fontstyle='italic', color='#1565C0',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(FIG_DIR / 'quantile_v3_effect.png')
plt.close(fig)
print(f'  저장: quantile_v3_effect.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 2: 비대칭 비율 시계열 + 구간 폭 시계열
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle('예측 구간 비대칭성 시계열 비교 — fold3 (2023~2025)',
             fontsize=15, fontweight='bold', y=0.98)

# ── (a) 비대칭 비율 비교 ──
ax = axes[0]
ar_v2 = asymmetry_ratio(v2_q05, v2_q50, v2_q95)
ar_v3 = asymmetry_ratio(v3b_q05, v3b_q50, v3b_q95)
ar_v2_clip = np.clip(ar_v2, 0.5, 2.0)
ar_v3_clip = np.clip(ar_v3, 0.5, 2.0)

# 20일 이동 평균으로 트렌드 표시
window = 20
ar_v2_ma = pd.Series(ar_v2_clip).rolling(window, min_periods=1).mean().values
ar_v3_ma = pd.Series(ar_v3_clip).rolling(window, min_periods=1).mean().values

ax.scatter(dates_plot, ar_v2_clip, s=3, alpha=0.15, color='#4CAF50')
ax.scatter(dates_plot, ar_v3_clip, s=3, alpha=0.15, color='#E53935')
ax.plot(dates_plot, ar_v2_ma, color='#2E7D32', lw=2, label=f'LSTM v2 (20일 MA, 중앙값={v2_ar_med:.3f})')
ax.plot(dates_plot, ar_v3_ma, color='#C62828', lw=2, label=f'AsymVar v3 (20일 MA, 중앙값={v3_ar_med:.3f})')
ax.axhline(1.0, color='grey', ls='--', lw=1.5, alpha=0.8, label='완전 대칭 (=1.0)')
ax.fill_between(dates_plot, 0.9, 1.1, alpha=0.08, color='grey')
ax.set_ylabel('비대칭 비율\n(q95-q50)/(q50-q05)', fontsize=11)
ax.set_title('(a) 비대칭 비율 시계열 — AsymVar가 더 넓은 변동 범위', fontsize=13, pad=8)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.55, 1.55)

# ── (b) 상방/하방 구간 폭 (AsymVar만) ──
ax = axes[1]
upper_w = v3b_q95 - v3b_q50
lower_w = v3b_q50 - v3b_q05

upper_ma = pd.Series(upper_w).rolling(window, min_periods=1).mean().values
lower_ma = pd.Series(lower_w).rolling(window, min_periods=1).mean().values

ax.fill_between(dates_plot, 0, upper_w, alpha=0.15, color='#E53935')
ax.fill_between(dates_plot, 0, -lower_w, alpha=0.15, color='#1565C0')
ax.plot(dates_plot, upper_ma, color='#C62828', lw=2, label='상방 폭 σ_up (20일 MA)')
ax.plot(dates_plot, -lower_ma, color='#0D47A1', lw=2, label='하방 폭 -σ_down (20일 MA)')
ax.axhline(0, color='grey', ls='-', lw=1)
ax.set_ylabel('구간 폭 (bp)', fontsize=11)
ax.set_title('(b) AsymVar v3 상방/하방 구간 폭 — σ_up과 σ_down이 독립 변동',
             fontsize=13, pad=8)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

# ── (c) 실제값 + 비대칭 구간이 잘 맞는 구간 하이라이트 ──
ax = axes[2]

# v3b 예측 구간
ax.fill_between(dates_plot, v3b_q50, v3b_q95, alpha=0.25, color='#E53935',
                label='상방 구간')
ax.fill_between(dates_plot, v3b_q05, v3b_q50, alpha=0.25, color='#1565C0',
                label='하방 구간')
ax.plot(dates_plot, v3b_q50, color='#4A148C', lw=1, label='q50')
ax.plot(dates_plot, ys_te, color='#212121', lw=0.5, alpha=0.6, label='Actual')
ax.axhline(0, color='grey', ls='--', lw=0.5)

# 비대칭이 큰 구간 (AR > 1.2 or AR < 0.8) 하이라이트
asym_big = (ar_v3 > 1.2) | (ar_v3 < 0.8)
for i in range(len(dates_plot)):
    if asym_big[i]:
        ax.axvspan(dates_plot[i] - pd.Timedelta(hours=12),
                   dates_plot[i] + pd.Timedelta(hours=12),
                   alpha=0.08, color='#FF9800', zorder=0)

ax.set_ylabel('Δy (bp)', fontsize=11)
ax.set_title('(c) AsymVar v3 예측 구간 + 강한 비대칭 구간 (주황 하이라이트)',
             fontsize=13, pad=8)
ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-30, 30)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIG_DIR / 'quantile_v3_asymmetry_ts.png')
plt.close(fig)
print(f'  저장: quantile_v3_asymmetry_ts.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 3: 구조 비교 다이어그램 (아키텍처 + 결과 요약)
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('LSTM v2 vs AsymVar v3 — 구조적 차이와 결과', fontsize=15, fontweight='bold')

# (1) 아키텍처 비교 다이어그램
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('모델 구조 비교', fontsize=13, fontweight='bold', pad=15)

# v2 구조
ax.text(2.5, 9.5, 'LSTM v2 (기존)', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
boxes_v2 = [
    (2.5, 8.2, 'LSTM (2-layer, h=64)'),
    (2.5, 6.8, 'Linear(64 → 3)'),
    (2.5, 5.2, 'q05  q50  q95'),
]
for x, y, txt in boxes_v2:
    ax.text(x, y, txt, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#4CAF50'))
ax.annotate('', xy=(2.5, 7.3), xytext=(2.5, 7.8),
            arrowprops=dict(arrowstyle='->', color='#666'))
ax.annotate('', xy=(2.5, 5.7), xytext=(2.5, 6.3),
            arrowprops=dict(arrowstyle='->', color='#666'))
ax.text(2.5, 4.3, '문제: 3개 출력이\n같은 weight 공유\n→ 대칭적 구간',
        ha='center', fontsize=9, color='#C62828', fontstyle='italic')

# v3b 구조
ax.text(7.5, 9.5, 'AsymVar v3 (개선)', ha='center', fontsize=12, fontweight='bold', color='#C62828')
boxes_v3 = [
    (7.5, 8.2, 'LSTM (2-layer, h=64)'),
    (7.5, 6.8, 'Shared Dense(64→64)'),
    (5.8, 5.2, 'mu\nhead'),
    (7.5, 5.2, 'σ_up\nhead'),
    (9.2, 5.2, 'σ_down\nhead'),
]
for x, y, txt in boxes_v3:
    ax.text(x, y, txt, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE', edgecolor='#E53935'))
ax.annotate('', xy=(7.5, 7.3), xytext=(7.5, 7.8),
            arrowprops=dict(arrowstyle='->', color='#666'))
for xp in [5.8, 7.5, 9.2]:
    ax.annotate('', xy=(xp, 5.8), xytext=(7.5, 6.3),
                arrowprops=dict(arrowstyle='->', color='#666'))
ax.text(7.5, 3.8, 'q50 = mu\nq95 = mu + softplus(σ_up)\nq05 = mu - softplus(σ_down)',
        ha='center', fontsize=9, color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', alpha=0.8))
ax.text(7.5, 2.5, '장점: crossing 불가능\nσ_up ≠ σ_down → 비대칭',
        ha='center', fontsize=9, color='#2E7D32', fontstyle='italic')

# (2) 핵심 지표 비교 막대
ax = axes[1]
ax.set_title('Interval Score 비교\n(낮을수록 좋음)', fontsize=13, fontweight='bold', pad=15)
bars = ax.bar(['LSTM v2', 'AsymVar v3'], [v2_is, v3_is],
              color=['#4CAF50', '#E53935'], edgecolor='white', alpha=0.85)
for b, v in zip(bars, [v2_is, v3_is]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.2, f'{v:.2f}',
            ha='center', fontsize=13, fontweight='bold')
ax.set_ylabel('Interval Score', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# 개선율 화살표
mid_x = 0.5
ax.annotate(f'{delta_is:+.1f}\n({delta_is/v2_is*100:+.1f}%)',
            xy=(1, v3_is), xytext=(0.5, (v2_is+v3_is)/2),
            fontsize=12, fontweight='bold', color='#1565C0', ha='center',
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

# (3) 지표 요약 테이블
ax = axes[2]
ax.axis('off')
ax.set_title('fold3 상세 비교', fontsize=13, fontweight='bold', pad=15)

table_data = [
    ['Interval Score', f'{v2_is:.2f}', f'{v3_is:.2f}', f'{delta_is:+.2f}'],
    ['Coverage 90%', f'{v2_cov:.1%}', f'{v3_cov:.1%}', f'{delta_cov:+.1f}%p'],
    ['Sharpness (bp)', f'{v2_sharp:.1f}', f'{v3_sharp:.1f}', f'{v3_sharp-v2_sharp:+.1f}'],
    ['Dir Accuracy', f'{v2_da:.1%}', f'{v3_da:.1%}', f'{(v3_da-v2_da)*100:+.1f}%p'],
    ['비대칭 비율 중앙값', f'{v2_ar_med:.3f}', f'{v3_ar_med:.3f}', '—'],
    ['비대칭 비율 std', f'{v2_ar_std:.3f}', f'{v3_ar_std:.3f}',
     f'{v3_ar_std/v2_ar_std:.1f}x' if v2_ar_std > 0 else '—'],
]

tbl = ax.table(cellText=table_data,
               colLabels=['지표', 'LSTM v2', 'AsymVar v3', '변화'],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.0, 2.0)

for j in range(4):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for i in range(1, len(table_data) + 1):
    for j in range(4):
        if i % 2 == 0:
            tbl[i, j].set_facecolor('#E3F2FD')
        else:
            tbl[i, j].set_facecolor('#FFFFFF')
    # 변화 열 색상
    cell_text = table_data[i-1][3]
    if cell_text.startswith('+') or cell_text.endswith('x'):
        tbl[i, 3].set_text_props(color='#C62828', fontweight='bold')
    elif cell_text.startswith('-'):
        tbl[i, 3].set_text_props(color='#2E7D32', fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'quantile_v3_architecture.png')
plt.close(fig)
print(f'  저장: quantile_v3_architecture.png')


print('\n' + '=' * 60)
print('완료!')
print('=' * 60)

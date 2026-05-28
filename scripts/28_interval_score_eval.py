"""
28_interval_score_eval.py — Interval Score 평가 + 비대칭성 분석

목적:
  기존 Pinball Loss 외에 Interval Score(IS)를 추가하여
  "좁고 정확한 예측 구간"을 정량화. XGBoost v2 & LSTM v2 비교.

Interval Score 공식:
  IS = (u - l) + (2/α) * max(l - y, 0) + (2/α) * max(y - u, 0)
  α = 0.10, l = q05, u = q95

산출물:
  - reports/no_leak_v2/interval_score_comparison.csv
  - reports/no_leak_v2/predictions_xgb_v2_all_folds.csv
  - reports/no_leak_v2/predictions_lstm_v2_fold3.csv
  - reports/figures/improved/interval_score_comparison.png
  - reports/figures/improved/interval_asymmetry_analysis.png

실행:
  python scripts/28_interval_score_eval.py
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

COLORS = {
    'xgboost': '#2ECC71', 'lstm': '#3498DB', 'naive': '#6C7A89',
    'arima': '#F39C12', 'accent': '#E74C3C',
    'bg_dark': '#1A1A2E', 'bg_card': '#16213E',
    'text': '#EAEAEA', 'grid': '#2C3E50',
}

FOLDS = [
    {'name': 'fold1',
     'train': ('2010-01-01', '2017-12-31'), 'val': ('2018-01-01', '2019-12-31'),
     'cal': ('2019-07-01', '2019-12-31'), 'test': ('2020-01-01', '2020-12-31')},
    {'name': 'fold2',
     'train': ('2010-01-01', '2019-12-31'), 'val': ('2020-01-01', '2020-12-31'),
     'cal': ('2020-07-01', '2020-12-31'), 'test': ('2021-01-01', '2022-12-31')},
    {'name': 'fold3',
     'train': ('2010-01-01', '2021-12-31'), 'val': ('2022-01-01', '2022-12-31'),
     'cal': ('2022-07-01', '2022-12-31'), 'test': ('2023-01-01', '2025-12-31')},
]

XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}

# ─────────────────────────────────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────────────────────────────────

def interval_score(y, q05, q95, alpha=0.10):
    """개별 샘플 Interval Score 계산"""
    width = q95 - q05
    penalty_low = (2.0 / alpha) * np.maximum(q05 - y, 0)
    penalty_high = (2.0 / alpha) * np.maximum(y - q95, 0)
    return width + penalty_low + penalty_high


def interval_score_mean(y, q05, q95, alpha=0.10):
    """평균 Interval Score"""
    return float(np.mean(interval_score(y, q05, q95, alpha)))


def asymmetry_ratio(q05, q50, q95):
    """비대칭 비율: (q95-q50)/(q50-q05). 1이면 대칭, >1이면 상단 넓음"""
    upper = q95 - q50
    lower = q50 - q05
    safe_lower = np.where(np.abs(lower) < 1e-8, 1e-8, lower)
    return upper / safe_lower


def eval_full(y, q05, q50, q95, label):
    """기존 지표 + Interval Score + 비대칭성"""
    err = y - q50
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))

    mask = (np.sign(q50) != 0) & (np.sign(y) != 0)
    da = float((np.sign(q50[mask]) == np.sign(y[mask])).mean()) if mask.sum() > 0 else float('nan')

    cov = float(np.mean((y >= q05) & (y <= q95)))
    sharp = float(np.mean(q95 - q05))

    is_val = interval_score_mean(y, q05, q95)
    is_per_sample = interval_score(y, q05, q95)

    ar = asymmetry_ratio(q05, q50, q95)
    ar_mean = float(np.nanmean(ar))
    ar_std = float(np.nanstd(ar))

    # 방향별 IS
    up_mask = y > 0
    dn_mask = y < 0
    is_up = float(np.mean(is_per_sample[up_mask])) if up_mask.sum() > 0 else float('nan')
    is_dn = float(np.mean(is_per_sample[dn_mask])) if dn_mask.sum() > 0 else float('nan')

    # 구간 폭 변동성 (높을수록 적응적)
    width_std = float(np.std(q95 - q05))

    return {
        'label': label,
        'RMSE_bp': round(rmse, 3),
        'MAE_bp': round(mae, 3),
        'Dir_Acc': round(da, 4),
        'Coverage_90': round(cov, 4),
        'Sharpness_bp': round(sharp, 2),
        'Interval_Score': round(is_val, 3),
        'IS_up': round(is_up, 3),
        'IS_down': round(is_dn, 3),
        'Asymmetry_mean': round(ar_mean, 3),
        'Asymmetry_std': round(ar_std, 3),
        'Width_std': round(width_std, 3),
    }


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


# ─────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────────────────────────────────

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
# MAIN
# ═══════════════════════════════════════════════════════════════════════

print('=' * 72)
print('28 — Interval Score 평가')
print(f'Device: {DEVICE}')
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
print(f'XGB features: {len(XGB_FEATURE_COLS)}, LSTM inputs: {len(RAW_INPUT_FOR_LSTM)}')

all_results = []
all_xgb_preds = []
all_lstm_preds = []

for fold in FOLDS:
    name = fold['name']
    print(f'\n{"=" * 60}')
    print(f'{name}: train {fold["train"]}, test {fold["test"]}')
    print(f'{"=" * 60}')

    def sl(p):
        return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[XGB_FEATURE_COLS]
    X_cal_raw = sl('cal')[XGB_FEATURE_COLS]
    X_val_raw = sl('val')[XGB_FEATURE_COLS]
    X_te_raw = sl('test')[XGB_FEATURE_COLS]

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X):
        return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
    X_tr, X_cal, X_val, X_te = s(X_tr_raw), s(X_cal_raw), s(X_val_raw), s(X_te_raw)

    y_tr = sl('train')['delta_y_bp']
    y_cal = sl('cal')['delta_y_bp']
    y_val = sl('val')['delta_y_bp']
    y_te = sl('test')['delta_y_bp']

    # ── XGBoost v2 ──
    print('  XGBoost v2 학습...')
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

    xgb_q05_cqr = xgb_preds['test'][0.05] - Q_hat
    xgb_q50 = xgb_preds['test'][0.5]
    xgb_q95_cqr = xgb_preds['test'][0.95] + Q_hat

    xgb_eval_raw = eval_full(y_te.values,
                             xgb_preds['test'][0.05], xgb_q50, xgb_preds['test'][0.95],
                             f'XGBoost raw ({name})')
    xgb_eval_raw['fold'] = name
    xgb_eval_raw['model'] = 'XGBoost v2'
    xgb_eval_raw['stage'] = 'raw'

    xgb_eval_cqr = eval_full(y_te.values, xgb_q05_cqr, xgb_q50, xgb_q95_cqr,
                             f'XGBoost CQR ({name})')
    xgb_eval_cqr['fold'] = name
    xgb_eval_cqr['model'] = 'XGBoost v2'
    xgb_eval_cqr['stage'] = 'CQR'

    all_results.extend([xgb_eval_raw, xgb_eval_cqr])

    print(f'  XGBoost raw: IS={xgb_eval_raw["Interval_Score"]:.2f}  '
          f'Cov={xgb_eval_raw["Coverage_90"]:.3f}  '
          f'Sharp={xgb_eval_raw["Sharpness_bp"]:.2f}  '
          f'Asym={xgb_eval_raw["Asymmetry_mean"]:.3f}')
    print(f'  XGBoost CQR: IS={xgb_eval_cqr["Interval_Score"]:.2f}  '
          f'Cov={xgb_eval_cqr["Coverage_90"]:.3f}  '
          f'Sharp={xgb_eval_cqr["Sharpness_bp"]:.2f}')

    # 개별 예측값 저장
    for i, dt in enumerate(X_te.index):
        all_xgb_preds.append({
            'date': dt, 'fold': name, 'y_true': y_te.values[i],
            'q05_raw': xgb_preds['test'][0.05][i],
            'q50': xgb_q50[i],
            'q95_raw': xgb_preds['test'][0.95][i],
            'q05_cqr': xgb_q05_cqr[i],
            'q95_cqr': xgb_q95_cqr[i],
            'IS_raw': interval_score(y_te.values[i:i+1],
                                     xgb_preds['test'][0.05][i:i+1],
                                     xgb_preds['test'][0.95][i:i+1])[0],
            'IS_cqr': interval_score(y_te.values[i:i+1],
                                     xgb_q05_cqr[i:i+1],
                                     xgb_q95_cqr[i:i+1])[0],
        })

    # ── LSTM v2 ──
    print(f'  LSTM v2 학습 (3 seeds)...')
    X_tr_lstm = X_tr[RAW_INPUT_FOR_LSTM]
    X_val_lstm = X_val[RAW_INPUT_FOR_LSTM]
    X_te_lstm = X_te[RAW_INPUT_FOR_LSTM]

    Xs_tr, ys_tr, _ = make_seq(X_tr_lstm, y_tr, LOOKBACK)
    Xs_val, ys_val, _ = make_seq(X_val_lstm, y_val, LOOKBACK)
    Xs_te, ys_te, dates_te = make_seq(X_te_lstm, y_te, LOOKBACK)

    seed_preds = []
    for seed in SEEDS:
        print(f'    seed={seed}...', end=' ', flush=True)
        m, bv, ne = train_lstm_one(seed, Xs_tr, ys_tr, Xs_val, ys_val)
        p_te = sort_qs(predict_lstm(m, Xs_te))
        seed_preds.append(p_te)
        e = eval_full(ys_te, p_te[0.05], p_te[0.5], p_te[0.95],
                       f'LSTM s={seed} ({name})')
        e['fold'] = name
        e['model'] = f'LSTM v2 (s={seed})'
        e['stage'] = 'raw'
        all_results.append(e)
        print(f'ep={ne} IS={e["Interval_Score"]:.2f} '
              f'Cov={e["Coverage_90"]:.3f} Asym={e["Asymmetry_mean"]:.3f}')

    # 시드 평균
    avg_q05 = np.mean([p[0.05] for p in seed_preds], axis=0)
    avg_q50 = np.mean([p[0.5] for p in seed_preds], axis=0)
    avg_q95 = np.mean([p[0.95] for p in seed_preds], axis=0)

    lstm_eval_avg = eval_full(ys_te, avg_q05, avg_q50, avg_q95,
                              f'LSTM avg ({name})')
    lstm_eval_avg['fold'] = name
    lstm_eval_avg['model'] = 'LSTM v2 (avg)'
    lstm_eval_avg['stage'] = 'raw'
    all_results.append(lstm_eval_avg)

    print(f'  LSTM avg:    IS={lstm_eval_avg["Interval_Score"]:.2f}  '
          f'Cov={lstm_eval_avg["Coverage_90"]:.3f}  '
          f'Sharp={lstm_eval_avg["Sharpness_bp"]:.2f}  '
          f'Asym={lstm_eval_avg["Asymmetry_mean"]:.3f}')

    # LSTM 예측값 저장
    for i, dt in enumerate(dates_te):
        all_lstm_preds.append({
            'date': dt, 'fold': name, 'y_true': ys_te[i],
            'q05_avg': avg_q05[i], 'q50_avg': avg_q50[i], 'q95_avg': avg_q95[i],
            'IS': interval_score(ys_te[i:i+1], avg_q05[i:i+1], avg_q95[i:i+1])[0],
        })


# ═══════════════════════════════════════════════════════════════════════
# 결과 저장
# ═══════════════════════════════════════════════════════════════════════

result_df = pd.DataFrame(all_results)
result_df.to_csv(REPORT_DIR / 'interval_score_comparison.csv', index=False)

xgb_pred_df = pd.DataFrame(all_xgb_preds)
xgb_pred_df.to_csv(REPORT_DIR / 'predictions_xgb_v2_all_folds.csv', index=False)

lstm_pred_df = pd.DataFrame(all_lstm_preds)
lstm_pred_df.to_csv(REPORT_DIR / 'predictions_lstm_v2_all_folds.csv', index=False)

print('\n' + '=' * 72)
print('결과 요약')
print('=' * 72)

# fold3 (test 2023-2025) 비교
f3 = result_df[result_df['fold'] == 'fold3'].copy()
f3_summary = f3[f3['model'].isin(['XGBoost v2', 'LSTM v2 (avg)'])]
f3_summary = f3_summary[f3_summary['stage'].isin(['raw', 'CQR', 'raw'])]
print('\n=== fold3 (2023-2025) Interval Score 비교 ===')
cols_show = ['label', 'Interval_Score', 'Coverage_90', 'Sharpness_bp',
             'Dir_Acc', 'Asymmetry_mean', 'Width_std', 'IS_up', 'IS_down']
print(f3[cols_show].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════
# 시각화 1: IS 비교 (fold별)
# ═══════════════════════════════════════════════════════════════════════

print('\n시각화 생성 중...')

# fold3 기준 비교 차트
fig, axes = plt.subplots(1, 4, figsize=(20, 6), facecolor=COLORS['bg_dark'])

f3_models = f3[f3['model'].isin(['XGBoost v2', 'LSTM v2 (avg)'])]
xgb_raw = f3[(f3['model'] == 'XGBoost v2') & (f3['stage'] == 'raw')].iloc[0]
xgb_cqr = f3[(f3['model'] == 'XGBoost v2') & (f3['stage'] == 'CQR')].iloc[0]
lstm_avg = f3[f3['model'] == 'LSTM v2 (avg)'].iloc[0]

labels = ['XGBoost\n(raw)', 'XGBoost\n(CQR)', 'LSTM\n(3시드 평균)']
colors_bar = [COLORS['xgboost'], '#27AE60', COLORS['lstm']]

# (1) Interval Score
ax = axes[0]
ax.set_facecolor(COLORS['bg_card'])
vals = [xgb_raw['Interval_Score'], xgb_cqr['Interval_Score'], lstm_avg['Interval_Score']]
bars = ax.bar(labels, vals, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('Interval Score (낮을수록 좋음)', fontsize=13, fontweight='bold',
             color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'], labelsize=10)
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

# (2) Coverage
ax = axes[1]
ax.set_facecolor(COLORS['bg_card'])
vals = [xgb_raw['Coverage_90'], xgb_cqr['Coverage_90'], lstm_avg['Coverage_90']]
vals_pct = [v * 100 for v in vals]
bars = ax.bar(labels, vals_pct, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)
ax.axhline(y=90, color='#F1C40F', linestyle='--', linewidth=2, alpha=0.8, label='목표 90%')
for b, v in zip(bars, vals_pct):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}%',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('Coverage 90%', fontsize=13, fontweight='bold', color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'], labelsize=10)
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
ax.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'])

# (3) Sharpness
ax = axes[2]
ax.set_facecolor(COLORS['bg_card'])
vals = [xgb_raw['Sharpness_bp'], xgb_cqr['Sharpness_bp'], lstm_avg['Sharpness_bp']]
bars = ax.bar(labels, vals, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.2f}',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('Sharpness (좁을수록 좋음)', fontsize=13, fontweight='bold',
             color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'], labelsize=10)
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

# (4) 구간 폭 변동성 (적응성)
ax = axes[3]
ax.set_facecolor(COLORS['bg_card'])
vals = [xgb_raw['Width_std'], xgb_cqr['Width_std'], lstm_avg['Width_std']]
bars = ax.bar(labels, vals, color=colors_bar, edgecolor='white', linewidth=0.5, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}',
            ha='center', fontsize=11, fontweight='bold', color=COLORS['text'])
ax.set_title('구간 폭 변동성 (높을수록 적응적)', fontsize=13, fontweight='bold',
             color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'], labelsize=10)
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])

fig.suptitle('Interval Score 종합 비교 — fold3 (Test: 2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)

fig.text(0.5, -0.03,
         '[IS = 구간폭 + 벗어남 페널티]  낮은 IS = 좁으면서 정확한 구간  |  '
         'Width_std = 구간 폭이 상황에 따라 변하는 정도 (적응성)',
         ha='center', fontsize=11, color='#F1C40F', fontstyle='italic')

plt.tight_layout()
fig.savefig(FIG_DIR / 'interval_score_comparison.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: reports/figures/improved/interval_score_comparison.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 2: 비대칭성 분석
# ═══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COLORS['bg_dark'])

# fold3 XGBoost raw 예측값
xgb_f3 = xgb_pred_df[xgb_pred_df['fold'] == 'fold3'].copy()
lstm_f3 = lstm_pred_df[lstm_pred_df['fold'] == 'fold3'].copy()

# (1) XGBoost 비대칭 비율 분포
ax = axes[0]
ax.set_facecolor(COLORS['bg_card'])
xgb_ar = asymmetry_ratio(xgb_f3['q05_raw'].values, xgb_f3['q50'].values, xgb_f3['q95_raw'].values)
xgb_ar_clip = np.clip(xgb_ar, 0, 5)
ax.hist(xgb_ar_clip, bins=40, color=COLORS['xgboost'], alpha=0.7, edgecolor='white', linewidth=0.3)
ax.axvline(x=1.0, color=COLORS['accent'], linestyle='--', linewidth=2, label='대칭 (=1.0)')
ax.axvline(x=np.nanmedian(xgb_ar), color='#F1C40F', linestyle='-', linewidth=2,
           label=f'중앙값 {np.nanmedian(xgb_ar):.2f}')
ax.set_title('XGBoost 비대칭 비율 분포\n(q95-q50)/(q50-q05)', fontsize=13,
             fontweight='bold', color=COLORS['text'], pad=12)
ax.set_xlabel('비대칭 비율', color=COLORS['text'])
ax.set_ylabel('빈도', color=COLORS['text'])
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'])

# (2) LSTM 비대칭 비율 분포
ax = axes[1]
ax.set_facecolor(COLORS['bg_card'])
lstm_ar = asymmetry_ratio(lstm_f3['q05_avg'].values, lstm_f3['q50_avg'].values, lstm_f3['q95_avg'].values)
lstm_ar_clip = np.clip(lstm_ar, 0, 5)
ax.hist(lstm_ar_clip, bins=40, color=COLORS['lstm'], alpha=0.7, edgecolor='white', linewidth=0.3)
ax.axvline(x=1.0, color=COLORS['accent'], linestyle='--', linewidth=2, label='대칭 (=1.0)')
ax.axvline(x=np.nanmedian(lstm_ar), color='#F1C40F', linestyle='-', linewidth=2,
           label=f'중앙값 {np.nanmedian(lstm_ar):.2f}')
ax.set_title('LSTM 비대칭 비율 분포\n(q95-q50)/(q50-q05)', fontsize=13,
             fontweight='bold', color=COLORS['text'], pad=12)
ax.set_xlabel('비대칭 비율', color=COLORS['text'])
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.legend(fontsize=9, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'])

# (3) 방향별 IS 비교
ax = axes[2]
ax.set_facecolor(COLORS['bg_card'])
x_pos = np.arange(3)
width = 0.35
is_up_vals = [xgb_raw['IS_up'], xgb_cqr['IS_up'], lstm_avg['IS_up']]
is_dn_vals = [xgb_raw['IS_down'], xgb_cqr['IS_down'], lstm_avg['IS_down']]
bars1 = ax.bar(x_pos - width/2, is_up_vals, width, color='#E74C3C', alpha=0.85,
               label='상승일 IS', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, is_dn_vals, width, color='#3498DB', alpha=0.85,
               label='하락일 IS', edgecolor='white', linewidth=0.5)
for b, v in zip(bars1, is_up_vals):
    if not np.isnan(v):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                ha='center', fontsize=10, fontweight='bold', color='#E74C3C')
for b, v in zip(bars2, is_dn_vals):
    if not np.isnan(v):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f'{v:.1f}',
                ha='center', fontsize=10, fontweight='bold', color='#3498DB')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=10, color=COLORS['text'])
ax.set_title('방향별 Interval Score\n(상승일 vs 하락일)', fontsize=13,
             fontweight='bold', color=COLORS['text'], pad=12)
ax.tick_params(colors=COLORS['text'])
ax.spines[:].set_color(COLORS['grid'])
ax.grid(axis='y', alpha=0.3, color=COLORS['grid'])
ax.legend(fontsize=10, facecolor=COLORS['bg_card'], edgecolor=COLORS['grid'],
          labelcolor=COLORS['text'])

fig.suptitle('예측 구간 비대칭성 분석 — fold3 (Test: 2023~2025)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)

fig.text(0.5, -0.03,
         '비대칭 비율 = 1.0이면 완전 대칭  |  >1이면 상단 넓음 (상승 확신)  |  <1이면 하단 넓음 (하락 확신)',
         ha='center', fontsize=11, color='#F1C40F', fontstyle='italic')

plt.tight_layout()
fig.savefig(FIG_DIR / 'interval_asymmetry_analysis.png',
            facecolor=fig.get_facecolor(), edgecolor='none', dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'  저장: reports/figures/improved/interval_asymmetry_analysis.png')


# ═══════════════════════════════════════════════════════════════════════
# 3-fold 평균 요약
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 72)
print('3-fold 평균 요약')
print('=' * 72)

for model_name in ['XGBoost v2', 'LSTM v2 (avg)']:
    stage_filter = 'raw' if 'LSTM' in model_name else 'CQR'
    subset = result_df[(result_df['model'] == model_name) &
                        (result_df['stage'] == stage_filter) &
                        (result_df['fold'].isin(['fold1', 'fold2', 'fold3']))]
    if len(subset) == 0:
        stage_filter = 'raw'
        subset = result_df[(result_df['model'] == model_name) &
                            (result_df['stage'] == stage_filter) &
                            (result_df['fold'].isin(['fold1', 'fold2', 'fold3']))]

    print(f'\n{model_name} ({stage_filter}):')
    for col in ['Interval_Score', 'Coverage_90', 'Sharpness_bp', 'Dir_Acc',
                'Asymmetry_mean', 'Width_std']:
        vals = subset[col].values
        print(f'  {col:20s}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}')

print('\n완료!')

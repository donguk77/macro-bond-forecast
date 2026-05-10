"""
13_rerun_no_leak.py — Cross-market timing leak (CL-05c) 제거 후 전체 모델 재학습

문제:
  - W2 leakage_audit_w2.csv 의 CL-05c 가 ❌ — 미국 마감변수(us_treasury_10y,
    us_breakeven_10y, vix, sp500, dxy)가 raw[t] 로 사용되어, KR 종가 시점
    (15:30 KST)에 아직 발표되지 않은 미국 종가가 모델 입력으로 들어감.
  - 결과: 방향성 정확도 65% (학술 합격선 53% 대비 +12%p) 비현실적 수치.

수정:
  - notebooks/02b_preprocess_baseline.ipynb §3 의 정책변수 shift(1) 적용 대상에
    미국 마감변수 5개를 추가.

실행:
  - Naive · ARIMA · XGBoost(분위수) · LSTM(분위수, 3 시드) 재학습
  - 평가: RMSE, MAE, dir_acc, Coverage_90, Sharpness, Pinball
  - 결과 저장: reports/no_leak/*.csv + leakage_fix_comparison.md
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import yaml
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.arima.model import ARIMA
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────
# 환경 설정
# ─────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[env] PROJECT_ROOT = {PROJECT_ROOT}')
print(f'[env] device       = {DEVICE}')
print(f'[env] torch        = {torch.__version__}')
print(f'[env] xgboost      = {xgb.__version__}')

# 02b 노트북과 동일한 4-way split (cal 신설)
SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}

TARGET = CONFIG['project']['target']
LAGS = CONFIG['features']['lags']
ROLL_WINDOWS = [5, 10, 20]
LOOKBACK = CONFIG['features']['lookback_window']
LSTM_CFG = CONFIG['models']['lstm']
XGB_CFG = CONFIG['models']['xgboost']
QUANTILES = [0.05, 0.5, 0.95]

# 누수 위치 — 정책 변수 + 미국 마감변수 모두 t-1 강제
POLICY_VARS = ['kr_base_rate', 'us_fed_funds']
US_MARKET_CLOSE_VARS = [
    'us_treasury_10y', 'us_treasury_2y',
    'us_breakeven_10y', 'vix', 'us_hy_oas',
    'wti_oil', 'sp500', 'dxy',
]

# ─────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────

def slice_period(df, period):
    s, e = SPLIT[period]
    return df.loc[s:e]


def metrics_point(y_true, y_pred, name, split):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = (np.sign(y_pred) != 0) & (np.sign(y_true) != 0)
    dir_acc = (
        float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())
        if mask.sum() > 0 else float('nan')
    )
    return {'model': name, 'split': split, 'RMSE_bp': rmse,
            'MAE_bp': mae, 'Dir_Acc': dir_acc}


def pinball_np(y, p, q):
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def eval_quantiles(p, y, label):
    out = {'split': label}
    for q in QUANTILES:
        out[f'pinball_q{int(q*100):02d}'] = pinball_np(y, p[q], q)
    out['coverage_90'] = float(np.mean((y >= p[0.05]) & (y <= p[0.95])))
    out['sharpness_bp'] = float(np.mean(p[0.95] - p[0.05]))
    err = y - p[0.5]
    out['rmse_q50_bp'] = float(np.sqrt(np.mean(err ** 2)))
    out['mae_q50_bp'] = float(np.mean(np.abs(err)))
    mask = (np.sign(p[0.5]) != 0) & (np.sign(y) != 0)
    out['dir_acc_q50'] = (
        float((np.sign(p[0.5][mask]) == np.sign(y[mask])).mean())
        if mask.sum() > 0 else float('nan')
    )
    return out


def sort_quantiles(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


def dm_test_hln(e1, e2, lag=6):
    """Diebold-Mariano test with Newey-West HAC + HLN small-sample correction.
    Returns (DM_HLN_stat, p_value).
    """
    from scipy.stats import t as t_dist
    d = e1 - e2
    n = len(d)
    d_mean = float(d.mean())
    # Newey-West variance
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, lag + 1):
        wk = 1.0 - k / (lag + 1)
        cov = float(np.mean((d[:-k] - d_mean) * (d[k:] - d_mean)))
        var += 2.0 * wk * cov
    var = max(var, 1e-12)
    dm = d_mean / np.sqrt(var / n)
    # HLN small-sample correction
    h = 1
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln_factor
    p_val = float(2.0 * (1.0 - t_dist.cdf(abs(dm_hln), df=n - 1)))
    return float(dm_hln), p_val


# ─────────────────────────────────────────────────────────────────────────
# 1. Features 재생성 (누수 제거)
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('1. Features 재생성 (CL-05c 누수 제거)')
print('=' * 72)

features_v1 = pd.read_csv(
    DATA_DIR / 'processed' / 'features_v1_candidate.csv',
    index_col='date', parse_dates=['date']
).sort_index()

FREEZE_FEATURES = [c for c in features_v1.columns if c != TARGET]
print(f'features_v1.shape = {features_v1.shape}')
print(f'freeze (W3 final 8) = {FREEZE_FEATURES}')

# CL-07: 한국 휴장일 타겟 drop
features_v1 = features_v1.dropna(subset=[TARGET])

# CL-05/05c: 정책변수 + 미국 마감변수 모두 t-1 강제
LAG_VARS = sorted(set(
    [v for v in POLICY_VARS if v in FREEZE_FEATURES]
    + [v for v in US_MARKET_CLOSE_VARS if v in FREEZE_FEATURES]
))
print(f'\nshift(1) 적용 대상 (정책+미국마감) = {LAG_VARS}')

features_safe = features_v1.copy()
for var in LAG_VARS:
    features_safe[var] = features_safe[var].shift(1)
features_safe = features_safe.dropna(subset=LAG_VARS)
print(f'shift 후 features_safe.shape = {features_safe.shape}')

# 타겟 Δy (bp) 생성
y_bp = (features_safe[TARGET].diff() * 100).rename('delta_y_bp')

# Lag features
lag_blocks = [features_safe[c].shift(k).rename(f'{c}__lag{k}')
              for c in FREEZE_FEATURES for k in LAGS]
df_lag = pd.concat(lag_blocks, axis=1)

# Rolling features
roll_blocks = []
for c in FREEZE_FEATURES:
    for w in ROLL_WINDOWS:
        roll_blocks.append(features_safe[c].rolling(w).mean().shift(1).rename(f'{c}__rmean{w}'))
        roll_blocks.append(features_safe[c].rolling(w).std().shift(1).rename(f'{c}__rstd{w}'))
df_roll = pd.concat(roll_blocks, axis=1)

df_features = pd.concat(
    [features_safe[FREEZE_FEATURES], df_lag, df_roll, y_bp.to_frame()],
    axis=1
).dropna()
print(f'lag/rolling 적용 후 df_features.shape = {df_features.shape}')

# Save
out_features = DATA_DIR / 'processed' / 'features_with_lags_v2_no_leak.csv'
df_features.to_csv(out_features, index_label='date')
print(f'[save] {out_features.relative_to(PROJECT_ROOT)}')

# ─────────────────────────────────────────────────────────────────────────
# 2. Train-only scaler
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('2. Scaler train-only fit')
print('=' * 72)

FEATURE_COLS = [c for c in df_features.columns if c != 'delta_y_bp']

X_train_raw = slice_period(df_features, 'train')[FEATURE_COLS]
X_cal_raw = slice_period(df_features, 'cal')[FEATURE_COLS]
X_val_raw = slice_period(df_features, 'val')[FEATURE_COLS]
X_test_raw = slice_period(df_features, 'test')[FEATURE_COLS]

scaler = RobustScaler()
scaler.fit(X_train_raw)


def to_scaled(X_raw):
    return pd.DataFrame(scaler.transform(X_raw), index=X_raw.index, columns=FEATURE_COLS)


X_train = to_scaled(X_train_raw)
X_cal = to_scaled(X_cal_raw)
X_val = to_scaled(X_val_raw)
X_test = to_scaled(X_test_raw)

y_train = slice_period(df_features, 'train')['delta_y_bp']
y_cal = slice_period(df_features, 'cal')['delta_y_bp']
y_val = slice_period(df_features, 'val')['delta_y_bp']
y_test = slice_period(df_features, 'test')['delta_y_bp']

for nm, X, yy in [('train', X_train, y_train), ('cal', X_cal, y_cal),
                  ('val', X_val, y_val), ('test', X_test, y_test)]:
    print(f'  {nm:6s}  X.shape={X.shape}  y.shape={yy.shape}')

# ─────────────────────────────────────────────────────────────────────────
# 3. Naive · ARIMA 베이스라인
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('3. Naive · ARIMA 베이스라인')
print('=' * 72)

baseline_rows = []

# Naive (Δŷ = 0)
for sp_name, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    pred = pd.Series(0.0, index=yy.index)
    baseline_rows.append(metrics_point(yy.values, pred.values, 'Naive', sp_name))

# ARIMA(1,0,1) on Δy
y_full = pd.concat([y_train, y_cal, y_val, y_test]).sort_index()
fit_train = ARIMA(y_train, order=(1, 0, 1), trend='c').fit()
fit_full = fit_train.apply(y_full, refit=False)
preds_full = fit_full.predict()

for sp_name, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    pred = preds_full.reindex(yy.index)
    valid = yy.dropna().index.intersection(pred.dropna().index)
    baseline_rows.append(metrics_point(
        yy.loc[valid].values, pred.loc[valid].values, 'ARIMA(1,0,1)', sp_name))

baseline_df = pd.DataFrame(baseline_rows)
print(baseline_df.round(3).to_string(index=False))
baseline_df.to_csv(REPORT_DIR / 'baseline_no_leak.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# 4. XGBoost 분위수 회귀
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('4. XGBoost 분위수 회귀')
print('=' * 72)


def fit_predict_xgb_quantile(q):
    model = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q,
        n_estimators=XGB_CFG['n_estimators'],
        max_depth=XGB_CFG['max_depth'],
        learning_rate=XGB_CFG['learning_rate'],
        early_stopping_rounds=XGB_CFG['early_stopping_rounds'],
        verbosity=0,
        tree_method='hist',
        random_state=42,
    )
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        verbose=False,
    )
    return {
        'train': model.predict(X_train.values),
        'cal':   model.predict(X_cal.values),
        'val':   model.predict(X_val.values),
        'test':  model.predict(X_test.values),
    }


xgb_preds_per_split = {sp: {} for sp in ['train', 'cal', 'val', 'test']}
for q in QUANTILES:
    print(f'  fitting q={q} ...')
    p_by_sp = fit_predict_xgb_quantile(q)
    for sp, arr in p_by_sp.items():
        xgb_preds_per_split[sp][q] = arr

# Sort post-process
xgb_sorted_per_split = {sp: sort_quantiles(p) for sp, p in xgb_preds_per_split.items()}

xgb_eval = []
for sp, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    xgb_eval.append({**eval_quantiles(xgb_sorted_per_split[sp], yy.values, sp), 'model': 'XGBoost(q)'})
xgb_eval_df = pd.DataFrame(xgb_eval)
print('\n' + xgb_eval_df.round(3).to_string(index=False))
xgb_eval_df.to_csv(REPORT_DIR / 'xgb_quantile_no_leak.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# 5. LSTM 분위수 회귀 (3 시드)
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('5. LSTM 분위수 회귀 (시드 3개)')
print('=' * 72)


def make_sequences(X_df, y_ser, lookback):
    idx_common = X_df.index.intersection(y_ser.index)
    X_arr = X_df.loc[idx_common].to_numpy(dtype=np.float32)
    y_arr = y_ser.loc[idx_common].to_numpy(dtype=np.float32)
    valid = ~np.isnan(y_arr)
    seqs, tgts, dates = [], [], []
    idx_arr = idx_common.to_numpy()
    for t in range(lookback - 1, len(X_arr)):
        if not valid[t]:
            continue
        window = X_arr[t - lookback + 1: t + 1]
        if np.isnan(window).any():
            continue
        seqs.append(window)
        tgts.append(y_arr[t])
        dates.append(idx_arr[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), np.array(dates)


# 04 노트북은 raw 변수만 입력으로 사용 (lag/rolling은 X_train의 일부, 시퀀스 길이로 시간성 표현)
# 동일하게 8개 raw freeze 변수만 사용
INPUT_COLS = FREEZE_FEATURES  # 8개

X_train_lstm = X_train[INPUT_COLS]
X_cal_lstm = X_cal[INPUT_COLS]
X_val_lstm = X_val[INPUT_COLS]
X_test_lstm = X_test[INPUT_COLS]

Xs_train, ys_train, _ = make_sequences(X_train_lstm, y_train, LOOKBACK)
Xs_val, ys_val, _ = make_sequences(X_val_lstm, y_val, LOOKBACK)
Xs_test, ys_test, _ = make_sequences(X_test_lstm, y_test, LOOKBACK)

print(f'  Xs_train.shape = {Xs_train.shape}')
print(f'  Xs_val.shape   = {Xs_val.shape}')
print(f'  Xs_test.shape  = {Xs_test.shape}')


class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, n_q)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def pinball_loss_torch(pred, target, qs=QUANTILES):
    target = target.unsqueeze(1)
    q = torch.tensor(qs, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    diff = target - pred
    return torch.maximum(q * diff, (q - 1) * diff).mean()


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


@torch.no_grad()
def predict_quantiles(model, Xs):
    model.eval()
    xb = torch.from_numpy(Xs).float().to(DEVICE)
    pred = model(xb).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}


def train_lstm(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = QuantileLSTM(
        input_dim=len(INPUT_COLS),
        hidden=LSTM_CFG['hidden_units'],
        num_layers=LSTM_CFG['num_layers'],
        dropout=LSTM_CFG['dropout'],
        n_q=len(QUANTILES),
    ).to(DEVICE)

    train_loader = DataLoader(SeqDataset(Xs_train, ys_train),
                              batch_size=LSTM_CFG['batch_size'],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(SeqDataset(Xs_val, ys_val),
                            batch_size=LSTM_CFG['batch_size'],
                            shuffle=False, drop_last=False)
    optim = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['learning_rate'])

    best_val, best_state, wait = float('inf'), None, 0
    epochs = LSTM_CFG['epochs']
    patience = LSTM_CFG['early_stopping_patience']
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = pinball_loss_torch(model(xb), yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl.append(float(pinball_loss_torch(model(xb), yb).item()))
            vl_loss = float(np.mean(vl))
        if vl_loss < best_val - 1e-6:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, ep


lstm_eval_rows = []
lstm_test_preds_per_seed = {}

for seed in [42, 123, 2024]:
    print(f'\n--- seed={seed} ---')
    model, best_val, n_ep = train_lstm(seed)
    print(f'  epochs={n_ep}  best_val_pinball={best_val:.4f}')

    for sp, Xs, ys in [('train', Xs_train, ys_train),
                       ('val', Xs_val, ys_val),
                       ('test', Xs_test, ys_test)]:
        p_raw = predict_quantiles(model, Xs)
        p_sorted = sort_quantiles(p_raw)
        row = eval_quantiles(p_sorted, ys, sp)
        row.update({'model': 'LSTM(q,sorted)', 'seed': seed,
                    'n_epochs': n_ep, 'best_val_pinball': best_val})
        lstm_eval_rows.append(row)
        if sp == 'test':
            lstm_test_preds_per_seed[seed] = p_sorted

lstm_eval_df = pd.DataFrame(lstm_eval_rows)
print('\n=== LSTM 3 시드 결과 ===')
print(lstm_eval_df.round(3).to_string(index=False))
lstm_eval_df.to_csv(REPORT_DIR / 'lstm_quantile_no_leak.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# 6. DM test (Test set, q50 RMSE 기준)
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('6. DM test (Test set, q50 squared error 기준)')
print('=' * 72)

# Naive vs LSTM(seed=123) on Test set
y_test_arr = ys_test
naive_pred_test = np.zeros_like(y_test_arr)
err_naive = (y_test_arr - naive_pred_test) ** 2

# XGBoost test prediction (q50)
xgb_test_q50 = xgb_sorted_per_split['test'][0.5]
# Note: XGBoost test set 길이는 X_test 기준이지만 LSTM은 lookback 적용 후 길이가 다름
# LSTM 길이로 정렬: X_test 의 마지막 N=len(ys_test) 행에 대응
xgb_test_q50_aligned = xgb_test_q50[-len(y_test_arr):]
err_xgb = (y_test_arr - xgb_test_q50_aligned) ** 2

dm_rows = []
for seed in [42, 123, 2024]:
    p = lstm_test_preds_per_seed[seed]
    err_lstm = (y_test_arr - p[0.5]) ** 2

    for opp_name, err_opp in [('Naive', err_naive), ('XGBoost', err_xgb)]:
        dm, p_val = dm_test_hln(err_lstm, err_opp, lag=6)
        dm_rows.append({
            'seed': seed, 'comparison': f'LSTM_vs_{opp_name}',
            'mean_se_lstm': float(err_lstm.mean()),
            'mean_se_opp': float(err_opp.mean()),
            'DM_HLN': dm, 'p_value': p_val,
            'winner': 'LSTM' if dm < 0 and p_val < 0.05 else (
                'OPP' if dm > 0 and p_val < 0.05 else 'tie'),
        })

dm_df = pd.DataFrame(dm_rows)
print(dm_df.round(4).to_string(index=False))
dm_df.to_csv(REPORT_DIR / 'dm_test_no_leak.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# 7. 누수 전 vs 후 비교 표
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('7. 누수 전 vs 후 비교 (Test set)')
print('=' * 72)

# 누수 전 결과 로드 (W5 final eval)
prev = pd.read_csv(PROJECT_ROOT / 'reports' / 'lstm_a0_final_eval_w5.csv')
prev_test = prev[prev['split'] == 'test'].reset_index(drop=True)

# 누수 후 결과 (현재 lstm_eval_df)
new_test = lstm_eval_df[lstm_eval_df['split'] == 'test'].reset_index(drop=True)

cmp_rows = []
for seed in [42, 123, 2024]:
    pr = prev_test[prev_test['seed'] == seed].iloc[0]
    nw = new_test[new_test['seed'] == seed].iloc[0]
    cmp_rows.append({
        'seed': seed,
        'dir_acc_BEFORE': pr['dir_acc_q50'],
        'dir_acc_AFTER':  nw['dir_acc_q50'],
        'dir_acc_DELTA':  nw['dir_acc_q50'] - pr['dir_acc_q50'],
        'rmse_BEFORE_bp': pr['rmse_q50_bp'],
        'rmse_AFTER_bp':  nw['rmse_q50_bp'],
        'cov_BEFORE':     pr['coverage_90'],
        'cov_AFTER':      nw['coverage_90'],
        'sharp_BEFORE':   pr['sharpness_bp'],
        'sharp_AFTER':    nw['sharpness_bp'],
    })
cmp_df = pd.DataFrame(cmp_rows)
print(cmp_df.round(4).to_string(index=False))
cmp_df.to_csv(REPORT_DIR / 'leakage_fix_comparison_test.csv', index=False)

# ─────────────────────────────────────────────────────────────────────────
# 8. 요약 마크다운
# ─────────────────────────────────────────────────────────────────────────

md_path = REPORT_DIR / 'leakage_fix_comparison.md'
mean_before = cmp_df['dir_acc_BEFORE'].mean()
mean_after = cmp_df['dir_acc_AFTER'].mean()
mean_delta = mean_after - mean_before

lines = []
lines.append('# 누수 제거 (CL-05c) 후 재학습 결과 비교')
lines.append('')
lines.append('## 누수 위치')
lines.append('- `notebooks/02b_preprocess_baseline.ipynb` §3 — 정책변수만 shift(1) 적용,')
lines.append('  미국 마감변수(us_treasury_10y, us_breakeven_10y, vix, sp500, dxy)는 raw[t] 사용')
lines.append('- KR 종가(15:30 KST) 시점에 아직 발표되지 않은 미국 종가 입력 → 미래 정보 누수')
lines.append('')
lines.append('## 수정')
lines.append('- 정책변수 + 미국 마감변수 모두 shift(1) → raw[t-1] 사용으로 통일')
lines.append('- 새 features: `data/processed/features_with_lags_v2_no_leak.csv`')
lines.append('')
lines.append('## Test set 비교 (LSTM 분위수 회귀, 3 시드)')
lines.append('')
lines.append('| seed | dir_acc 누수전 | dir_acc 누수후 | Δ | RMSE 전 | RMSE 후 | Cov 전 | Cov 후 |')
lines.append('|---|---|---|---|---|---|---|---|')
for _, r in cmp_df.iterrows():
    lines.append(f"| {int(r['seed'])} "
                 f"| {r['dir_acc_BEFORE']:.4f} "
                 f"| {r['dir_acc_AFTER']:.4f} "
                 f"| **{r['dir_acc_DELTA']:+.4f}** "
                 f"| {r['rmse_BEFORE_bp']:.3f} "
                 f"| {r['rmse_AFTER_bp']:.3f} "
                 f"| {r['cov_BEFORE']:.4f} "
                 f"| {r['cov_AFTER']:.4f} |")
lines.append('')
lines.append(f'**평균 dir_acc**: {mean_before:.4f} → {mean_after:.4f} (Δ {mean_delta:+.4f})')
lines.append('')
lines.append('## DM test (LSTM vs Naive · XGBoost, q50 squared error)')
lines.append('')
lines.append('| seed | comparison | DM_HLN | p_value | winner |')
lines.append('|---|---|---|---|---|')
for _, r in dm_df.iterrows():
    lines.append(f"| {int(r['seed'])} | {r['comparison']} "
                 f"| {r['DM_HLN']:.3f} | {r['p_value']:.4f} | {r['winner']} |")
lines.append('')
lines.append('## 해석')
if mean_delta < -0.05:
    lines.append('- **누수 영향 큼**: 평균 dir_acc 가 5%p 이상 하락 — 누수가 결과를 크게 부풀렸음')
elif mean_delta < -0.02:
    lines.append('- **누수 영향 중간**: 평균 dir_acc 2~5%p 하락 — 영향 있으나 모델 자체 가치 일부 유지')
else:
    lines.append('- **누수 영향 작음**: 평균 dir_acc 변화 < 2%p — 모델 결과 대체로 유효')
lines.append(f'- 학술 합격선 53% 대비: 누수 후 평균 {mean_after:.1%}')
lines.append('')

md_path.write_text('\n'.join(lines), encoding='utf-8')
print(f'\n[save] {md_path.relative_to(PROJECT_ROOT)}')
print('\n=== 완료 ===')

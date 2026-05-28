"""
16f_lstm_v3_predictions.py
v3 LSTM(13변수, kospi 제외) walk-forward 3-fold × 3-seed 재학습.
seed-평균 q05/q50/q95 예측을 날짜별로 저장 → reports/no_leak_v2/predictions_lstm_v3_all_folds.csv
(16c_lstm_v2_full_metrics.py 와 동일 파이프라인, 입력 변수만 kospi 제외)
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
SEEDS = [42, 123, 2024]

print(f'16f — LSTM v3 (kospi 제외) | device={DEVICE} | lookback={LOOKBACK}')

FOLDS = [
    {'name': 'fold1', 'train': ('2010-01-01','2017-12-31'), 'val': ('2018-01-01','2019-12-31'), 'test': ('2020-01-01','2020-12-31')},
    {'name': 'fold2', 'train': ('2010-01-01','2019-12-31'), 'val': ('2020-01-01','2020-12-31'), 'test': ('2021-01-01','2022-12-31')},
    {'name': 'fold3', 'train': ('2010-01-01','2021-12-31'), 'val': ('2022-01-01','2022-12-31'), 'test': ('2023-01-01','2025-12-31')},
]

df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

# v3: kospi 제외한 13개 raw 입력
RAW_INPUT = ['kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
             'us_breakeven_10y','vix','sp500','dxy',
             'spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy']
RAW_INPUT = [c for c in RAW_INPUT if c in df.columns]
print(f'LSTM v3 inputs ({len(RAW_INPUT)}):', RAW_INPUT)


class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
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
        self.X = torch.from_numpy(X).float(); self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_seq(X_df, y_ser, lookback):
    idx = X_df.index.intersection(y_ser.index)
    X_arr = X_df.loc[idx].to_numpy(dtype=np.float32)
    y_arr = y_ser.loc[idx].to_numpy(dtype=np.float32)
    valid = ~np.isnan(y_arr)
    date_index = X_df.loc[idx].index
    seqs, tgts, dates = [], [], []
    for t in range(lookback - 1, len(X_arr)):
        if not valid[t]: continue
        win = X_arr[t - lookback + 1: t + 1]
        if np.isnan(win).any(): continue
        seqs.append(win); tgts.append(y_arr[t]); dates.append(date_index[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), dates


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


def train_one(seed, Xtr, ytr, Xval, yval):
    torch.manual_seed(seed); np.random.seed(seed)
    m = QuantileLSTM(Xtr.shape[2], LSTM_CFG['hidden_units'], LSTM_CFG['num_layers'],
                     LSTM_CFG['dropout'], len(QUANTILES)).to(DEVICE)
    tr = DataLoader(SeqDS(Xtr, ytr), batch_size=LSTM_CFG['batch_size'], shuffle=True)
    vl = DataLoader(SeqDS(Xval, yval), batch_size=LSTM_CFG['batch_size'], shuffle=False)
    opt = torch.optim.Adam(m.parameters(), lr=LSTM_CFG['learning_rate'])
    best, best_st, wait = float('inf'), None, 0
    for ep in range(1, LSTM_CFG['epochs'] + 1):
        m.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); pinball_loss_torch(m(xb), yb).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vloss = float(np.mean([float(pinball_loss_torch(m(xb.to(DEVICE)), yb.to(DEVICE)).item()) for xb, yb in vl]))
        if vloss < best - 1e-6:
            best, wait = vloss, 0
            best_st = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= LSTM_CFG['early_stopping_patience']: break
    if best_st is not None: m.load_state_dict(best_st)
    return m, ep


@torch.no_grad()
def predict(m, Xs):
    m.eval()
    pred = m(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    return {q: pred[:, i] for i, q in enumerate(QUANTILES)}


rows = []
for fold in FOLDS:
    name = fold['name']
    def sl(p): return df.loc[fold[p][0]:fold[p][1]]
    scaler = RobustScaler().fit(sl('train')[RAW_INPUT])
    def s(X): return pd.DataFrame(scaler.transform(X[RAW_INPUT]), index=X.index, columns=RAW_INPUT)
    Xtr, ytr_s = s(sl('train')), sl('train')['delta_y_bp']
    Xval, yval_s = s(sl('val')), sl('val')['delta_y_bp']
    Xte, yte_s = s(sl('test')), sl('test')['delta_y_bp']
    Xs_tr, ys_tr, _ = make_seq(Xtr, ytr_s, LOOKBACK)
    Xs_val, ys_val, _ = make_seq(Xval, yval_s, LOOKBACK)
    Xs_te, ys_te, dates_te = make_seq(Xte, yte_s, LOOKBACK)
    print(f'{name}: tr={Xs_tr.shape} val={Xs_val.shape} te={Xs_te.shape}')

    seed_preds = {q: [] for q in QUANTILES}
    for seed in SEEDS:
        m, ne = train_one(seed, Xs_tr, ys_tr, Xs_val, ys_val)
        p = sort_qs(predict(m, Xs_te))
        for q in QUANTILES: seed_preds[q].append(p[q])
        da = float((np.sign(p[0.5][(np.sign(p[0.5])!=0)&(np.sign(ys_te)!=0)]) ==
                    np.sign(ys_te[(np.sign(p[0.5])!=0)&(np.sign(ys_te)!=0)])).mean())
        print(f'  seed={seed} epochs={ne} dir={da:.4f}')
    avg = {q: np.mean(seed_preds[q], axis=0) for q in QUANTILES}
    width = avg[0.95] - avg[0.05]
    IS = width + (2/ALPHA)*(avg[0.05]-ys_te)*(ys_te<avg[0.05]) + (2/ALPHA)*(ys_te-avg[0.95])*(ys_te>avg[0.95])
    for i, dt in enumerate(dates_te):
        rows.append({'date': dt, 'fold': name, 'y_true': float(ys_te[i]),
                     'q05_avg': float(avg[0.05][i]), 'q50_avg': float(avg[0.5][i]),
                     'q95_avg': float(avg[0.95][i]), 'IS': float(IS[i])})

out = pd.DataFrame(rows)
out.to_csv(REPORT_DIR / 'predictions_lstm_v3_all_folds.csv', index=False)
y = out['y_true'].values; p = out['q50_avg'].values
msk = (np.sign(p) != 0) & (np.sign(y) != 0)
print(f'\nSaved predictions_lstm_v3_all_folds.csv {out.shape}')
print(f'pooled RMSE={np.sqrt(np.mean((y-p)**2)):.3f} dir={(np.sign(p[msk])==np.sign(y[msk])).mean():.4f}')

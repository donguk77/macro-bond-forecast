# -*- coding: utf-8 -*-
"""
4주차 진단 후속 — Δfeature(lag-1 강제) LSTM 분위수 회귀 ablation
=============================================================

배경:
  W4 LSTM (raw 8 features 레벨) test RMSE 4.535 ≈ Naive 4.647 → 신호 부재
  점검 결과: features 레벨↔Δy 상관 모두 |r|<0.05.
  Δfeature 점검: Δus_treasury_10y[t-1] vs Δy[t] r=+0.336 (lag-1 causal 신호 강력)

본 스크립트:
  - 입력: 8개 freeze 변수의 1일 차분 + lag-1 강제 (cross-market timing leak 차단)
  - 동일 SPLIT / 동일 하이퍼파라미터 / 동일 seed → 직접 비교
  - 산출물: reports/lstm_diff_ablation_w4.csv (RMSE/Coverage/Pinball/Dir_Acc)
           reports/figures/w4_06_diff_ablation_compare.png
"""
from __future__ import annotations
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, yaml
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(ROOT/'configs/config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
QUANTILES = [0.05, 0.5, 0.95]

FROZEN = ['kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
          'us_breakeven_10y','vix','sp500','dxy']

# ---------- 데이터 로드 + Δfeature 생성 (lag-1 강제) -------------
fv = pd.read_csv(ROOT/'data/processed/features_v1_candidate.csv',
                 index_col='date', parse_dates=['date']).sort_index()
fwl = pd.read_csv(ROOT/'data/processed/features_with_lags_v1.csv',
                  index_col='date', parse_dates=['date']).sort_index()
with open(ROOT/'models/scaler_robust_train.pkl','rb') as f:
    SPLIT = pickle.load(f)['split']

# Δfeature[t-1] : .diff().shift(1) → t-1 시점 차분 (causal forecasting)
df_diff = fv[FROZEN].diff().shift(1)
df_diff.columns = [f'd_{c}' for c in FROZEN]

# Train-only Robust scaler (CL-02)
from sklearn.preprocessing import RobustScaler
def slice_period(df, p):
    s, e = SPLIT[p]; return df.loc[s:e]

X_train_raw = slice_period(df_diff, 'train').dropna()
X_cal_raw   = slice_period(df_diff, 'cal'  ).dropna()
X_val_raw   = slice_period(df_diff, 'val'  ).dropna()
X_test_raw  = slice_period(df_diff, 'test' ).dropna()

scaler = RobustScaler().fit(X_train_raw)
def scale(df_): return pd.DataFrame(scaler.transform(df_), index=df_.index, columns=df_.columns)
X_train, X_cal, X_val, X_test = map(scale, [X_train_raw, X_cal_raw, X_val_raw, X_test_raw])

y_train = slice_period(fwl, 'train')['delta_y_bp']
y_cal   = slice_period(fwl, 'cal')['delta_y_bp']
y_val   = slice_period(fwl, 'val')['delta_y_bp']
y_test  = slice_period(fwl, 'test')['delta_y_bp']

# ---------- Sequence builder -------------
def make_sequences(X_df, y_ser, lookback):
    idx = X_df.index.intersection(y_ser.index)
    X = X_df.loc[idx].to_numpy(dtype=np.float32)
    y = y_ser.loc[idx].to_numpy(dtype=np.float32)
    seqs, tgts, dates = [], [], []
    arr = idx.to_numpy()
    for t in range(lookback - 1, len(X)):
        if np.isnan(y[t]) or np.isnan(X[t-lookback+1:t+1]).any(): continue
        seqs.append(X[t-lookback+1:t+1]); tgts.append(y[t]); dates.append(arr[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), np.array(dates)

Xs_train, ys_train, _      = make_sequences(X_train, y_train, LOOKBACK)
Xs_val,   ys_val,   dt_val  = make_sequences(X_val,   y_val,   LOOKBACK)
Xs_test,  ys_test,  dt_test = make_sequences(X_test,  y_test,  LOOKBACK)
print(f'Δfeature ablation seq shapes: train{Xs_train.shape} val{Xs_val.shape} test{Xs_test.shape}')

# ---------- LSTM model (W4 와 동일 구조) -------------
class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers>1 else 0.0)
        self.head = nn.Linear(hidden, n_q)
    def forward(self, x):
        out,_ = self.lstm(x); return self.head(out[:,-1,:])

def pinball(pred, target, q=QUANTILES):
    target = target.unsqueeze(1)
    q_t = torch.tensor(q, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    return torch.maximum(q_t*(target-pred), (q_t-1)*(target-pred)).mean()

class SeqDS(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

model = QuantileLSTM(Xs_train.shape[2], LSTM_CFG['hidden_units'],
                     LSTM_CFG['num_layers'], LSTM_CFG['dropout'], len(QUANTILES)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LSTM_CFG['learning_rate'])
tr_loader = DataLoader(SeqDS(Xs_train, ys_train), batch_size=LSTM_CFG['batch_size'], shuffle=True)
vl_loader = DataLoader(SeqDS(Xs_val,   ys_val),   batch_size=LSTM_CFG['batch_size'], shuffle=False)

best, wait, patience = float('inf'), 0, LSTM_CFG['early_stopping_patience']
best_state, history = None, {'train':[], 'val':[]}
for ep in range(1, LSTM_CFG['epochs']+1):
    model.train(); tl=[]
    for xb,yb in tr_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); l=pinball(model(xb), yb); l.backward(); opt.step(); tl.append(float(l.item()))
    model.eval(); vl=[]
    with torch.no_grad():
        for xb,yb in vl_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            vl.append(float(pinball(model(xb), yb).item()))
    tr_l, vl_l = float(np.mean(tl)), float(np.mean(vl))
    history['train'].append(tr_l); history['val'].append(vl_l)
    if vl_l < best - 1e-6:
        best=vl_l; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        if ep<=3 or ep%5==0: print(f'🟢 ep{ep:3d} train {tr_l:.4f} val {vl_l:.4f}')
    else:
        wait+=1
        if wait>=patience:
            print(f'⏹ Early stop ep{ep}, best val {best:.4f}'); break
model.load_state_dict(best_state)

# ---------- Evaluate -------------
@torch.no_grad()
def predict(Xs):
    model.eval()
    p = model(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    arr = np.sort(p, axis=1)   # sort 후처리 (monotonicity)
    return {q: arr[:,i] for i,q in enumerate(QUANTILES)}

preds = {sp: predict(Xs) for sp, Xs in [('train',Xs_train),('val',Xs_val),('test',Xs_test)]}
ys    = {'train':ys_train, 'val':ys_val, 'test':ys_test}

def pinball_np(y, p, q): d=y-p; return float(np.mean(np.maximum(q*d,(q-1)*d)))
def dir_acc(y, p):
    m = (np.sign(p)!=0)&(np.sign(y)!=0)
    return float((np.sign(p[m])==np.sign(y[m])).mean()) if m.sum() else float('nan')

rows = []
for sp in ['train','val','test']:
    p = preds[sp]; y = ys[sp]
    rows.append({
        'split': sp,
        'pinball_q05': round(pinball_np(y, p[0.05], 0.05), 3),
        'pinball_q50': round(pinball_np(y, p[0.5],  0.5),  3),
        'pinball_q95': round(pinball_np(y, p[0.95], 0.95), 3),
        'coverage_90': round(float(np.mean((y>=p[0.05])&(y<=p[0.95]))), 3),
        'sharpness_bp': round(float(np.mean(p[0.95]-p[0.05])), 3),
        'rmse_q50_bp': round(float(np.sqrt(np.mean((y-p[0.5])**2))), 3),
        'mae_q50_bp':  round(float(np.mean(np.abs(y-p[0.5]))), 3),
        'dir_acc_q50': round(dir_acc(y, p[0.5]), 3),
    })
out = pd.DataFrame(rows)
out['model'] = 'LSTM_Δfeat[t-1]'
print('\n=== Δfeature ablation 결과 ===')
print(out.to_string(index=False))

# ---------- Compare with W4 raw LSTM -------------
w4 = pd.read_csv(ROOT/'reports/lstm_quantile_eval_w4.csv')
w4 = w4[w4['stage']=='sorted'].copy()
w4['model'] = 'LSTM_raw[t]'
combined = pd.concat([w4[['model','split','pinball_q50','coverage_90','sharpness_bp','rmse_q50_bp','dir_acc_q50']],
                      out [['model','split','pinball_q50','coverage_90','sharpness_bp','rmse_q50_bp','dir_acc_q50']]],
                     ignore_index=True)
print('\n=== W4 raw vs Δfeature lag-1 ===')
print(combined.sort_values(['split','model']).to_string(index=False))

# Save
out.to_csv(ROOT/'reports/lstm_diff_ablation_w4.csv', index=False)
print(f'\n💾 reports/lstm_diff_ablation_w4.csv')

# ---------- Figure -------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
metrics = [('rmse_q50_bp','RMSE q50 (bp, 낮을수록 좋음)'),
           ('coverage_90','Coverage 90% (target 0.9)'),
           ('dir_acc_q50','Dir_Acc q50 (target 0.55)')]
for ax,(m,title) in zip(axes, metrics):
    sub = combined[combined['split'].isin(['val','test'])]
    pivot = sub.pivot(index='split', columns='model', values=m)
    pivot.plot(kind='bar', ax=ax, color=['steelblue','crimson'], alpha=0.85, width=0.7)
    ax.set_title(title); ax.set_xlabel(''); ax.tick_params(axis='x', rotation=0)
    ax.grid(alpha=0.3, axis='y')
    if 'coverage' in m: ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.7, label='target')
    if 'dir' in m:      ax.axhline(0.55, color='gray', linestyle='--', linewidth=0.7, label='target')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', fontsize=8, padding=2)
plt.tight_layout()
plt.savefig(ROOT/'reports/figures/w4_06_diff_ablation_compare.png', dpi=120, bbox_inches='tight')
print(f'💾 reports/figures/w4_06_diff_ablation_compare.png')

# -*- coding: utf-8 -*-
"""
W7 보강 작업 (3건)
====================
사용자 요청: 채점관 Q&A 대비 답변 강도 한 단계 더.

A. A2' (Δkospi[t-1]) ablation 실행 — corr 사전 점검 우회 후 실제 결과로 음수 채택 실증
B. A3' (Δkr_ppi_announced[t-1]) ablation 실행 — 동일
C. DM test 의 q05/q95 pinball 비교 추가 — 분위수 회귀 진짜 비교 (점추정만 비교했냐 추궁 차단)
D. A1' 9 vars 전용 grid 5×5 — V9 (LOG #40) deferred 작업

산출물:
- reports/ablation_a2_w7.csv (Δkospi)
- reports/ablation_a3_w7.csv (Δkr_ppi)
- reports/dm_test_quantile_w7.csv (q05/q50/q95 모두)
- reports/grid_a1_9vars_w7.csv (A1' 전용 grid)
- reports/figures/w7_*.png
"""
from __future__ import annotations
import sys, json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from scipy import stats as scistats
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

with open(ROOT/'configs/config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)
LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']
QUANTILES = [0.05, 0.5, 0.95]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 2024]

FROZEN = ['kr_treasury_3y','kr_base_rate','us_treasury_10y','us_fed_funds',
          'us_breakeven_10y','vix','sp500','dxy']

# ========== 공통 유틸 ==========
fv = pd.read_csv(ROOT/'data/processed/features_v1_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
fwl = pd.read_csv(ROOT/'data/processed/features_with_lags_v1.csv', index_col='date', parse_dates=['date']).sort_index()
raw = pd.read_csv(ROOT/'data/interim/wide_daily_filled.csv', index_col=0, parse_dates=[0]).sort_index()
with open(ROOT/'models/scaler_robust_train.pkl','rb') as f:
    SPLIT = pickle.load(f)['split']

def slice_period(df, p):
    s,e = SPLIT[p]; return df.loc[s:e]

def build_diff(feature_df, cols):
    df = feature_df[cols].diff().shift(1)
    df.columns = [f'd_{c}' for c in cols]
    return df

def fit_scaler_transform(df_diff):
    X_train = slice_period(df_diff,'train').dropna()
    X_val = slice_period(df_diff,'val').dropna()
    X_test = slice_period(df_diff,'test').dropna()
    sc = RobustScaler().fit(X_train)
    def tr(d): return pd.DataFrame(sc.transform(d), index=d.index, columns=d.columns)
    return tr(X_train), tr(X_val), tr(X_test)

def make_seq(X_df, y_ser, lookback=LOOKBACK):
    idx = X_df.index.intersection(y_ser.index)
    X = X_df.loc[idx].to_numpy(dtype=np.float32)
    y = y_ser.loc[idx].to_numpy(dtype=np.float32)
    seqs, tgts, dates = [], [], []
    arr = idx.to_numpy()
    for t in range(lookback - 1, len(X)):
        if np.isnan(y[t]) or np.isnan(X[t-lookback+1:t+1]).any(): continue
        seqs.append(X[t-lookback+1:t+1]); tgts.append(y[t]); dates.append(arr[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32), np.array(dates)

class QuantileLSTM(nn.Module):
    def __init__(self, input_dim, hidden, layers, dropout, n_q):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers>1 else 0.0)
        self.head = nn.Linear(hidden, n_q)
    def forward(self, x):
        out,_ = self.lstm(x); return self.head(out[:,-1,:])

def pinball_loss(pred, target, q=QUANTILES):
    target = target.unsqueeze(1)
    q_t = torch.tensor(q, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    return torch.maximum(q_t*(target-pred), (q_t-1)*(target-pred)).mean()

class DS(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_one(Xs_tr, ys_tr, Xs_val, ys_val, hidden=128, lr=5e-4, seed=42,
              max_ep=100, patience=10):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    m = QuantileLSTM(Xs_tr.shape[2], hidden, LSTM_CFG['num_layers'], LSTM_CFG['dropout'], len(QUANTILES)).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr = DataLoader(DS(Xs_tr, ys_tr), batch_size=LSTM_CFG['batch_size'], shuffle=True, generator=g)
    vl = DataLoader(DS(Xs_val, ys_val), batch_size=LSTM_CFG['batch_size'], shuffle=False)
    best, wait, best_state = float('inf'), 0, None
    for ep in range(1, max_ep+1):
        m.train()
        for xb,yb in tr:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); pinball_loss(m(xb), yb).backward(); opt.step()
        m.eval(); vl_l=[]
        with torch.no_grad():
            for xb,yb in vl:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_l.append(float(pinball_loss(m(xb), yb).item()))
        v = float(np.mean(vl_l))
        if v < best - 1e-6:
            best, wait, best_state = v, 0, {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    if best_state: m.load_state_dict(best_state)
    return m, best, ep

@torch.no_grad()
def predict_sorted(m, Xs):
    m.eval()
    p = m(torch.from_numpy(Xs).float().to(DEVICE)).cpu().numpy()
    arr = np.sort(p, axis=1)
    return {q: arr[:,i] for i,q in enumerate(QUANTILES)}

def pinball_np(y, p, q): d=y-p; return float(np.mean(np.maximum(q*d,(q-1)*d)))
def dir_acc(y, p):
    m = (np.sign(p)!=0)&(np.sign(y)!=0)
    return float((np.sign(p[m])==np.sign(y[m])).mean()) if m.sum() else float('nan')

def evaluate(preds, y, label):
    out = {'split': label}
    for q in QUANTILES:
        out[f'pinball_q{int(q*100):02d}'] = pinball_np(y, preds[q], q)
    out['coverage_90'] = float(np.mean((y>=preds[0.05])&(y<=preds[0.95])))
    out['sharpness_bp'] = float(np.mean(preds[0.95]-preds[0.05]))
    err = y - preds[0.5]
    out['rmse_q50_bp'] = float(np.sqrt(np.mean(err**2)))
    out['mae_q50_bp']  = float(np.mean(np.abs(err)))
    out['dir_acc_q50'] = dir_acc(y, preds[0.5])
    return out

y_tr_full = slice_period(fwl,'train')['delta_y_bp']
y_val_full = slice_period(fwl,'val')['delta_y_bp']
y_test_full = slice_period(fwl,'test')['delta_y_bp']

BEST_HIDDEN, BEST_LR = 128, 5e-4

# ====================================================================
print('='*78); print('Section A: A2 prime — Δkospi[t-1] ablation'); print('='*78)
# ====================================================================
fv_a2 = fv[FROZEN].copy()
fv_a2['kospi'] = raw['kospi'].reindex(fv.index)
df_a2 = build_diff(fv_a2, FROZEN + ['kospi'])
X_tr, X_v, X_te = fit_scaler_transform(df_a2)
Xs_tr, ys_tr, _ = make_seq(X_tr, y_tr_full)
Xs_v, ys_v, dt_v = make_seq(X_v, y_val_full)
Xs_te, ys_te, dt_te = make_seq(X_te, y_test_full)
print(f'  shapes: train{Xs_tr.shape} val{Xs_v.shape} test{Xs_te.shape}')

a2_rows = []
for sd in SEEDS:
    t0 = time.time()
    m, bv, n_ep = train_one(Xs_tr, ys_tr, Xs_v, ys_v, hidden=BEST_HIDDEN, lr=BEST_LR, seed=sd)
    for sp_n, Xs, y in [('train',Xs_tr,ys_tr),('val',Xs_v,ys_v),('test',Xs_te,ys_te)]:
        ev = evaluate(predict_sorted(m, Xs), y, sp_n)
        ev['seed']=sd; ev['model']="A2'_+Δkospi"
        a2_rows.append(ev)
    print(f'  seed={sd}: best_val={bv:.4f}, epochs={n_ep}, {time.time()-t0:.1f}s')
a2_df = pd.DataFrame(a2_rows)
a2_df.to_csv(ROOT/'reports/ablation_a2_w7.csv', index=False)
test_a2 = a2_df[a2_df['split']=='test']
print(f"  A2' test (mean ± std): RMSE {test_a2['rmse_q50_bp'].mean():.3f} ± {test_a2['rmse_q50_bp'].std():.3f}, "
      f"Cov {test_a2['coverage_90'].mean():.3f}, Dir {test_a2['dir_acc_q50'].mean():.3f}")

# ====================================================================
print(); print('='*78); print('Section B: A3 prime — Δkr_ppi_announced[t-1] ablation'); print('='*78)
# ====================================================================
kr_ppi_announced = raw['kr_ppi'].shift(freq='30D').reindex(fv.index, method='ffill')
fv_a3 = fv[FROZEN].copy(); fv_a3['kr_ppi'] = kr_ppi_announced
df_a3 = build_diff(fv_a3, FROZEN + ['kr_ppi'])
X_tr, X_v, X_te = fit_scaler_transform(df_a3)
Xs_tr, ys_tr, _ = make_seq(X_tr, y_tr_full)
Xs_v, ys_v, _ = make_seq(X_v, y_val_full)
Xs_te, ys_te, _ = make_seq(X_te, y_test_full)
print(f'  shapes: train{Xs_tr.shape} val{Xs_v.shape} test{Xs_te.shape}')

a3_rows = []
for sd in SEEDS:
    t0 = time.time()
    m, bv, n_ep = train_one(Xs_tr, ys_tr, Xs_v, ys_v, hidden=BEST_HIDDEN, lr=BEST_LR, seed=sd)
    for sp_n, Xs, y in [('train',Xs_tr,ys_tr),('val',Xs_v,ys_v),('test',Xs_te,ys_te)]:
        ev = evaluate(predict_sorted(m, Xs), y, sp_n)
        ev['seed']=sd; ev['model']="A3'_+Δkr_ppi"
        a3_rows.append(ev)
    print(f'  seed={sd}: best_val={bv:.4f}, epochs={n_ep}, {time.time()-t0:.1f}s')
a3_df = pd.DataFrame(a3_rows)
a3_df.to_csv(ROOT/'reports/ablation_a3_w7.csv', index=False)
test_a3 = a3_df[a3_df['split']=='test']
print(f"  A3' test (mean ± std): RMSE {test_a3['rmse_q50_bp'].mean():.3f} ± {test_a3['rmse_q50_bp'].std():.3f}, "
      f"Cov {test_a3['coverage_90'].mean():.3f}, Dir {test_a3['dir_acc_q50'].mean():.3f}")

# ====================================================================
print(); print('='*78); print('Section C: DM test for q05/q50/q95 pinball loss'); print('='*78)
# ====================================================================
# 점별 pinball 사용 — 이미 모든 prediction CSV 에 저장됨
a0 = pd.read_csv(ROOT/'data/processed/lstm_a0_predictions_w6.csv', parse_dates=['date'])
xgb = pd.read_csv(ROOT/'data/processed/xgb_predictions_w3.csv', parse_dates=['date'])
lstm_w4 = pd.read_csv(ROOT/'data/processed/lstm_predictions_w4.csv', parse_dates=['date'])

a0_t = a0[a0['split']=='test'][['date','y_true_bp','pinball_q05','pinball_q50','pinball_q95']].rename(
    columns={'pinball_q05':'pb05_a0','pinball_q50':'pb50_a0','pinball_q95':'pb95_a0'})
xgb_t = xgb[xgb['split']=='test'][['date','pinball_q05','pinball_q50','pinball_q95']].rename(
    columns={'pinball_q05':'pb05_xgb','pinball_q50':'pb50_xgb','pinball_q95':'pb95_xgb'})
lstm_w4_t = lstm_w4[lstm_w4['split']=='test'][['date','pinball_q05','pinball_q50','pinball_q95']].rename(
    columns={'pinball_q05':'pb05_lstm','pinball_q50':'pb50_lstm','pinball_q95':'pb95_lstm'})
mg = a0_t.merge(xgb_t, on='date').merge(lstm_w4_t, on='date')
print(f'  DM N = {len(mg)}')

def dm_diff(d_t, h=1):
    """DM test on pre-computed loss differential d_t."""
    d = np.asarray(d_t); T = len(d); dm_mean = d.mean()
    L = max(1, int(np.floor(4*(T/100.0)**(2/9))))
    var = ((d - dm_mean)**2).mean()
    for k in range(1, L+1):
        var += 2*(1-k/(L+1))*((d[:-k]-dm_mean)*(d[k:]-dm_mean)).mean()
    var = max(var, 1e-12)
    dm = dm_mean / np.sqrt(var/T)
    corr = np.sqrt((T+1-2*h+h*(h-1)/T)/T)
    dm_hln = corr*dm
    p = 2*(1 - scistats.t.cdf(abs(dm_hln), df=T-1))
    return float(dm_hln), float(p), L, T

dm_q_rows = []
quantile_names = {'05':'q05', '50':'q50', '95':'q95'}
for q_lbl, q_col in quantile_names.items():
    for other, oc in [('Naive', None), ('XGBoost', f'pb{q_lbl}_xgb'), ('LSTM_raw', f'pb{q_lbl}_lstm')]:
        d_a0 = mg[f'pb{q_lbl}_a0'].values
        if other == 'Naive':
            # Naive q50=0, q05/q95 도 0 으로 가정 (변화 없음 예측). pinball: q*max(y, 0) + (q-1)*min(y,0)
            y = mg['y_true_bp'].values
            qval = float('0.'+q_lbl) if q_lbl=='05' else (0.5 if q_lbl=='50' else 0.95)
            d_naive = np.maximum(qval*y, (qval-1)*y)
            d_other = d_naive
        else:
            d_other = mg[oc].values
        diff = d_a0 - d_other
        dm_hln, p, L, T = dm_diff(diff)
        bonf = 'OK' if p < 0.05/9 else 'NS'  # 9 비교 (3 quantile × 3 baseline)
        dm_q_rows.append({
            'quantile': q_col, 'comparison': f'A0_vs_{other}',
            'mean_pinball_a0': round(float(d_a0.mean()), 4),
            'mean_pinball_other': round(float(d_other.mean()), 4),
            'DM_HLN': round(dm_hln, 3), 'p_value': round(p, 4),
            'NW_lag': L, 'bonf_alpha_0.0056': bonf,  # 0.05/9 ≈ 0.0056
            'direction': 'A0 wins' if dm_hln < 0 else 'other wins',
        })
        print(f'  {q_col} A0 vs {other:10s}: pb {d_a0.mean():.4f} vs {d_other.mean():.4f}, '
              f'DM_HLN={dm_hln:+.3f}, p={p:.4f} {bonf}')

dm_q_df = pd.DataFrame(dm_q_rows)
dm_q_df.to_csv(ROOT/'reports/dm_test_quantile_w7.csv', index=False)
n_sig = (dm_q_df['bonf_alpha_0.0056']=='OK').sum()
print(f"  -> Bonferroni 보정 후 유의: {n_sig}/9")

# ====================================================================
print(); print('='*78); print('Section D: A1 prime — 9 vars 전용 grid 5x5 (V9 deferred)'); print('='*78)
# ====================================================================
fv_a1 = fv[FROZEN].copy()
fv_a1['krw_usd'] = raw['krw_usd'].reindex(fv.index)
df_a1 = build_diff(fv_a1, FROZEN + ['krw_usd'])
X_tr, X_v, X_te = fit_scaler_transform(df_a1)
Xs_tr, ys_tr, _ = make_seq(X_tr, y_tr_full)
Xs_v, ys_v, _ = make_seq(X_v, y_val_full)
Xs_te, ys_te, _ = make_seq(X_te, y_test_full)

LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
HIDDENS = [32, 48, 64, 96, 128]
grid_a1 = []
t_grid = time.time()
for h in HIDDENS:
    for lr in LRS:
        t0 = time.time()
        m, bv, n_ep = train_one(Xs_tr, ys_tr, Xs_v, ys_v, hidden=h, lr=lr, seed=42)
        ev_t = evaluate(predict_sorted(m, Xs_te), ys_te, 'test')
        grid_a1.append({
            'lr': lr, 'hidden': h, 'best_val_pinball': round(bv, 4),
            'n_epochs': n_ep, 'elapsed_s': round(time.time()-t0, 1),
            'test_pinball_q50': round(ev_t['pinball_q50'], 4),
            'test_rmse_q50': round(ev_t['rmse_q50_bp'], 4),
            'test_coverage': round(ev_t['coverage_90'], 4),
            'test_dir_acc': round(ev_t['dir_acc_q50'], 4),
        })
        print(f'  h={h:>3d} lr={lr:.0e}  val={bv:.4f}  test_RMSE={ev_t["rmse_q50_bp"]:.3f}  ({time.time()-t0:.1f}s)')
ga = pd.DataFrame(grid_a1)
ga.to_csv(ROOT/'reports/grid_a1_9vars_w7.csv', index=False)
print(f'  total grid time: {(time.time()-t_grid)/60:.1f} min')
best_a1 = ga.nsmallest(1, 'best_val_pinball').iloc[0]
print(f"  A1' best HP (val pinball 기준): lr={best_a1['lr']}, hidden={int(best_a1['hidden'])}, val={best_a1['best_val_pinball']:.4f}")
print(f"    test RMSE={best_a1['test_rmse_q50']:.3f}, Cov={best_a1['test_coverage']:.3f}, Dir={best_a1['test_dir_acc']:.3f}")

# A0 best HP (lr=5e-4, h=128) 와 A1' best HP 동일 여부 점검
match_hp = (best_a1['lr']==BEST_LR) and (int(best_a1['hidden'])==BEST_HIDDEN)
a1_lr = best_a1['lr']; a1_h = int(best_a1['hidden'])
print(f"  A0 best HP (lr={BEST_LR}, h={BEST_HIDDEN}) == A1' best HP (lr={a1_lr}, h={a1_h})? {'YES' if match_hp else 'NO'}")

# ====================================================================
print(); print('='*78); print('통합 요약'); print('='*78)
# ====================================================================
print(f"\n  A2' (+Δkospi)   test mean  RMSE {test_a2['rmse_q50_bp'].mean():.3f}, Cov {test_a2['coverage_90'].mean():.3f}, Dir {test_a2['dir_acc_q50'].mean():.3f}")
print(f"  A3' (+Δkr_ppi)  test mean  RMSE {test_a3['rmse_q50_bp'].mean():.3f}, Cov {test_a3['coverage_90'].mean():.3f}, Dir {test_a3['dir_acc_q50'].mean():.3f}")
# A0 reference
a0_final = pd.read_csv(ROOT/'reports/lstm_a0_final_eval_w5.csv')
a0_test = a0_final[a0_final['split']=='test']
print(f"  A0 reference    test mean  RMSE {a0_test['rmse_q50_bp'].mean():.3f}, Cov {a0_test['coverage_90'].mean():.3f}, Dir {a0_test['dir_acc_q50'].mean():.3f}")
print(f"\n  DM test 분위수별 유의 (Bonf α*={0.05/9:.4f}): {n_sig}/9")
print(f"\n  A1' best HP (9 vars 전용 grid):  lr={best_a1['lr']}, hidden={int(best_a1['hidden'])}")
print(f"  A0 best HP (8 vars):              lr={BEST_LR}, hidden={BEST_HIDDEN}")
print(f"  HP 일관성: {'+' if match_hp else 'X (재검토 필요)'}")

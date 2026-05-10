"""
16_walkforward.py — Walk-forward 3-fold expanding window

Day 3 작업:
  3 fold 정의 (expanding window):
    fold 1: train 2010-2017, val 2018-2019, cal 2019, test 2020
    fold 2: train 2010-2019, val 2020-2020, cal 2020 후반, test 2021-2022
    fold 3: train 2010-2021, val 2022, cal 2022, test 2023-2025

  각 fold에서:
    - XGBoost 분위수 (15_xgb_grid_cqr 의 best params 재사용으로 시간 단축)
    - LSTM 분위수 (3 시드)
    - DM test (XGB vs Naive)
    - CQR 보정

  Pooled metrics: 3 fold test 예측 stack → 전체 dir_acc/coverage/DM
  per-fold: fold별 결과 + 평균 ± 표준편차

출력:
  reports/no_leak_v2/walkforward_xgb_v2.csv
  reports/no_leak_v2/walkforward_lstm_v2.csv
  reports/no_leak_v2/walkforward_dm_v2.csv
  reports/no_leak_v2/walkforward_summary_v2.md
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
from scipy.stats import t as t_dist
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10

LSTM_CFG = CONFIG['models']['lstm']
LOOKBACK = CONFIG['features']['lookback_window']

print('=' * 72)
print('16_walkforward.py — 3-fold expanding window (XGB + LSTM)')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# Fold 정의
# ─────────────────────────────────────────────────────────────────────────
FOLDS = [
    {
        'name': 'fold1',
        'train': ('2010-01-01', '2017-12-31'),
        'val':   ('2018-01-01', '2019-12-31'),
        'cal':   ('2019-07-01', '2019-12-31'),
        'test':  ('2020-01-01', '2020-12-31'),
    },
    {
        'name': 'fold2',
        'train': ('2010-01-01', '2019-12-31'),
        'val':   ('2020-01-01', '2020-12-31'),
        'cal':   ('2020-07-01', '2020-12-31'),
        'test':  ('2021-01-01', '2022-12-31'),
    },
    {
        'name': 'fold3',
        'train': ('2010-01-01', '2021-12-31'),
        'val':   ('2022-01-01', '2022-12-31'),
        'cal':   ('2022-07-01', '2022-12-31'),
        'test':  ('2023-01-01', '2025-12-31'),
    },
]

# ─────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()

# v2 raw 입력 (LSTM 용 — 14개)
RAW_INPUT_FOR_LSTM = [
    'kr_treasury_3y', 'kr_base_rate', 'us_treasury_10y', 'us_fed_funds',
    'us_breakeven_10y', 'vix', 'kospi', 'sp500', 'dxy',
    'spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1', 'delta_dxy_t1', 'crisis_dummy',
]
RAW_INPUT_FOR_LSTM = [c for c in RAW_INPUT_FOR_LSTM if c in df.columns]
print(f'[input] LSTM 입력 변수 {len(RAW_INPUT_FOR_LSTM)}개')

# XGBoost 는 lag/roll 다 사용 (162개)
XGB_FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']
print(f'[input] XGB feature {len(XGB_FEATURE_COLS)}개')


# ─────────────────────────────────────────────────────────────────────────
# 평가 함수
# ─────────────────────────────────────────────────────────────────────────
def pinball(y, p, q):
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def dir_acc(y, p):
    mask = (np.sign(p) != 0) & (np.sign(y) != 0)
    if mask.sum() == 0:
        return float('nan')
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
    if n < 10:
        return float('nan'), float('nan')
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


# ─────────────────────────────────────────────────────────────────────────
# XGBoost 학습 (best params 재사용)
# ─────────────────────────────────────────────────────────────────────────
# 15_xgb_grid_cqr 결과를 그대로 사용 (시간 단축)
XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}


def fit_xgb(q, X_tr, y_tr, X_val, y_val):
    p = XGB_BEST[q]
    m = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q,
        n_estimators=p['n_estimators'],
        max_depth=p['max_depth'],
        learning_rate=p['learning_rate'],
        early_stopping_rounds=50,
        verbosity=0, tree_method='hist', random_state=42,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


# ─────────────────────────────────────────────────────────────────────────
# LSTM 정의
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
    seqs, tgts = [], []
    for t in range(lookback - 1, len(X_arr)):
        if not valid[t]:
            continue
        win = X_arr[t - lookback + 1: t + 1]
        if np.isnan(win).any():
            continue
        seqs.append(win)
        tgts.append(y_arr[t])
    return np.stack(seqs), np.array(tgts, dtype=np.float32)


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


# ─────────────────────────────────────────────────────────────────────────
# 메인 walk-forward 루프
# ─────────────────────────────────────────────────────────────────────────
all_xgb_eval = []
all_lstm_eval = []
all_dm = []
pooled_test_xgb_q50 = []
pooled_test_lstm_q50 = {seed: [] for seed in [42, 123, 2024]}
pooled_test_y = []

for fold in FOLDS:
    name = fold['name']
    print(f'\n{"="*72}\n{name}: train {fold["train"]}, val {fold["val"]}, cal {fold["cal"]}, test {fold["test"]}\n{"="*72}')

    # split
    def sl(p): return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[XGB_FEATURE_COLS]
    X_cal_raw = sl('cal')[XGB_FEATURE_COLS]
    X_val_raw = sl('val')[XGB_FEATURE_COLS]
    X_te_raw = sl('test')[XGB_FEATURE_COLS]

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=XGB_FEATURE_COLS)
    X_tr = s(X_tr_raw)
    X_cal = s(X_cal_raw)
    X_val = s(X_val_raw)
    X_te = s(X_te_raw)

    y_tr = sl('train')['delta_y_bp']
    y_cal = sl('cal')['delta_y_bp']
    y_val = sl('val')['delta_y_bp']
    y_te = sl('test')['delta_y_bp']

    print(f'  shapes: tr={X_tr.shape} cal={X_cal.shape} val={X_val.shape} te={X_te.shape}')

    # XGBoost
    print('  [XGB] 학습 q05/q50/q95 ...')
    xgb_preds = {sp: {} for sp in ['cal', 'test']}
    for q in QUANTILES:
        m = fit_xgb(q, X_tr.values, y_tr.values, X_val.values, y_val.values)
        xgb_preds['cal'][q] = m.predict(X_cal.values)
        xgb_preds['test'][q] = m.predict(X_te.values)
    xgb_preds['cal'] = sort_qs(xgb_preds['cal'])
    xgb_preds['test'] = sort_qs(xgb_preds['test'])

    # CQR 보정 (cal conformity score)
    cal_q05 = xgb_preds['cal'][0.05]
    cal_q95 = xgb_preds['cal'][0.95]
    cal_y = y_cal.values
    sc = np.maximum(cal_q05 - cal_y, cal_y - cal_q95)
    n_c = len(sc)
    k = min(int(np.ceil((n_c + 1) * (1 - ALPHA))), n_c)
    Q_hat = float(np.sort(sc)[k - 1])
    print(f'  [XGB] CQR Q_hat = {Q_hat:.4f} bp (n_cal={n_c})')

    xgb_test_cqr = {
        0.05: xgb_preds['test'][0.05] - Q_hat,
        0.5:  xgb_preds['test'][0.5],
        0.95: xgb_preds['test'][0.95] + Q_hat,
    }

    # 평가 (test only)
    raw_e = eval_q(xgb_preds['test'], y_te.values, 'test')
    raw_e.update({'fold': name, 'stage': 'raw', 'model': 'XGB(q,v2)'})
    cqr_e = eval_q(xgb_test_cqr, y_te.values, 'test')
    cqr_e.update({'fold': name, 'stage': 'CQR', 'model': 'XGB(q,v2)'})
    all_xgb_eval.extend([raw_e, cqr_e])
    print(f'  [XGB] dir_acc test = {raw_e["dir_acc_q50"]:.4f}, cov raw {raw_e["coverage_90"]:.4f} → CQR {cqr_e["coverage_90"]:.4f}')

    # Pooled 누적
    pooled_test_xgb_q50.append(xgb_preds['test'][0.5])
    pooled_test_y.append(y_te.values)

    # LSTM (3 시드)
    print('  [LSTM] 시퀀스 생성·학습 (시드 3개)')
    X_tr_lstm = X_tr[RAW_INPUT_FOR_LSTM]
    X_val_lstm = X_val[RAW_INPUT_FOR_LSTM]
    X_te_lstm = X_te[RAW_INPUT_FOR_LSTM]

    Xs_tr, ys_tr = make_seq(X_tr_lstm, y_tr, LOOKBACK)
    Xs_val, ys_val = make_seq(X_val_lstm, y_val, LOOKBACK)
    Xs_te, ys_te = make_seq(X_te_lstm, y_te, LOOKBACK)
    print(f'  [LSTM] Xs_tr={Xs_tr.shape}, Xs_val={Xs_val.shape}, Xs_te={Xs_te.shape}')

    for seed in [42, 123, 2024]:
        m, bv, ne = train_lstm_one(seed, Xs_tr, ys_tr, Xs_val, ys_val)
        p_te = sort_qs(predict_lstm(m, Xs_te))
        e = eval_q(p_te, ys_te, 'test')
        e.update({'fold': name, 'seed': seed, 'n_epochs': ne,
                  'best_val_pinball': bv, 'model': 'LSTM(q,v2)'})
        all_lstm_eval.append(e)
        pooled_test_lstm_q50[seed].append(p_te[0.5])
        print(f'    seed={seed}  epochs={ne}  dir_acc test={e["dir_acc_q50"]:.4f}')

    # DM test (XGB raw vs Naive, fold별)
    err_xgb = (y_te.values - xgb_preds['test'][0.5]) ** 2
    err_naive = y_te.values ** 2
    dm, pv = dm_test_hln(err_xgb, err_naive, lag=6)
    all_dm.append({'fold': name, 'comparison': 'XGBv2_vs_Naive',
                   'mean_se_xgb': float(err_xgb.mean()),
                   'mean_se_naive': float(err_naive.mean()),
                   'DM_HLN': dm, 'p_value': pv,
                   'winner': ('XGB' if dm < 0 and pv < 0.0167 else
                              'OPP' if dm > 0 and pv < 0.0167 else 'tie')})
    print(f'  [DM] XGB vs Naive: DM={dm:.3f}, p={pv:.4f}')

# ─────────────────────────────────────────────────────────────────────────
# Pooled metrics (3 fold test 합쳐 한 번에)
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('Pooled metrics (3 fold test 합쳐 평가)')
print('=' * 72)

y_pool = np.concatenate(pooled_test_y)
xgb_q50_pool = np.concatenate(pooled_test_xgb_q50)
err_xgb_pool = (y_pool - xgb_q50_pool) ** 2
err_naive_pool = y_pool ** 2
dm_pool, p_pool = dm_test_hln(err_xgb_pool, err_naive_pool, lag=6)
print(f'  XGB pooled: dir_acc = {dir_acc(y_pool, xgb_q50_pool):.4f}, RMSE = {np.sqrt(err_xgb_pool.mean()):.3f} bp')
print(f'  Naive pooled: RMSE = {np.sqrt(err_naive_pool.mean()):.3f} bp')
print(f'  DM XGB vs Naive (pooled): DM={dm_pool:.3f}, p={p_pool:.4f}, Bonferroni α=0.0167 → {"OK" if p_pool<0.0167 else "NO"}')

all_dm.append({'fold': 'POOLED', 'comparison': 'XGBv2_vs_Naive',
               'mean_se_xgb': float(err_xgb_pool.mean()),
               'mean_se_naive': float(err_naive_pool.mean()),
               'DM_HLN': dm_pool, 'p_value': p_pool,
               'winner': ('XGB' if dm_pool < 0 and p_pool < 0.0167 else
                          'OPP' if dm_pool > 0 and p_pool < 0.0167 else 'tie')})

# LSTM pooled (시드 평균)
lstm_pool_dir = []
for seed in [42, 123, 2024]:
    p_pool = np.concatenate(pooled_test_lstm_q50[seed])
    lstm_pool_dir.append(dir_acc(y_pool, p_pool))
print(f'  LSTM pooled dir_acc (시드 평균): {np.mean(lstm_pool_dir):.4f} ± {np.std(lstm_pool_dir):.4f}')

# ─────────────────────────────────────────────────────────────────────────
# 저장 + 요약
# ─────────────────────────────────────────────────────────────────────────
xgb_df = pd.DataFrame(all_xgb_eval)
lstm_df = pd.DataFrame(all_lstm_eval)
dm_df = pd.DataFrame(all_dm)

xgb_df.to_csv(REPORT_DIR / 'walkforward_xgb_v2.csv', index=False)
lstm_df.to_csv(REPORT_DIR / 'walkforward_lstm_v2.csv', index=False)
dm_df.to_csv(REPORT_DIR / 'walkforward_dm_v2.csv', index=False)

# Per-fold 평균/표준편차 (XGB CQR)
xgb_cqr = xgb_df[xgb_df['stage'] == 'CQR']
print('\n=== XGB CQR Per-fold (test set) ===')
print(xgb_cqr[['fold', 'dir_acc_q50', 'coverage_90', 'sharpness_bp', 'rmse_q50_bp']].round(4).to_string(index=False))

mean_dir = xgb_cqr['dir_acc_q50'].mean()
std_dir = xgb_cqr['dir_acc_q50'].std()
mean_cov = xgb_cqr['coverage_90'].mean()
std_cov = xgb_cqr['coverage_90'].std()
print(f'\n  XGB CQR 평균: dir_acc={mean_dir:.4f}±{std_dir:.4f}, coverage={mean_cov:.4f}±{std_cov:.4f}')

# 마크다운 요약
md_lines = [
    '# Walk-forward 3-fold v2 결과',
    '',
    '## Fold 정의',
    '',
    '| fold | train | val | cal | test |',
    '|---|---|---|---|---|',
]
for f in FOLDS:
    md_lines.append(f"| {f['name']} | {f['train'][0]}~{f['train'][1]} | {f['val'][0]}~{f['val'][1]} | {f['cal'][0]}~{f['cal'][1]} | {f['test'][0]}~{f['test'][1]} |")

md_lines.extend([
    '',
    '## XGBoost CQR per-fold (test)',
    '',
    '| fold | dir_acc | Coverage 90% | Sharpness (bp) | RMSE (bp) |',
    '|---|---|---|---|---|',
])
for _, r in xgb_cqr.iterrows():
    md_lines.append(f"| {r['fold']} | {r['dir_acc_q50']:.4f} | {r['coverage_90']:.4f} | {r['sharpness_bp']:.3f} | {r['rmse_q50_bp']:.3f} |")

md_lines.extend([
    '',
    f'**평균**: dir_acc {mean_dir:.4f} ± {std_dir:.4f}, Coverage {mean_cov:.4f} ± {std_cov:.4f}',
    '',
    '## DM test per-fold + Pooled (XGB vs Naive, q50 SE)',
    '',
    '| fold | DM_HLN | p-value | Bonferroni α=0.0167 | winner |',
    '|---|---|---|---|---|',
])
for _, r in dm_df.iterrows():
    bonf = 'OK' if r['p_value'] < 0.0167 else 'NO'
    md_lines.append(f"| {r['fold']} | {r['DM_HLN']:.3f} | {r['p_value']:.4f} | {bonf} | {r['winner']} |")

md_lines.extend([
    '',
    '## 비교 — single-split (15_xgb_grid_cqr) vs walk-forward 3-fold',
    '',
    '| 지표 | single-split (test 2023~25) | 3-fold 평균 |',
    '|---|---|---|',
    f'| dir_acc | (참조: 15 결과) | {mean_dir:.4f} |',
    f'| Coverage | (참조: 15 결과) | {mean_cov:.4f} |',
    '',
    '## 결론',
    '',
])

if mean_dir >= 0.55 and mean_cov >= 0.85:
    md_lines.append('- ✅ 3 fold 평균 dir_acc ≥ 55%, Coverage ≥ 85% — 우리 목표 안정적 달성')
elif mean_dir >= 0.55:
    md_lines.append('- 🟡 dir_acc 목표 달성, Coverage 약간 부족 (CQR 강화 검토)')
else:
    md_lines.append('- 🔴 dir_acc 목표 미달 — 재검토 필요')

md_path = REPORT_DIR / 'walkforward_summary_v2.md'
md_path.write_text('\n'.join(md_lines), encoding='utf-8')
print(f'\n[save] {md_path.relative_to(PROJECT_ROOT)}')
print('\n=== Day 3 완료 ===')

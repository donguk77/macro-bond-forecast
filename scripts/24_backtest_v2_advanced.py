# -*- coding: utf-8 -*-
"""
24_backtest_v2_advanced.py — v2 백테스트 정확도 보강 (Level B + 전략 다양화)

추가 항목:
1. Convexity 2차항 (PnL = ±D×Δy ∓ ½C×Δy²)
2. Carry term (보유에 따른 수익률, ±y/252)
3. 거래비용 sensitivity (0, 0.5, 1, 2, 3 bp)
4. 5개 전략 비교:
   - S0: Buy-and-Hold (always long)
   - S1: Sign(q50) — 단순 방향
   - S2: Confidence filter — |q50|>τ 일 때만 매매
   - S3: Vol-targeted Sign(q50) — 변동성 역수 비례 포지션
   - S4: Dual quantile — q05>0 long 또는 q95<0 short, 그 외 cash
5. Sharpe ratio bootstrap 95% CI (1000 resamples)
6. Walk-forward 3-fold 재학습 + 백테스트 (fold1 코로나·fold2 인상기·fold3 안정+충격)

PnL 부호 convention:
- pos = sign(q50): +1 = predict yield UP → short bond
- Price PnL: pos × D × Δy_bp / 10000
  (yield up일 때 short 이득)
- Convexity PnL: -pos × (1/2) × C × (Δy_bp/10000)²
  (short은 convexity가 손해, long은 이득)
- Carry PnL: -pos × y / 252
  (long은 coupon 받음, short은 지급)
- Txn cost: position 변경 시 cost_bp × D / 10000
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
OUT = ROOT / 'reports' / 'no_leak_v2'
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────
# 파라미터
# ─────────────────────────────────────────────────────────────────
D = 8.0           # KR 10y modified duration (~3.5% yield)
C = 85.0          # KR 10y convexity (10y vanilla, ~3.5% yield)
COSTS_BP = [0.0, 0.5, 1.0, 2.0, 3.0]   # 거래비용 sensitivity
TARGET_VOL_ANNUAL = 0.05               # vol-targeted: 연 5% 변동성 목표
CONFIDENCE_THRESHOLD_BP = 1.0          # S2: |q50|>1bp 일 때만 매매
N_BOOTSTRAP = 1000                     # Sharpe CI

SPLIT_SINGLE = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}

# Walk-forward 3-fold (script 16 와 동일)
SPLIT_WF = {
    'fold1': {'train': ('2010-01-01', '2017-12-31'),
              'val':   ('2018-01-01', '2019-12-31'),
              'cal':   ('2019-07-01', '2019-12-31'),
              'test':  ('2020-01-01', '2020-12-31')},
    'fold2': {'train': ('2010-01-01', '2019-12-31'),
              'val':   ('2020-01-01', '2020-12-31'),
              'cal':   ('2020-07-01', '2020-12-31'),
              'test':  ('2021-01-01', '2022-12-31')},
    'fold3': {'train': ('2010-01-01', '2021-12-31'),
              'val':   ('2022-01-01', '2022-12-31'),
              'cal':   ('2022-07-01', '2022-12-31'),
              'test':  ('2023-01-01', '2025-12-31')},
}

# Best params from script 15 grid
BEST_PARAMS = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 87},
    0.50: {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 215},
    0.95: {'max_depth': 4, 'learning_rate': 0.10, 'n_estimators': 53},
}

QS = [0.05, 0.5, 0.95]

print('=' * 78)
print('24_backtest_v2_advanced.py — Level B + 전략 다양화')
print('=' * 78)

# ─────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
FEATURES = [c for c in df.columns if c != 'delta_y_bp']
print(f'\n[load] features shape={df.shape}, n_feat={len(FEATURES)}')

# kr_treasury_10y level 로드 (carry 계산용 — features에 lag 변환된 게 있어 raw 따로 로드)
raw_ecos = pd.read_csv(DATA / 'raw' / 'raw_ecos.csv', parse_dates=['date'])
y_level = (raw_ecos[raw_ecos['variable'] == 'kr_treasury_10y']
           .set_index('date')['value'].sort_index())
print(f'[load] kr_treasury_10y level: {len(y_level)} obs, '
      f'mean={y_level.mean():.3f}%, range=[{y_level.min():.2f}, {y_level.max():.2f}]%')


def slice_d(p, split):
    s, e = split[p]
    return df.loc[s:e]


# ─────────────────────────────────────────────────────────────────
# Helper: 학습된 v2 모델 로드 (single-split 만)
# ─────────────────────────────────────────────────────────────────
def load_v2_models():
    out = {}
    for q in QS:
        m = xgb.XGBRegressor()
        m.load_model(str(MODELS / f'xgb_v2_q{int(q*100):02d}.json'))
        out[q] = m
    return out


# ─────────────────────────────────────────────────────────────────
# Helper: walk-forward 별 모델 재학습
# ─────────────────────────────────────────────────────────────────
def train_xgb_quantile(X_tr, y_tr, X_val, y_val, q, params):
    m = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q,
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        verbosity=0,
        tree_method='hist',
        random_state=42,
    )
    m.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], verbose=False)
    return m


def fit_predict_fold(split):
    """Fit RobustScaler + XGB v2 (best params) on fold's train, predict on test."""
    X_tr_raw = slice_d('train', split)[FEATURES]
    X_val_raw = slice_d('val', split)[FEATURES]
    X_te_raw = slice_d('test', split)[FEATURES]
    y_tr = slice_d('train', split)['delta_y_bp']
    y_val = slice_d('val', split)['delta_y_bp']
    y_te = slice_d('test', split)['delta_y_bp']

    scaler = RobustScaler().fit(X_tr_raw)
    X_tr = pd.DataFrame(scaler.transform(X_tr_raw), index=X_tr_raw.index, columns=FEATURES)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), index=X_val_raw.index, columns=FEATURES)
    X_te = pd.DataFrame(scaler.transform(X_te_raw), index=X_te_raw.index, columns=FEATURES)

    preds = {}
    for q in QS:
        m = train_xgb_quantile(X_tr, y_tr, X_val, y_val, q, BEST_PARAMS[q])
        preds[q] = m.predict(X_te.values)

    arr = np.column_stack([preds[q] for q in QS])
    arr = np.sort(arr, axis=1)
    q05, q50, q95 = arr[:, 0], arr[:, 1], arr[:, 2]
    return X_te.index, y_te.values, q05, q50, q95


# ─────────────────────────────────────────────────────────────────
# 전략 정의
# ─────────────────────────────────────────────────────────────────
def strategy_position(name: str, q05: np.ndarray, q50: np.ndarray, q95: np.ndarray,
                       y_true: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """전략별 포지션 (+1=short bond, -1=long bond, 0=cash)"""
    if name == 'S0_BuyHold':
        return -np.ones(len(q50), dtype=float)  # always long
    if name == 'S1_SignQ50':
        return np.sign(q50).astype(float)
    if name == 'S2_ConfFilter':
        # |q50| > threshold 일 때만 매매
        pos = np.where(np.abs(q50) > CONFIDENCE_THRESHOLD_BP, np.sign(q50), 0).astype(float)
        return pos
    if name == 'S3_VolTarget':
        # vol-targeted: position size = target_vol / realized_vol, capped at 1.0
        # realized vol: 20일 rolling std of Δy_bp → daily price vol = D × σ_y / 10000
        s = pd.Series(y_true, index=dates)
        rolling_vol_bp = s.rolling(20).std().shift(1).bfill().values  # bp
        # daily price vol (decimals): D × σ_bp / 10000
        daily_price_vol = D * rolling_vol_bp / 10000
        target_daily_vol = TARGET_VOL_ANNUAL / np.sqrt(252)
        # pos magnitude = target / realized, capped
        size = np.clip(target_daily_vol / np.where(daily_price_vol > 0, daily_price_vol, 1e-6),
                       0.0, 1.0)
        return np.sign(q50) * size
    if name == 'S4_DualQuantile':
        # long if q05 > 0 (high confidence yield up = short)
        # short if q95 < 0 (high confidence yield down = long)
        # 사실: q05>0 → 95% 확신으로 yield up → SHORT bond → pos=+1
        #       q95<0 → 95% 확신으로 yield down → LONG bond → pos=-1
        pos = np.where(q05 > 0, +1.0, np.where(q95 < 0, -1.0, 0.0))
        return pos
    raise ValueError(f'Unknown strategy: {name}')


STRATEGIES = ['S0_BuyHold', 'S1_SignQ50', 'S2_ConfFilter', 'S3_VolTarget', 'S4_DualQuantile']


# ─────────────────────────────────────────────────────────────────
# PnL 계산 (Convexity + Carry 포함)
# ─────────────────────────────────────────────────────────────────
def compute_pnl(pos: np.ndarray, y_true_bp: np.ndarray, y_level_pct: np.ndarray,
                cost_bp: float = 1.0):
    """일별 PnL (decimal)"""
    dy = y_true_bp / 10000  # bp → decimal

    # 1) Price PnL (1차): pos × D × Δy
    pnl_price = pos * D * dy

    # 2) Convexity (2차): -pos × (1/2) × C × Δy²
    pnl_convex = -pos * 0.5 * C * dy ** 2

    # 3) Carry: -pos × y / 252 (long bond=받음, short=지급)
    pnl_carry = -pos * (y_level_pct / 100) / 252

    # 4) Transaction cost: 포지션 변경 시 cost_bp × D / 10000
    pos_change = np.abs(np.diff(pos, prepend=pos[0])) / 2  # 0 to 1 scale
    # vol-targeted 같이 0~1 사이값일 수도 있어서 |Δpos| 사용
    pos_change = np.abs(np.diff(pos, prepend=pos[0]))
    txn_cost = pos_change * cost_bp * D / 10000

    pnl_net = pnl_price + pnl_convex + pnl_carry - txn_cost

    return {
        'price': pnl_price,
        'convex': pnl_convex,
        'carry': pnl_carry,
        'cost': txn_cost,
        'net': pnl_net,
    }


# ─────────────────────────────────────────────────────────────────
# 성과 metric
# ─────────────────────────────────────────────────────────────────
def metrics(pnl: np.ndarray):
    cum = np.cumsum(pnl)
    total_ret = float(cum[-1]) * 100
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else float('nan')
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    mdd = float(dd.min()) * 100
    win = int((pnl > 0).sum())
    loss = int((pnl < 0).sum())
    n = win + loss
    win_rate = win / n * 100 if n else float('nan')
    return {'total_return_%': round(total_ret, 3), 'sharpe_252': round(sharpe, 3),
            'max_drawdown_%': round(mdd, 3), 'win_rate_%': round(win_rate, 3),
            'days_active': n}


def sharpe_bootstrap_ci(pnl: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pnl[idx]
        s = float(sample.mean() / sample.std() * np.sqrt(252)) if sample.std() > 0 else float('nan')
        sharpes[i] = s
    return float(np.quantile(sharpes, 0.025)), float(np.quantile(sharpes, 0.975))


# ─────────────────────────────────────────────────────────────────
# 1. Single-split 백테스트 (test 2023-2025)
# ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 78)
print('Section 1: Single-split (test 2023-2025) — Level B 보강')
print('=' * 78)

# v2 모델 로드 + 예측
m_v2 = load_v2_models()
X_te_raw = slice_d('test', SPLIT_SINGLE)[FEATURES]
X_tr_raw = slice_d('train', SPLIT_SINGLE)[FEATURES]
y_te = slice_d('test', SPLIT_SINGLE)['delta_y_bp']

scaler_single = RobustScaler().fit(X_tr_raw)
X_te = pd.DataFrame(scaler_single.transform(X_te_raw), index=X_te_raw.index, columns=FEATURES)

preds_single = {q: m_v2[q].predict(X_te.values) for q in QS}
arr = np.column_stack([preds_single[q] for q in QS])
arr = np.sort(arr, axis=1)
q05, q50, q95 = arr[:, 0], arr[:, 1], arr[:, 2]
y_arr = y_te.values
dates = X_te.index

# y_level 매칭 (test 구간)
y_lev = y_level.reindex(dates).ffill().values  # %

# 검증
mask = (np.sign(q50) != 0) & (np.sign(y_arr) != 0)
dir_acc = float((np.sign(q50[mask]) == np.sign(y_arr[mask])).mean())
print(f'\n[verify] test dir_acc = {dir_acc:.4f} (기대 0.6113)')
assert abs(dir_acc - 0.6113) < 0.005, 'dir_acc 불일치'

# 전략별 × 비용 sensitivity 테이블
results_main = []
pnl_storage = {}  # (strategy, cost) → pnl array

for strat in STRATEGIES:
    pos = strategy_position(strat, q05, q50, q95, y_arr, dates)
    for cost in COSTS_BP:
        pnl = compute_pnl(pos, y_arr, y_lev, cost_bp=cost)
        m = metrics(pnl['net'])
        # decompose
        m['strategy'] = strat
        m['cost_bp'] = cost
        m['avg_pos_abs'] = round(float(np.abs(pos).mean()), 3)
        m['n_pos_changes'] = int(np.sum(np.abs(np.diff(pos, prepend=pos[0])) > 0.01))
        m['carry_total_%'] = round(float(pnl['carry'].sum()) * 100, 3)
        m['convex_total_%'] = round(float(pnl['convex'].sum()) * 100, 3)
        m['cost_total_%'] = round(float(pnl['cost'].sum()) * 100, 3)
        results_main.append(m)
        pnl_storage[(strat, cost)] = pnl['net']

res_main = pd.DataFrame(results_main)
res_main = res_main[['strategy', 'cost_bp', 'total_return_%', 'sharpe_252', 'max_drawdown_%',
                     'win_rate_%', 'avg_pos_abs', 'n_pos_changes',
                     'carry_total_%', 'convex_total_%', 'cost_total_%', 'days_active']]
print(f'\n  전략별 × 거래비용 sensitivity ({len(STRATEGIES)} 전략 × {len(COSTS_BP)} cost = {len(res_main)} 행):')
print(res_main.to_string(index=False))

res_main.to_csv(OUT / 'backtest_v2_advanced.csv', index=False)
print(f'\n[save] reports/no_leak_v2/backtest_v2_advanced.csv')

# Sharpe CI bootstrap (cost=1.0 만)
print(f'\n  Sharpe 95% CI bootstrap (n={N_BOOTSTRAP}, cost=1.0bp):')
ci_results = []
for strat in STRATEGIES:
    pnl = pnl_storage[(strat, 1.0)]
    m = metrics(pnl)
    lo, hi = sharpe_bootstrap_ci(pnl)
    ci_results.append({
        'strategy': strat,
        'sharpe': m['sharpe_252'],
        'sharpe_CI_low': round(lo, 3),
        'sharpe_CI_high': round(hi, 3),
        'sharpe_CI_excludes_0': lo > 0,
    })
ci_df = pd.DataFrame(ci_results)
print(ci_df.to_string(index=False))
ci_df.to_csv(OUT / 'backtest_v2_sharpe_ci.csv', index=False)
print(f'[save] reports/no_leak_v2/backtest_v2_sharpe_ci.csv')


# ─────────────────────────────────────────────────────────────────
# 2. Walk-forward 3-fold backtest
# ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 78)
print('Section 2: Walk-forward 3-fold (재학습 + 백테스트, cost=1bp)')
print('=' * 78)

wf_results = []
wf_pnl = {}  # (fold, strat) → pnl

for fold_name, split in SPLIT_WF.items():
    print(f'\n  [{fold_name}] training ... ', end='', flush=True)
    dates_f, y_f, q05_f, q50_f, q95_f = fit_predict_fold(split)
    y_lev_f = y_level.reindex(dates_f).ffill().values

    mask_f = (np.sign(q50_f) != 0) & (np.sign(y_f) != 0)
    dir_f = float((np.sign(q50_f[mask_f]) == np.sign(y_f[mask_f])).mean())
    print(f'dir_acc={dir_f:.4f}, test_n={len(y_f)}')

    for strat in STRATEGIES:
        pos = strategy_position(strat, q05_f, q50_f, q95_f, y_f, dates_f)
        pnl = compute_pnl(pos, y_f, y_lev_f, cost_bp=1.0)
        m = metrics(pnl['net'])
        m['fold'] = fold_name
        m['strategy'] = strat
        m['dir_acc_q50'] = round(dir_f, 4)
        wf_results.append(m)
        wf_pnl[(fold_name, strat)] = pnl['net']

wf_df = pd.DataFrame(wf_results)
wf_df = wf_df[['fold', 'strategy', 'total_return_%', 'sharpe_252', 'max_drawdown_%',
               'win_rate_%', 'days_active', 'dir_acc_q50']]
print(f'\n  Walk-forward 결과:')
print(wf_df.to_string(index=False))

# Pooled backtest: 3 fold test 합쳐서 1개 시계열로
print(f'\n  Pooled (3 fold test 합산):')
for strat in STRATEGIES:
    pooled = np.concatenate([wf_pnl[(f, strat)] for f in SPLIT_WF])
    m = metrics(pooled)
    lo, hi = sharpe_bootstrap_ci(pooled, n_boot=N_BOOTSTRAP)
    wf_results.append({
        'fold': 'POOLED',
        'strategy': strat,
        **m,
        'dir_acc_q50': float('nan'),
    })
    print(f"    {strat:18s}: Total {m['total_return_%']:+6.2f}%, "
          f"Sharpe {m['sharpe_252']:>5.2f} [{lo:+.2f}, {hi:+.2f}], "
          f"MDD {m['max_drawdown_%']:>6.2f}%, Win {m['win_rate_%']:.1f}%")

wf_df_full = pd.DataFrame(wf_results)
wf_df_full = wf_df_full[['fold', 'strategy', 'total_return_%', 'sharpe_252',
                         'max_drawdown_%', 'win_rate_%', 'days_active', 'dir_acc_q50']]
wf_df_full.to_csv(OUT / 'backtest_v2_walkforward.csv', index=False)
print(f'\n[save] reports/no_leak_v2/backtest_v2_walkforward.csv')


# ─────────────────────────────────────────────────────────────────
# 3. 시각화
# ─────────────────────────────────────────────────────────────────
print('\n' + '=' * 78)
print('Section 3: 시각화')
print('=' * 78)

# Fig 1: Single-split 누적 수익 (cost=1bp, 5 전략 비교)
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
ax = axes[0]
colors = {'S0_BuyHold': 'steelblue', 'S1_SignQ50': 'crimson',
          'S2_ConfFilter': 'orange', 'S3_VolTarget': 'green',
          'S4_DualQuantile': 'purple'}
for strat in STRATEGIES:
    pnl = pnl_storage[(strat, 1.0)]
    cum = np.cumsum(pnl) * 100
    ax.plot(dates, cum, label=strat, color=colors[strat], linewidth=1.5 if strat != 'S0_BuyHold' else 1.0)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('누적 수익률 (%)')
ax.set_title(f'Single-split 백테스트 (test 2023-2025, cost=1bp, D={D}, C={C}, carry 포함)')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# Drawdown
ax = axes[1]
for strat in STRATEGIES:
    cum = np.cumsum(pnl_storage[(strat, 1.0)]) * 100
    rmax = np.maximum.accumulate(cum)
    dd = cum - rmax
    ax.fill_between(dates, dd, 0, alpha=0.4, color=colors[strat], label=strat)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = FIG / '07_backtest_v2_advanced.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'[save] {fig_path.relative_to(ROOT)}')

# Fig 2: Walk-forward fold 비교
fig, ax = plt.subplots(figsize=(12, 7))
fold_labels = list(SPLIT_WF.keys())
strat_idx = np.arange(len(STRATEGIES))
width = 0.25
for i, fold in enumerate(fold_labels):
    sharpes = [wf_df[(wf_df['fold'] == fold) & (wf_df['strategy'] == s)]['sharpe_252'].iloc[0]
               for s in STRATEGIES]
    ax.bar(strat_idx + i * width, sharpes, width,
           label=f'{fold} ({SPLIT_WF[fold]["test"][0][:7]}~{SPLIT_WF[fold]["test"][1][:7]})')
ax.set_xticks(strat_idx + width)
ax.set_xticklabels(STRATEGIES, rotation=15)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Sharpe ratio (252-day)')
ax.set_title('Walk-forward 3-fold Sharpe 비교 — fold별 안정성 검증')
ax.legend(loc='best')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
fig_path = FIG / '08_backtest_walkforward.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'[save] {fig_path.relative_to(ROOT)}')

# Fig 3: 비용 sensitivity (S1 SignQ50 만, cost ∈ COSTS_BP)
fig, ax = plt.subplots(figsize=(12, 6))
for cost in COSTS_BP:
    pnl = pnl_storage[('S1_SignQ50', cost)]
    cum = np.cumsum(pnl) * 100
    ax.plot(dates, cum, label=f'cost={cost}bp', linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('누적 수익률 (%)')
ax.set_title('S1 SignQ50 — 거래비용 sensitivity (single-split)')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = FIG / '09_backtest_cost_sensitivity.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'[save] {fig_path.relative_to(ROOT)}')

print('\n완료. 추가 분석은 reports/no_leak_v2/backtest_v2_advanced.csv 등 참조.')

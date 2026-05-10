# -*- coding: utf-8 -*-
"""
23_backtest_v2.py — v2 (XGBoost CQR no_leak) 기반 backtest + 거래비용

옵션 B 진행:
- v0/A0 시기 backtest (reports/backtest_w7.csv) 는 누수 영향 가능성 → v2 로 재실행
- 거래비용 1bp/포지션 변경 추가 (현실 가정)
- 동일 metric: total return, Sharpe(252), MDD, win rate

전략:
- Buy-and-Hold (always long bond)
- XGB v2 (sign q50, no cost)
- XGB v2 (sign q50, 1bp cost)  ← 메인 비교 대상
- Random ±1

PnL 모델 (기존 11_w7_backtest_case.py 와 동일 convention):
- Δy in bp → 일별 채권 수익률 = -D × Δy / 10000  (D=8 modified duration)
- sign(q50) > 0 (yield up 예측) → 포지션 = +1 (short bond) → PnL = +Δy × D / 10000
- 거래비용: 포지션 변경 시 1bp × D / 10000 차감

출력:
- reports/no_leak_v2/backtest_v2.csv (전략별 metric)
- reports/no_leak_v2/backtest_v2_predictions.csv (date + Δy + q05/q50/q95 + PnL)
- reports/no_leak_v2/figures/06_backtest_v2_cumret.png
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
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# 한글 폰트 (안 깔려 있으면 무시)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 파라미터
D = 8.0  # KR 10y modified duration (~3.5% yield)
TXN_COST_BP = 1.0  # 1bp per position change (round-trip 가정)
SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}
QS = [0.05, 0.5, 0.95]

print('=' * 72)
print('23_backtest_v2.py — v2 XGBoost CQR + transaction cost backtest')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────
# 1. 데이터 로드 + 분할 (script 15 와 동일)
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
print(f'\n[load] features shape = {df.shape}')

FEATURES = [c for c in df.columns if c != 'delta_y_bp']
print(f'[load] feature count = {len(FEATURES)}')


def slice_p(p):
    s, e = SPLIT[p]
    return df.loc[s:e]


X_train_raw = slice_p('train')[FEATURES]
X_test_raw  = slice_p('test')[FEATURES]
y_test      = slice_p('test')['delta_y_bp']

scaler = RobustScaler().fit(X_train_raw)
X_test = pd.DataFrame(scaler.transform(X_test_raw),
                      index=X_test_raw.index, columns=FEATURES)

print(f'[split] test = {X_test.shape}, range {X_test.index.min().date()} ~ {X_test.index.max().date()}')

# ─────────────────────────────────────────────────────────────────
# 2. v2 XGB 모델 로드 + test 예측
# ─────────────────────────────────────────────────────────────────
def load_xgb(path):
    m = xgb.XGBRegressor()
    m.load_model(str(path))
    return m


m_q05 = load_xgb(MODELS / 'xgb_v2_q05.json')
m_q50 = load_xgb(MODELS / 'xgb_v2_q50.json')
m_q95 = load_xgb(MODELS / 'xgb_v2_q95.json')

q05 = m_q05.predict(X_test.values)
q50 = m_q50.predict(X_test.values)
q95 = m_q95.predict(X_test.values)

# Sort 후처리 (monotonicity)
arr = np.column_stack([q05, q50, q95])
arr = np.sort(arr, axis=1)
q05, q50, q95 = arr[:, 0], arr[:, 1], arr[:, 2]

# 검증: dir_acc 가 기존 결과 (0.6113) 와 일치하는가?
y_arr = y_test.values
mask = (np.sign(q50) != 0) & (np.sign(y_arr) != 0)
dir_acc = float((np.sign(q50[mask]) == np.sign(y_arr[mask])).mean())
print(f'\n[verify] test dir_acc = {dir_acc:.4f} (기대값 0.6113)')

if abs(dir_acc - 0.6113) > 0.005:
    print(f'  ⚠ dir_acc 불일치: 기대 0.6113 vs 실제 {dir_acc:.4f}')
    print(f'  → 모델 또는 scaler 차이 가능성. 그대로 진행하지만 결과 검토 필요.')
else:
    print(f'  ✓ 기존 결과와 일치 — 모델·scaler 재현 OK')

# ─────────────────────────────────────────────────────────────────
# 3. Backtest
# ─────────────────────────────────────────────────────────────────
print(f'\n=== Backtest ===')
print(f'  Modified duration: D = {D}')
print(f'  Transaction cost: {TXN_COST_BP} bp / 포지션 변경')
print(f'  Test 구간: {len(y_test)} 영업일')

bt = pd.DataFrame({
    'date': X_test.index,
    'y_true_bp': y_arr,
    'q05': q05, 'q50': q50, 'q95': q95,
}).reset_index(drop=True)

# 전략별 일별 수익률
bt['pnl_buyhold'] = -D * bt['y_true_bp'] / 10000  # always long
bt['position_xgb_v2'] = np.sign(bt['q50']).astype(int)
bt['pnl_xgb_v2_gross'] = bt['position_xgb_v2'] * bt['y_true_bp'] * D / 10000

# 거래비용: 포지션 변경 시점에만 차감
bt['position_change'] = bt['position_xgb_v2'].diff().fillna(0).abs() / 2  # 0 or 1
bt['txn_cost'] = bt['position_change'] * TXN_COST_BP * D / 10000
bt['pnl_xgb_v2_net'] = bt['pnl_xgb_v2_gross'] - bt['txn_cost']

# Random ±1
np.random.seed(42)
random_sign = np.random.choice([-1, 1], size=len(bt))
bt['pnl_random'] = random_sign * bt['y_true_bp'] * D / 10000


def summarize(pnl: pd.Series, name: str):
    cum = pnl.cumsum()
    total_ret = float(cum.iloc[-1]) * 100  # %
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else float('nan')
    rmax = cum.cummax()
    dd = cum - rmax
    mdd = float(dd.min()) * 100
    win = int((pnl > 0).sum())
    loss = int((pnl < 0).sum())
    n = win + loss
    win_rate = win / n * 100 if n else float('nan')
    return {
        'strategy': name,
        'total_return_%': round(total_ret, 3),
        'sharpe_252': round(sharpe, 3),
        'max_drawdown_%': round(mdd, 3),
        'win_rate_%': round(win_rate, 3),
        'days_active': n,
    }


results = [
    summarize(bt['pnl_buyhold'], 'Buy-and-Hold (always long)'),
    summarize(bt['pnl_xgb_v2_gross'], 'XGB v2 (sign q50, no cost)'),
    summarize(bt['pnl_xgb_v2_net'], 'XGB v2 (sign q50, 1bp cost)'),
    summarize(bt['pnl_random'], 'Random ±1'),
]
res_df = pd.DataFrame(results)
print(f'\n  Backtest 결과 (test 2023-2025, {len(bt)} 영업일):')
print(res_df.to_string(index=False))

# 거래 통계
n_changes = int(bt['position_change'].sum())
total_cost_bp = float(bt['txn_cost'].sum() * 10000 / D)
print(f'\n  포지션 변경: {n_changes} / {len(bt)} ({n_changes/len(bt)*100:.1f}%)')
print(f'  누적 거래비용: {total_cost_bp:.1f} bp ({n_changes} × {TXN_COST_BP} bp)')

# 저장
res_df.to_csv(OUT / 'backtest_v2.csv', index=False)
bt.to_csv(OUT / 'backtest_v2_predictions.csv', index=False)
print(f'\n[save] reports/no_leak_v2/backtest_v2.csv')
print(f'[save] reports/no_leak_v2/backtest_v2_predictions.csv')

# ─────────────────────────────────────────────────────────────────
# 4. 시각화 (누적 수익 + Drawdown)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# (a) 누적 수익률
ax = axes[0]
strategies = [
    ('pnl_buyhold', 'Buy-and-Hold (always long)', 'steelblue'),
    ('pnl_xgb_v2_gross', 'XGB v2 (sign q50, no cost)', 'crimson'),
    ('pnl_xgb_v2_net', 'XGB v2 (sign q50, 1bp cost)', 'darkred'),
    ('pnl_random', 'Random ±1', 'gray'),
]
for col, label, color in strategies:
    cum = bt[col].cumsum() * 100
    lw = 1.8 if 'xgb_v2_net' in col else 1.0
    ax.plot(bt['date'], cum, label=label, color=color, linewidth=lw)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('누적 수익률 (%)')
ax.set_title(f'v2 Backtest 누적 수익 — KR 10y D={D}, 거래비용 {TXN_COST_BP}bp/변경, test 2023-2025')
ax.legend(loc='best')
ax.grid(alpha=0.3)

# A0 vs v2 비교 텍스트
v2_net = res_df[res_df['strategy'].str.contains('1bp')].iloc[0]
ax.text(0.02, 0.98,
        f"v2 net (1bp cost):\n  Total {v2_net['total_return_%']:+.2f}%\n  Sharpe {v2_net['sharpe_252']:.2f}\n  MDD {v2_net['max_drawdown_%']:.2f}%",
        transform=ax.transAxes, fontsize=10, va='top', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='darkred', linewidth=1.5))

# (b) Drawdown
ax = axes[1]
for col, label, color in strategies:
    cum = bt[col].cumsum() * 100
    rmax = cum.cummax()
    dd = cum - rmax
    ax.fill_between(bt['date'], dd, 0, alpha=0.4, color=color, label=label)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdown — peak 대비 하락')
ax.legend(loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = FIG / '06_backtest_v2_cumret.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'[save] {fig_path.relative_to(ROOT)}')

# ─────────────────────────────────────────────────────────────────
# 5. 요약 비교 (v2 net vs A0 LSTM)
# ─────────────────────────────────────────────────────────────────
print(f'\n' + '=' * 72)
print(f'A0 LSTM (v0 시기, 누수 의심) vs v2 XGB (정직, 1bp cost) 비교')
print('=' * 72)

# 기존 A0 결과 로드
a0_path = ROOT / 'reports' / 'backtest_w7.csv'
if a0_path.exists():
    a0_df = pd.read_csv(a0_path)
    a0_lstm = a0_df[a0_df['strategy'].str.contains('A0 LSTM')].iloc[0]
    print(f"\n  기존 A0 LSTM (no cost, 누수 의심):")
    print(f"    Total {a0_lstm['total_return_%']:+.2f}% | Sharpe {a0_lstm['sharpe_252']:.2f} | "
          f"MDD {a0_lstm['max_drawdown_%']:.2f}% | Win {a0_lstm['win_rate_%']:.1f}%")
    print(f"\n  신규 v2 XGB (1bp cost, 정직):")
    print(f"    Total {v2_net['total_return_%']:+.2f}% | Sharpe {v2_net['sharpe_252']:.2f} | "
          f"MDD {v2_net['max_drawdown_%']:.2f}% | Win {v2_net['win_rate_%']:.1f}%")
    print(f"\n  → v2 가 더 보수적이지만 ALL strategies 통계 우위 (DM=-8.78 Pooled) 와 일관.")
else:
    print(f"  [warn] {a0_path} 없음")

print(f'\n완료. 발표 자료에 v2 backtest 결과 인용 가능.')

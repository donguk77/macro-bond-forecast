# -*- coding: utf-8 -*-
"""
44 — 2026 라이브 OOS 체결 현실 시각화 (슬12 보강용)
슬9(40_viz_execution_cost.py)와 동일 규칙을 2026 라이브에 적용:
  pos = -sign(q50), 시초가 진입→당일 종가 청산(o2c), 종가-종가(c2c, 이론), B&H 비교.
출력: reports/figures/live_2026_execution.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / 'reports' / 'no_leak_v2'
FIG = ROOT / 'reports' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

sig = pd.read_csv(REP / 'live_oos_2026_xgb.csv', parse_dates=['date'])[['date', 'q50']]
etf = pd.read_csv(ROOT / 'data/raw/kosef_10y_daily_2023_2026.csv', parse_dates=['date']).sort_values('date')
etf['c2c'] = etf['close'] / etf['close'].shift(1) - 1
etf['o2c'] = etf['close'] / etf['open'] - 1
m = etf.merge(sig, on='date', how='inner').dropna(subset=['q50', 'o2c', 'c2c']).reset_index(drop=True)

pos = -np.sign(m['q50'].values)
dpos = np.abs(np.diff(pos, prepend=pos[0]))
g_o2c = pos * m['o2c'].values
g_c2c = pos * m['c2c'].values

c2c_cum = np.cumsum(g_c2c) * 100
o2c0_cum = np.cumsum(g_o2c) * 100
o2c1_cum = np.cumsum(g_o2c - dpos * 1.0 / 10000) * 100
bh_cum = np.cumsum(m['c2c'].values) * 100
d = m['date']

fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.55, 1]})

# ── 좌: 누적수익 곡선 ──
ax[0].axhline(0, color='black', lw=0.8, alpha=0.4)
ax[0].plot(d, c2c_cum, color='gray', ls='--', lw=1.8, label=f'종가체결 (이론·갭 포함)   {c2c_cum[-1]:+.1f}%')
ax[0].plot(d, o2c0_cum, color='royalblue', lw=1.9, label=f'시초가체결 (현실·비용0)   {o2c0_cum[-1]:+.1f}%')
ax[0].plot(d, o2c1_cum, color='crimson', lw=2.4, label=f'시초가체결 + 1bp 비용 (현실)   {o2c1_cum[-1]:+.1f}%')
ax[0].plot(d, bh_cum, color='seagreen', lw=1.9, ls=':', label=f'Buy & Hold (ETF 보유)   {bh_cum[-1]:+.1f}%')
# 갭 영역 강조
ax[0].fill_between(d, o2c0_cum, c2c_cum, color='orange', alpha=0.12)
ax[0].annotate('이 격차 = 밤사이 갭\n(현실에선 못 먹음)',
               xy=(d.iloc[int(len(d) * 0.62)], (c2c_cum[int(len(d) * 0.62)] + o2c0_cum[int(len(d) * 0.62)]) / 2),
               xytext=(d.iloc[int(len(d) * 0.18)], c2c_cum.max() * 0.78),
               fontsize=11, color='darkorange', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))
ax[0].set_title('2026 라이브 OOS — 체결 시점별 누적수익', fontsize=14, fontweight='bold')
ax[0].set_ylabel('누적수익 (%)', fontsize=12)
ax[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
ax[0].grid(alpha=0.25)

# ── 우: 최종 누적수익 막대 ──
labels = ['종가이론\n(갭 포함)', '시초가\n현실(1bp)', 'Buy &\nHold']
vals = [c2c_cum[-1], o2c1_cum[-1], bh_cum[-1]]
colors = ['gray', 'crimson', 'seagreen']
bars = ax[1].bar(labels, vals, color=colors, alpha=0.85, width=0.6)
ax[1].axhline(0, color='black', lw=0.9)
for b, v in zip(bars, vals):
    ax[1].text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -0.9),
               f'{v:+.1f}%', ha='center', fontsize=12, fontweight='bold',
               color='black' if v >= 0 else 'crimson')
ax[1].set_title('최종 누적수익 비교', fontsize=14, fontweight='bold')
ax[1].set_ylabel('누적수익 (%)', fontsize=12)
ax[1].grid(axis='y', alpha=0.25)

fig.suptitle('엣지는 전부 밤사이 갭 — 현실 체결로는 손실 (2026 라이브, 93거래일)',
             fontsize=15, fontweight='bold', y=1.00)
fig.tight_layout()
out = FIG / 'live_2026_execution.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
print('Saved', out.relative_to(ROOT))
print(f'종가이론 {c2c_cum[-1]:+.1f}% / 시초가0 {o2c0_cum[-1]:+.1f}% / 시초가1bp {o2c1_cum[-1]:+.1f}% / B&H {bh_cum[-1]:+.1f}%')

# -*- coding: utf-8 -*-
"""
40 — 체결 시점·거래비용의 현실 시각화 (기말 정직성 핵심 그림, v2)

정직한 전체 그림:
- (좌) 누적수익 곡선: 종가체결(이론)·시초가무비용·시초가+현실비용(1bp 틱수준)·B&H
- (우) 비용 민감도: 편도 스프레드별 최종 누적수익 + 손익분기(3.7bp) + 틱수준 구간 + CS평균(부풀림)

핵심: 엣지 상당부분은 overnight 갭(종가체결만 잡힘)·비현실. 시초가(현실) 장중 엣지는 실재하며
      현실 틱수준 비용(~0.5 price-bp)에선 +25%로 양수, 단 손익분기 3.7bp 로 마진 얇음.
입력: predictions_xgb_v3_intervals.csv + live_oos_2026_xgb.csv + kosef_10y_daily_2023_2026.csv
출력: reports/figures/execution_cost_reality.png
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
REALISTIC_BP = 1.0   # 좌측 곡선용 현실 비용(틱수준, price-bp 편도)

iv = pd.read_csv(REP / 'predictions_xgb_v3_intervals.csv', parse_dates=['date'])
f3 = iv[iv['fold'] == 'fold3'][['date', 'q50']]
sig = f3.sort_values('date')   # 2023-2025 (라이브 2026 제외 — 슬12에서만 사용)
etf = pd.read_csv(ROOT / 'data/raw/kosef_10y_daily_2023_2026.csv', parse_dates=['date']).sort_values('date')
etf['c2c'] = etf['close'] / etf['close'].shift(1) - 1
etf['o2c'] = etf['close'] / etf['open'] - 1
m = etf.merge(sig, on='date', how='inner').dropna().reset_index(drop=True)

pos = -np.sign(m['q50'].values)
dpos = np.abs(np.diff(pos, prepend=pos[0]))
sum_dpos = dpos.sum()
g_o2c = pos * m['o2c'].values
g_c2c = pos * m['c2c'].values
breakeven = g_o2c.sum() / sum_dpos * 10000   # price-bp 편도

c2c0 = np.cumsum(g_c2c) * 100
o2c0 = np.cumsum(g_o2c) * 100
o2c_r = np.cumsum(g_o2c - dpos * REALISTIC_BP / 10000) * 100
bh = np.cumsum(m['c2c'].values) * 100
d = m['date']

fig, ax = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.4, 1]})

# 좌: 누적 곡선 (현실 비용 1bp)
ax[0].plot(d, c2c0, color='gray', ls='--', lw=1.7, label=f'종가체결(이론, 갭 포함)  {c2c0[-1]:+.0f}%')
ax[0].plot(d, o2c0, color='royalblue', lw=1.9, label=f'시초가체결(현실)·비용 0  {o2c0[-1]:+.0f}%')
ax[0].plot(d, o2c_r, color='crimson', lw=2.3, label=f'시초가체결 + 현실비용 {REALISTIC_BP:.0f}bp(틱수준)  {o2c_r[-1]:+.0f}%')
ax[0].plot(d, bh, color='black', lw=1.0, alpha=0.5, label=f'Buy & Hold  {bh[-1]:+.0f}%')
ax[0].axhline(0, color='black', lw=0.6)
ax[0].set_title('누적 수익률 — 체결 가정·현실 비용(틱수준 1bp)', fontsize=13, fontweight='bold')
ax[0].set_ylabel('누적 수익률 (%)')
ax[0].legend(loc='upper left', fontsize=9.5)
ax[0].grid(alpha=0.3)

# 우: 비용 민감도 (편도 스프레드별 최종 수익)
cs_grid = np.linspace(0, 6, 121)
finals = [(g_o2c - dpos * c / 10000).sum() * 100 for c in cs_grid]
ax[1].axhline(0, color='black', lw=0.8)
# 구간 음영: 틱수준(현실) vs 손익분기 위(손실)
ax[1].axvspan(0.24, 1.5, color='seagreen', alpha=0.12, label='현실 틱수준 (~0.25~1.5bp)')
ax[1].axvspan(breakeven, 6, color='crimson', alpha=0.07)
ax[1].plot(cs_grid, finals, color='navy', lw=2.4)
ax[1].axvline(breakeven, color='black', ls=':', lw=1.3)
ax[1].text(breakeven + 0.12, max(finals) * 0.5, f'손익분기\n{breakeven:.1f}bp', fontsize=9, va='center')
# 마커
for c in [0.5, 1.0]:
    y = (g_o2c - dpos * c / 10000).sum() * 100
    ax[1].plot(c, y, 'o', color='seagreen', ms=7)
    ax[1].text(c, y + 2.5, f'{c}bp\n{y:+.0f}%', ha='center', fontsize=8.5, color='seagreen', fontweight='bold')
ax[1].set_title('비용 민감도 — 편도 스프레드별 최종 수익(시초가)', fontsize=13, fontweight='bold')
ax[1].set_xlabel('편도 거래비용 (price-bp,  1bp = 0.01%)')
ax[1].set_ylabel('전체기간 누적 수익률 (%)')
ax[1].legend(loc='lower left', fontsize=9)
ax[1].grid(alpha=0.3)
ax[1].set_xlim(0, 6)

plt.tight_layout()
out = ROOT / 'reports' / 'figures' / 'execution_cost_reality.png'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f'[save] {out}')
print(f'매매 {int((dpos>0).sum())}회 · Σdpos {int(sum_dpos)} · 손익분기 편도 {breakeven:.2f} price-bp (={breakeven/8:.3f} yield-bp)')
print(f'종가이론 {c2c0[-1]:+.0f}% / 시초가0 {o2c0[-1]:+.0f}% / 시초가0.5bp {(g_o2c-dpos*0.5/10000).sum()*100:+.0f}% / '
      f'시초가1bp {o2c_r[-1]:+.0f}% / B&H {bh[-1]:+.0f}%')

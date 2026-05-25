# -*- coding: utf-8 -*-
"""
트레이딩 시그널 시각화 — S2 Confidence Filter 전략
==================================================
실제 한국 10년 국고채 금리 차트 위에 매수/매도 시그널 + 누적 수익률 표시.

출력:
  reports/no_leak_v2/figures/10_trading_signals.png
  reports/no_leak_v2/figures/11_trading_signals_zoom.png (최근 6개월 확대)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

D = 8.0
C = 85.0
CONF_THRESHOLD = 1.0  # |q50| > 1bp

# ── 데이터 로드 ──
pred = pd.read_csv(ROOT / 'reports/no_leak_v2/backtest_v2_predictions.csv',
                   parse_dates=['date'])

raw_ecos = pd.read_csv(ROOT / 'data/raw/raw_ecos.csv', parse_dates=['date'])
kr10y = (raw_ecos[raw_ecos['variable'] == 'kr_treasury_10y']
         [['date', 'value']].rename(columns={'value': 'yield_10y'}))

df = pred.merge(kr10y, on='date', how='left')

# ── S2 포지션 계산 (pos=-1: Long, pos=+1: Short, 0: 관망) ──
df['s2_pos'] = np.where(np.abs(df['q50']) > CONF_THRESHOLD,
                        np.sign(df['q50']), 0).astype(float)

# ── PnL 계산 (24_backtest_v2_advanced.py 공식 동일) ──
dy = df['y_true_bp'].values / 10000
s2_pos = df['s2_pos'].values
y_level = df['yield_10y'].values

# price + convexity + carry - cost
s2_price = s2_pos * D * dy
s2_convex = -s2_pos * 0.5 * C * dy ** 2
s2_carry = -s2_pos * (y_level / 100) / 252
s2_pos_change = np.abs(np.diff(s2_pos, prepend=s2_pos[0]))
s2_txn = s2_pos_change * 1.0 * D / 10000
df['s2_pnl'] = s2_price + s2_convex + s2_carry - s2_txn
df['s2_cumret'] = df['s2_pnl'].cumsum() * 100  # %

# Buy-and-Hold (pos=-1 항상 Long)
bh_pos = -np.ones(len(df))
bh_price = bh_pos * D * dy
bh_convex = -bh_pos * 0.5 * C * dy ** 2
bh_carry = -bh_pos * (y_level / 100) / 252
df['bh_pnl'] = bh_price + bh_convex + bh_carry
df['bh_cumret'] = df['bh_pnl'].cumsum() * 100

# ── 시그널 전환 포인트 감지 ──
df['prev_pos'] = df['s2_pos'].shift(1).fillna(0)
df['signal_change'] = df['s2_pos'] != df['prev_pos']

# pos=-1 → Long 진입 (금리 하락 예측 → 채권 매수)
# pos=+1 → Short 진입 (금리 상승 예측 → 채권 매도)
buy_signals = df[(df['signal_change']) & (df['s2_pos'] == -1)]
sell_signals = df[(df['signal_change']) & (df['s2_pos'] == 1)]
exit_signals = df[(df['signal_change']) & (df['s2_pos'] == 0)]


def plot_trading_chart(data, title_suffix='', filename='10_trading_signals.png'):
    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                             gridspec_kw={'height_ratios': [3, 1, 2]},
                             sharex=True)
    fig.subplots_adjust(hspace=0.08)

    dates = data['date']

    # ── Panel 1: 금리 차트 + 시그널 ──
    ax1 = axes[0]

    # 포지션 배경색 (pos=-1: Long=파랑, pos=+1: Short=빨강)
    for i in range(len(data) - 1):
        pos = data['s2_pos'].iloc[i]
        if pos == -1:
            ax1.axvspan(dates.iloc[i], dates.iloc[i + 1],
                        alpha=0.15, color='#2196F3', linewidth=0)
        elif pos == 1:
            ax1.axvspan(dates.iloc[i], dates.iloc[i + 1],
                        alpha=0.15, color='#F44336', linewidth=0)

    ax1.plot(dates, data['yield_10y'], color='#333333', linewidth=1.0, zorder=3)

    # 매수/매도 시그널 마커
    buy_in = data[(data['signal_change']) & (data['s2_pos'] == -1)]
    sell_in = data[(data['signal_change']) & (data['s2_pos'] == 1)]
    exit_in = data[(data['signal_change']) & (data['s2_pos'] == 0)]

    ax1.scatter(buy_in['date'], buy_in['yield_10y'],
                marker='^', s=80, color='#2196F3', edgecolors='navy',
                linewidths=0.8, zorder=5, label='Long 진입')
    ax1.scatter(sell_in['date'], sell_in['yield_10y'],
                marker='v', s=80, color='#F44336', edgecolors='darkred',
                linewidths=0.8, zorder=5, label='Short 진입')
    ax1.scatter(exit_in['date'], exit_in['yield_10y'],
                marker='x', s=50, color='#9E9E9E',
                linewidths=1.5, zorder=5, label='포지션 청산')

    ax1.set_ylabel('한국 10년 국채 금리 (%)', fontsize=11)
    ax1.set_title(f'S2 Confidence Filter 트레이딩 시그널{title_suffix}',
                  fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)

    # ── Panel 2: 포지션 상태 ──
    ax2 = axes[1]
    ax2.fill_between(dates, data['s2_pos'], step='post', alpha=0.6,
                     where=data['s2_pos'] < 0, color='#2196F3')
    ax2.fill_between(dates, data['s2_pos'], step='post', alpha=0.6,
                     where=data['s2_pos'] > 0, color='#F44336')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_ylabel('포지션', fontsize=11)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['Long', '관망', 'Short'], fontsize=9)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(axis='y', alpha=0.3)

    # ── Panel 3: 누적 수익률 ──
    ax3 = axes[2]
    ax3.plot(dates, data['bh_cumret'], color='steelblue',
             linewidth=1.2, linestyle='--', label='S0: Buy-and-Hold')
    ax3.plot(dates, data['s2_cumret'], color='#FF6F00',
             linewidth=1.8, label='S2: Confidence Filter (1bp 비용)')
    ax3.fill_between(dates, data['s2_cumret'], alpha=0.15, color='#FF6F00')
    ax3.axhline(0, color='gray', linewidth=0.5)
    ax3.set_ylabel('누적 수익률 (%)', fontsize=11)
    ax3.set_xlabel('날짜', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.grid(axis='y', alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 통계 요약 박스
    total_trades = int(data['signal_change'].sum())
    long_days = int((data['s2_pos'] == -1).sum())
    short_days = int((data['s2_pos'] == 1).sum())
    neutral_days = int((data['s2_pos'] == 0).sum())
    final_ret = data['s2_cumret'].iloc[-1]
    bh_ret = data['bh_cumret'].iloc[-1]

    stats_text = (
        f'총 시그널 전환: {total_trades}회\n'
        f'Long: {long_days}일 | Short: {short_days}일 | 관망: {neutral_days}일\n'
        f'S2 수익률: {final_ret:+.1f}%  |  B&H: {bh_ret:+.1f}%'
    )
    ax3.text(0.98, 0.95, stats_text, transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    out_path = ROOT / 'reports/no_leak_v2/figures' / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'[SAVED] {out_path}')


# ── 전체 기간 차트 ──
plot_trading_chart(df, title_suffix=' (2023-01 ~ 2025-12, Test 전체)',
                   filename='10_trading_signals.png')

# ── 최근 6개월 확대 차트 ──
cutoff = df['date'].max() - pd.Timedelta(days=180)
df_zoom = df[df['date'] >= cutoff].copy()
df_zoom['s2_cumret'] = df_zoom['s2_pnl'].cumsum() * 100
df_zoom['bh_cumret'] = df_zoom['bh_pnl'].cumsum() * 100
df_zoom['prev_pos'] = df_zoom['s2_pos'].shift(1).fillna(df_zoom['s2_pos'].iloc[0])
df_zoom['signal_change'] = df_zoom['s2_pos'] != df_zoom['prev_pos']

plot_trading_chart(df_zoom, title_suffix=' (최근 6개월 확대)',
                   filename='11_trading_signals_zoom.png')

# ── 예측 구간 + 시그널 차트 (추가) ──
fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                         gridspec_kw={'height_ratios': [2, 1]},
                         sharex=True)
fig.subplots_adjust(hspace=0.08)

ax1 = axes[0]
ax1.fill_between(df['date'], df['q05'], df['q95'],
                 alpha=0.2, color='steelblue', label='90% 예측구간 (q05~q95)')
ax1.plot(df['date'], df['q50'], color='steelblue', linewidth=1.0,
         label='q50 (중앙값 예측)')
ax1.scatter(df['date'], df['y_true_bp'], s=3, color='black',
            alpha=0.4, label='실제 Δy', zorder=4)

# 1bp 임계선
ax1.axhline(CONF_THRESHOLD, color='orange', linewidth=0.8,
            linestyle='--', alpha=0.7)
ax1.axhline(-CONF_THRESHOLD, color='orange', linewidth=0.8,
            linestyle='--', alpha=0.7, label='±1bp 임계선')
ax1.axhline(0, color='gray', linewidth=0.5)

ax1.set_ylabel('Δy (bp)', fontsize=11)
ax1.set_title('XGBoost v2 분위수 예측 + S2 Confidence Filter 임계선',
              fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: 포지션 + 누적 수익률
ax2 = axes[1]
ax2_twin = ax2.twinx()

ax2.fill_between(df['date'], df['s2_pos'], step='post', alpha=0.4,
                 where=df['s2_pos'] < 0, color='#2196F3', label='Long')
ax2.fill_between(df['date'], df['s2_pos'], step='post', alpha=0.4,
                 where=df['s2_pos'] > 0, color='#F44336', label='Short')
ax2.set_ylabel('포지션', fontsize=11)
ax2.set_yticks([-1, 0, 1])
ax2.set_yticklabels(['Short', '관망', 'Long'], fontsize=9)
ax2.set_ylim(-1.5, 1.5)

ax2_twin.plot(df['date'], df['s2_cumret'], color='#FF6F00',
              linewidth=1.5, label='S2 누적 수익률')
ax2_twin.set_ylabel('누적 수익률 (%)', fontsize=11, color='#FF6F00')
ax2_twin.tick_params(axis='y', labelcolor='#FF6F00')

lines1 = [Patch(facecolor='#2196F3', alpha=0.4, label='Long'),
          Patch(facecolor='#F44336', alpha=0.4, label='Short'),
          Line2D([0], [0], color='#FF6F00', linewidth=1.5, label='누적 수익률')]
ax2.legend(handles=lines1, loc='upper left', fontsize=9, framealpha=0.9)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

out_path = ROOT / 'reports/no_leak_v2/figures/12_prediction_with_signals.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'[SAVED] {out_path}')

print('\n=== 완료 ===')
print(f'Long 진입: {len(buy_signals)}회')
print(f'Short 진입: {len(sell_signals)}회')
print(f'포지션 청산: {len(exit_signals)}회')
print(f'S2 최종 수익률: {df["s2_cumret"].iloc[-1]:+.1f}%')
print(f'Buy-Hold 최종 수익률: {df["bh_cumret"].iloc[-1]:+.1f}%')

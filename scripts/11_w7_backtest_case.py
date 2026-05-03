# -*- coding: utf-8 -*-
"""
W7 발표 자료 보강 — 2024 계엄 케이스 + 거래 backtest
====================================================
사용자 요청 (선택지 2 + 4):

A. 2024 계엄 케이스 스터디 (2024-12-03 비상계엄 선포)
   - 전후 5~10일 q05/q50/q95 + 실제 Δy 시각화
   - 그날 모델이 capture 했는지 정량
   - 충격 흡수 능력 평가

B. 거래 backtest (모델 실용성 검증)
   - 전략: position = sign(q50) → bond short/long
   - PnL: sign(q50) × Δy × D / 10000 (D=8 modified duration)
   - 비교: A0 LSTM vs Random ±1 vs Buy-and-Hold (always long)
   - 지표: 누적 수익, Sharpe (252 영업일), Max Drawdown, 승률
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import warnings; warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 데이터 로드
pred = pd.read_csv(ROOT/'data/processed/lstm_a0_predictions_w6.csv', parse_dates=['date'])
test = pred[pred['split']=='test'].copy().reset_index(drop=True)
crisis = pd.read_csv(ROOT/'reports/crisis_labels_w6.csv', parse_dates=['date'])
test = test.merge(crisis[['date','is_crisis']], on='date', how='left')
test['is_crisis'] = test['is_crisis'].fillna(False)

print(f'test N = {len(test)}, range {test["date"].min().date()} ~ {test["date"].max().date()}')

# ==================================================================
print('\n' + '='*78)
print('Section A: 2024 비상계엄 케이스 스터디 (2024-12-03 22:30 선포)')
print('='*78)
# ==================================================================
event_date = pd.Timestamp('2024-12-03')
window = pd.Timedelta(days=10)
case = test[(test['date']>=event_date-window) & (test['date']<=event_date+window)].copy().reset_index(drop=True)
print(f'\n  케이스 윈도우 ({len(case)}일): {case["date"].min().date()} ~ {case["date"].max().date()}')
print(f'\n  일자별 표:')
case_show = case[['date','y_true_bp','q05','q50','q95','is_crisis']].copy()
case_show['in_band'] = (case_show['y_true_bp']>=case_show['q05']) & (case_show['y_true_bp']<=case_show['q95'])
case_show['sign_correct'] = np.sign(case_show['q50'])==np.sign(case_show['y_true_bp'])
case_show['err_q50'] = case_show['y_true_bp'] - case_show['q50']
print(case_show.to_string(index=False))

# 12-04 (계엄 다음 영업일) 분석
post_dates = case[case['date'] > event_date].head(5)
print(f'\n  계엄 후 첫 5 영업일 정량:')
for _, r in post_dates.iterrows():
    in_band = r['q05'] <= r['y_true_bp'] <= r['q95']
    sign_ok = np.sign(r['q50'])==np.sign(r['y_true_bp']) and r['y_true_bp']!=0
    days_after = (r['date']-event_date).days
    print(f"    +{days_after:>2}d ({r['date'].date()}): Δy={r['y_true_bp']:+.2f} bp, "
          f"q50={r['q50']:+.2f}, [q05,q95]=[{r['q05']:+.2f}, {r['q95']:+.2f}]  "
          f"in_band={'✓' if in_band else '✗'}, sign={'✓' if sign_ok else '✗'}")

n_in_band = case_show['in_band'].sum()
n_sign = (case_show['sign_correct'] & (case_show['y_true_bp']!=0)).sum()
print(f'\n  케이스 윈도우 종합: in_band {n_in_band}/{len(case_show)} ({n_in_band/len(case_show)*100:.0f}%), '
      f'방향 정확 {n_sign}/{len(case_show)} ({n_sign/len(case_show)*100:.0f}%)')

# 시각화
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.fill_between(case['date'], case['q05'], case['q95'], alpha=0.25, color='steelblue', label='90% 예측 구간')
ax.plot(case['date'], case['q50'], color='steelblue', linewidth=1.5, label='q50 (median)', marker='o', markersize=4)
ax.scatter(case['date'], case['y_true_bp'], s=35, color='black', alpha=0.85, zorder=3, label='실제 Δy')
ax.axhline(0, color='gray', linewidth=0.5)
# 계엄일 표시
ax.axvline(event_date, color='crimson', linewidth=2, linestyle='--', alpha=0.8, label='2024-12-03 비상계엄 선포 (22:30)')
ax.text(event_date, ax.get_ylim()[1]*0.9, '  ⚠ 계엄', color='crimson', fontsize=11, fontweight='bold')
# 미스 시점 강조
miss = case[~case_show['in_band']]
ax.scatter(miss['date'], miss['y_true_bp'], s=120, facecolors='none', edgecolors='crimson', linewidths=2, zorder=4, label='band miss')
ax.set_xlabel(''); ax.set_ylabel('Δy (bp)')
ax.set_title('2024 비상계엄 전후 ±10일 — A0 LSTM 분위수 회귀 케이스 스터디')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.xticks(rotation=0)
plt.tight_layout()
fig_path = ROOT/'reports/figures/w7_case_2024_martial_law.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f'\n  💾 {fig_path.relative_to(ROOT)}')
plt.close()

# CSV 저장
case_show.to_csv(ROOT/'reports/case_study_w7.csv', index=False)
print(f'  💾 reports/case_study_w7.csv')

# ==================================================================
print('\n' + '='*78)
print('Section B: 거래 Backtest (A0 vs Random vs Buy-and-Hold)')
print('='*78)
# ==================================================================
D = 8.0  # KR 10y modified duration 근사 (yield 3.5% 기준)
# Δy in bp → 채권 일별 수익률 (decimal): -D × Δy / 10000
# 전략별 일별 수익:
#   - Long always: -D × Δy / 10000  (금리 내릴수록 수익)
#   - Short always: +D × Δy / 10000
#   - 모델 (sign q50 따라 short/long): sign(q50) × Δy × D / 10000
#     (q50>0 = 금리상승 예측 → short → 금리상승 시 수익)
#   - Random ±1: random sign × Δy × D / 10000

print(f'  Modified duration (KR 10y, ~3.5% yield 기준): D = {D}')
print(f'  변환: Δy_bp → 일별 채권 수익률 = -D × Δy / 10000')
print(f'  test 구간 누적 Δy = {test["y_true_bp"].sum():+.1f} bp ({test["y_true_bp"].sum()/100:+.3f}%p)')

# 전략별 일별 수익
np.random.seed(42)
test['pnl_buyhold'] = -D * test['y_true_bp'] / 10000  # always long
test['pnl_a0']     = np.sign(test['q50']) * test['y_true_bp'] * D / 10000
random_sign = np.random.choice([-1, 1], size=len(test))
test['pnl_random'] = random_sign * test['y_true_bp'] * D / 10000

# 위치가 0 인 경우 (q50==0) → no position, PnL=0
n_zero_q50 = (test['q50']==0).sum()
print(f'  q50==0 (no position) 일수: {n_zero_q50}')

# 누적 수익률 (단순 합산 — 작은 일별 변화 가정)
strategies = ['buyhold','a0','random']
labels = {'buyhold':'Buy-and-Hold (always long)', 'a0':'A0 LSTM (sign q50)', 'random':'Random ±1'}
colors = {'buyhold':'steelblue', 'a0':'crimson', 'random':'gray'}

# 정량 지표
def summarize(pnl_series):
    cum = pnl_series.cumsum()
    total_ret = float(cum.iloc[-1]) * 100  # in %
    mean_d = float(pnl_series.mean())
    std_d = float(pnl_series.std())
    sharpe = mean_d / std_d * np.sqrt(252) if std_d > 0 else float('nan')
    # Max Drawdown
    running_max = cum.cummax()
    dd = (cum - running_max)
    max_dd = float(dd.min()) * 100  # 음수
    # Win rate (excluding zero PnL days)
    win = (pnl_series > 0).sum()
    loss = (pnl_series < 0).sum()
    n = win + loss
    win_rate = win/n*100 if n else float('nan')
    return {'total_return_%': total_ret, 'sharpe_252': sharpe, 'max_drawdown_%': max_dd,
            'win_rate_%': win_rate, 'days_active': n}

results = []
for s in strategies:
    summ = summarize(test[f'pnl_{s}'])
    summ['strategy'] = labels[s]
    results.append(summ)
res_df = pd.DataFrame(results)[['strategy','total_return_%','sharpe_252','max_drawdown_%','win_rate_%','days_active']]
res_df = res_df.round(3)
print(f'\n  Backtest 결과 (test 2023-2025, {len(test)} 영업일):')
print(res_df.to_string(index=False))
res_df.to_csv(ROOT/'reports/backtest_w7.csv', index=False)
print(f'  💾 reports/backtest_w7.csv')

# 시각화 1: 누적 수익률
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax = axes[0]
for s in strategies:
    cum = test[f'pnl_{s}'].cumsum() * 100
    ax.plot(test['date'], cum, label=labels[s], color=colors[s], linewidth=1.6 if s=='a0' else 1.0)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('누적 수익률 (%)')
ax.set_title(f'백테스트 누적 수익 — KR 10y D={D}, test 2023-2025 ({len(test)} 영업일)')
ax.legend(loc='best')
ax.grid(alpha=0.3)
# A0 final 누적 표시
final_a0 = test['pnl_a0'].sum()*100
ax.text(0.02, 0.98, f'A0 final: {final_a0:+.2f}%\nSharpe: {summarize(test["pnl_a0"])["sharpe_252"]:.2f}',
        transform=ax.transAxes, fontsize=11, va='top', fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='crimson', linewidth=1.5))

# 시각화 2: Drawdown
ax = axes[1]
for s in strategies:
    cum = test[f'pnl_{s}'].cumsum() * 100
    rmax = cum.cummax()
    dd = cum - rmax
    ax.fill_between(test['date'], dd, 0, alpha=0.4, color=colors[s], label=labels[s])
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('')
ax.set_title('Drawdown — 누적 수익 peak 대비 하락')
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = ROOT/'reports/figures/w7_backtest_cumret.png'
plt.savefig(fig_path, dpi=120, bbox_inches='tight')
print(f'  💾 {fig_path.relative_to(ROOT)}')
plt.close()

# ==================================================================
print('\n' + '='*78)
print('통합 요약')
print('='*78)
# ==================================================================
print(f'\n  A. 계엄 케이스 (2024-12-03 ±10일, {len(case_show)} 영업일):')
print(f'     - in_band {n_in_band}/{len(case_show)} ({n_in_band/len(case_show)*100:.0f}%)')
print(f'     - 방향 정확 {n_sign}/{len(case_show)} ({n_sign/len(case_show)*100:.0f}%)')

print(f'\n  B. Backtest (test 2023-2025, {len(test)} 영업일, D={D}):')
for s in strategies:
    summ = summarize(test[f'pnl_{s}'])
    print(f"     {labels[s]:30s}: Total {summ['total_return_%']:+6.2f}%, "
          f"Sharpe {summ['sharpe_252']:>5.2f}, MDD {summ['max_drawdown_%']:>6.2f}%, "
          f"Win {summ['win_rate_%']:.1f}%")

# Q&A 답변 자료
a0_summ = summarize(test['pnl_a0'])
bh_summ = summarize(test['pnl_buyhold'])
print(f"\n  발표 Q&A 답변 자료:")
print(f"     'A0 LSTM 65.2% 방향성이 진짜 돈 되나?' →")
print(f"       3년 누적 {a0_summ['total_return_%']:+.2f}% (Sharpe {a0_summ['sharpe_252']:.2f}, MDD {a0_summ['max_drawdown_%']:.2f}%)")
print(f"       Buy-and-Hold {bh_summ['total_return_%']:+.2f}% 대비 alpha {a0_summ['total_return_%']-bh_summ['total_return_%']:+.2f}%p")

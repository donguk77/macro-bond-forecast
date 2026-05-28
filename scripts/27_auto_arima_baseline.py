"""
27_auto_arima_baseline.py — AIC 기준 최적 ARIMA 차수 탐색 + 베이스라인 재평가

목적:
  - 기존 ARIMA(1,0,1) 고정 차수 → auto_arima로 AIC 최적 (p,d,q) 탐색
  - d=0 고정 (이미 Δy 변화량을 타겟으로 사용)
  - 최적 ARIMA를 베이스라인으로 채택 → "가장 유리한 전통 모형"을 이겨야 ML 성과

실행:
  python scripts/27_auto_arima_baseline.py

산출물:
  - reports/auto_arima_search_results.csv  — 탐색한 모든 (p,q) 조합의 AIC/BIC
  - reports/auto_arima_baseline.csv        — 최적 ARIMA vs Naive 성능 비교
  - reports/figures/improved/auto_arima_comparison.png — 시각화
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ─────────────────────────────────────────────────────────────────────────
# 환경 설정
# ─────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports'
FIG_DIR = REPORT_DIR / 'figures' / 'improved'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

TARGET = CONFIG['project']['target']

SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}


def slice_period(df, period):
    s, e = SPLIT[period]
    return df.loc[s:e]


# ─────────────────────────────────────────────────────────────────────────
# 1. 데이터 로드 + Δy 생성 (02b 노트북과 동일)
# ─────────────────────────────────────────────────────────────────────────

print('=' * 72)
print('1. 데이터 로드')
print('=' * 72)

features_v1 = pd.read_csv(
    DATA_DIR / 'processed' / 'features_v1_candidate.csv',
    index_col='date', parse_dates=['date']
).sort_index()

features_v1 = features_v1.dropna(subset=[TARGET])

# 미국 마감변수 + 정책변수 shift(1) — 13_rerun_no_leak.py와 동일
US_MARKET_CLOSE_VARS = [
    v for v in ['us_treasury_10y', 'us_breakeven_10y', 'vix', 'sp500', 'dxy']
    if v in features_v1.columns
]
POLICY_VARS = [v for v in ['kr_base_rate', 'us_fed_funds'] if v in features_v1.columns]
SHIFT_VARS = POLICY_VARS + US_MARKET_CLOSE_VARS

features_safe = features_v1.copy()
for var in SHIFT_VARS:
    features_safe[var] = features_safe[var].shift(1)
features_safe = features_safe.dropna()

y = features_safe[TARGET]
delta_y = y.diff() * 100  # bp
delta_y.name = 'delta_y_bp'
delta_y = delta_y.dropna()

y_train = slice_period(delta_y, 'train')
y_cal = slice_period(delta_y, 'cal')
y_val = slice_period(delta_y, 'val')
y_test = slice_period(delta_y, 'test')

print(f'y_train: {len(y_train):,d} ({y_train.index.min().date()} ~ {y_train.index.max().date()})')
print(f'y_cal:   {len(y_cal):,d}')
print(f'y_val:   {len(y_val):,d}')
print(f'y_test:  {len(y_test):,d}')


# ─────────────────────────────────────────────────────────────────────────
# 2. Auto ARIMA — Grid Search (p, q) with d=0 fixed
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('2. Auto ARIMA — AIC 기준 최적 차수 탐색')
print('=' * 72)

MAX_P = 5
MAX_Q = 5

search_results = []

for p in range(0, MAX_P + 1):
    for q in range(0, MAX_Q + 1):
        if p == 0 and q == 0:
            continue
        order = (p, 0, q)
        try:
            model = ARIMA(y_train, order=order, trend='c').fit()
            search_results.append({
                'p': p, 'd': 0, 'q': q,
                'order': str(order),
                'AIC': model.aic,
                'BIC': model.bic,
                'n_params': model.df_model + 1,
                'converged': True,
            })
            print(f'  ARIMA{order}  AIC={model.aic:.1f}  BIC={model.bic:.1f}')
        except Exception as e:
            search_results.append({
                'p': p, 'd': 0, 'q': q,
                'order': str(order),
                'AIC': np.nan, 'BIC': np.nan,
                'n_params': np.nan,
                'converged': False,
            })
            print(f'  ARIMA{order}  FAILED: {str(e)[:60]}')

search_df = pd.DataFrame(search_results)
search_df = search_df.sort_values('AIC').reset_index(drop=True)

# 저장
search_df.to_csv(REPORT_DIR / 'auto_arima_search_results.csv', index=False)
print(f'\n탐색 결과 저장: reports/auto_arima_search_results.csv')

# 상위 10개 출력
print('\n=== AIC 기준 Top 10 ===')
print(search_df.head(10)[['order', 'AIC', 'BIC', 'n_params']].to_string(index=False))

best = search_df.iloc[0]
BEST_ORDER = (int(best['p']), 0, int(best['q']))
print(f'\n최적 차수: ARIMA{BEST_ORDER}  (AIC={best["AIC"]:.1f})')


# ─────────────────────────────────────────────────────────────────────────
# 3. 최적 ARIMA vs 기존 ARIMA(1,0,1) vs Naive 비교 평가
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('3. 베이스라인 성능 비교')
print('=' * 72)


def metrics_point(y_true, y_pred, name, split_name):
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mask = (np.sign(y_pred) != 0) & (np.sign(y_true) != 0)
    dir_acc = (
        float((np.sign(y_pred[mask]) == np.sign(y_true[mask])).mean())
        if mask.sum() > 0 else float('nan')
    )
    return {
        'model': name, 'split': split_name,
        'RMSE_bp': round(rmse, 3),
        'MAE_bp': round(mae, 3),
        'Dir_Acc': round(dir_acc, 4),
    }


y_full = pd.concat([y_train, y_cal, y_val, y_test]).sort_index()

all_rows = []

# (A) Naive (Δŷ = 0)
for sp, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    pred = pd.Series(0.0, index=yy.index)
    all_rows.append(metrics_point(yy.values, pred.values, 'Naive (Δŷ=0)', sp))

# (B) ARIMA(1,0,1) — 기존 베이스라인
ORDER_OLD = (1, 0, 1)
fit_old = ARIMA(y_train, order=ORDER_OLD, trend='c').fit()
fit_old_full = fit_old.apply(y_full, refit=False)
preds_old = fit_old_full.predict()

for sp, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    pred = preds_old.reindex(yy.index)
    valid = yy.dropna().index.intersection(pred.dropna().index)
    all_rows.append(metrics_point(
        yy.loc[valid].values, pred.loc[valid].values,
        f'ARIMA{ORDER_OLD}', sp))

# (C) 최적 ARIMA — auto_arima 결과
fit_best = ARIMA(y_train, order=BEST_ORDER, trend='c').fit()
fit_best_full = fit_best.apply(y_full, refit=False)
preds_best = fit_best_full.predict()

for sp, yy in [('train', y_train), ('cal', y_cal), ('val', y_val), ('test', y_test)]:
    pred = preds_best.reindex(yy.index)
    valid = yy.dropna().index.intersection(pred.dropna().index)
    all_rows.append(metrics_point(
        yy.loc[valid].values, pred.loc[valid].values,
        f'ARIMA{BEST_ORDER} (auto)', sp))

result_df = pd.DataFrame(all_rows)
result_df.to_csv(REPORT_DIR / 'auto_arima_baseline.csv', index=False)

print('\n=== 전체 결과 ===')
print(result_df.to_string(index=False))

# Test 구간만 요약
test_result = result_df[result_df['split'] == 'test'].copy()
print('\n=== Test 구간 (2023~2025) 비교 ===')
print(test_result[['model', 'RMSE_bp', 'MAE_bp', 'Dir_Acc']].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# 4. 최적 ARIMA 파라미터 상세 출력
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print(f'4. 최적 ARIMA{BEST_ORDER} 파라미터 상세')
print('=' * 72)

print(f'\nAIC = {fit_best.aic:.1f}')
print(f'BIC = {fit_best.bic:.1f}')
print(f'\n파라미터:')
print(fit_best.params.round(5).to_string())

# 기존 대비 AIC 개선
aic_old = fit_old.aic
aic_best = fit_best.aic
print(f'\nAIC 비교: ARIMA(1,0,1)={aic_old:.1f} → ARIMA{BEST_ORDER}={aic_best:.1f}  (Δ={aic_best - aic_old:+.1f})')


# ─────────────────────────────────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('5. 시각화')
print('=' * 72)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

test_only = result_df[result_df['split'] == 'test'].copy()
models = test_only['model'].tolist()
colors = ['#808080', '#E8A838', '#2ECC71']

# (1) RMSE
ax = axes[0]
bars = ax.barh(models, test_only['RMSE_bp'], color=colors, alpha=0.85)
ax.set_xlabel('RMSE (bp)')
ax.set_title('Test RMSE (bp)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, test_only['RMSE_bp']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=11)
ax.set_xlim(right=test_only['RMSE_bp'].max() * 1.15)
ax.grid(alpha=0.3, axis='x')

# (2) MAE
ax = axes[1]
bars = ax.barh(models, test_only['MAE_bp'], color=colors, alpha=0.85)
ax.set_xlabel('MAE (bp)')
ax.set_title('Test MAE (bp)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, test_only['MAE_bp']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', fontsize=11)
ax.set_xlim(right=test_only['MAE_bp'].max() * 1.15)
ax.grid(alpha=0.3, axis='x')

# (3) Dir Acc — Naive는 NaN이므로 0으로 표시하고 주석 추가
ax = axes[2]
dir_vals = test_only['Dir_Acc'].fillna(0).values * 100
bars = ax.barh(models, dir_vals, color=colors, alpha=0.85)
ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='랜덤 50%')
ax.set_xlabel('방향 정확도 (%)')
ax.set_title('방향 정확도 (%)', fontsize=13, fontweight='bold', color='#E74C3C')
for bar, val, orig in zip(bars, dir_vals, test_only['Dir_Acc'].values):
    if np.isnan(orig):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                'N/A (Δŷ=0)', va='center', fontsize=10, color='gray')
    else:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=11)
ax.set_xlim(right=60)
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='x')

fig.suptitle(
    f'베이스라인 비교 — Naive vs ARIMA(1,0,1) vs ARIMA{BEST_ORDER} (auto)\n'
    f'Test: 2023~2025  |  AIC: {aic_old:.0f} → {aic_best:.0f}',
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()

out_path = FIG_DIR / 'auto_arima_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print(f'저장: {out_path.relative_to(PROJECT_ROOT)}')


# ─────────────────────────────────────────────────────────────────────────
# 6. AIC Heatmap (p x q)
# ─────────────────────────────────────────────────────────────────────────

converged = search_df[search_df['converged']].copy()
pivot = converged.pivot_table(index='p', columns='q', values='AIC')

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns.astype(int))
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index.astype(int))
ax.set_xlabel('q (MA 차수)')
ax.set_ylabel('p (AR 차수)')
ax.set_title(f'ARIMA(p, 0, q) AIC Heatmap\n최적: ARIMA{BEST_ORDER} (AIC={aic_best:.0f})',
             fontsize=13, fontweight='bold')

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = 'white' if val > pivot.values[~np.isnan(pivot.values)].mean() else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='AIC')
plt.tight_layout()

heatmap_path = FIG_DIR / 'auto_arima_aic_heatmap.png'
fig.savefig(heatmap_path, dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print(f'저장: {heatmap_path.relative_to(PROJECT_ROOT)}')


# ─────────────────────────────────────────────────────────────────────────
# 7. 결론 요약 출력
# ─────────────────────────────────────────────────────────────────────────

print('\n' + '=' * 72)
print('결론')
print('=' * 72)

rmse_old = test_result[test_result['model'] == f'ARIMA{ORDER_OLD}']['RMSE_bp'].values[0]
rmse_best = test_result[test_result['model'] == f'ARIMA{BEST_ORDER} (auto)']['RMSE_bp'].values[0]
rmse_naive = test_result[test_result['model'] == 'Naive (Δŷ=0)']['RMSE_bp'].values[0]

da_old = test_result[test_result['model'] == f'ARIMA{ORDER_OLD}']['Dir_Acc'].values[0]
da_best = test_result[test_result['model'] == f'ARIMA{BEST_ORDER} (auto)']['Dir_Acc'].values[0]
da_naive = test_result[test_result['model'] == 'Naive (Δŷ=0)']['Dir_Acc'].values[0]

print(f'1. 최적 차수: ARIMA{BEST_ORDER} (AIC 기준)')
print(f'2. AIC 개선: {aic_old:.0f} → {aic_best:.0f} (Δ={aic_best - aic_old:+.0f})')
print(f'3. Test RMSE: Naive={rmse_naive:.3f} / ARIMA(1,0,1)={rmse_old:.3f} / ARIMA{BEST_ORDER}={rmse_best:.3f}')
print(f'4. Test Dir_Acc: Naive={da_naive:.1%} / ARIMA(1,0,1)={da_old:.1%} / ARIMA{BEST_ORDER}={da_best:.1%}')

if rmse_best < rmse_old:
    print(f'\n→ ARIMA{BEST_ORDER}가 ARIMA(1,0,1) 대비 RMSE {rmse_old - rmse_best:.3f}bp 개선')
    print(f'  발표에서 "AIC 기준 최적 ARIMA를 베이스라인으로 사용" 으로 정당화 가능')
else:
    print(f'\n→ ARIMA(1,0,1)과 ARIMA{BEST_ORDER}의 Test RMSE 차이가 미미함')
    print(f'  어느 것을 써도 무방하나, auto_arima로 탐색했다는 사실 자체가 방법론적 정당화')

print('\n완료!')

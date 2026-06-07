"""
31_shap_analysis.py — XGBoost v2 SHAP 변수 영향도 분석

14개 기본 변수 → 162개 파생 피처(lag, rolling)를 원본 변수 그룹으로 묶어 해석
fold3 (train 2010-2021, test 2023-2025) 기준

산출물:
  reports/figures/improved/shap_group_importance.png   변수 그룹별 중요도
  reports/figures/improved/shap_summary_top20.png      개별 피처 beeswarm
  reports/figures/improved/shap_dependence_top6.png    상위 6개 dependence
  reports/figures/improved/shap_feature_type.png       현재값/lag/rolling 기여도
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 11,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
FIG_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'improved'
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

XGB_Q50 = {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400}

FOLD3 = {
    'train': ('2010-01-01', '2021-12-31'),
    'val': ('2022-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2025-12-31'),
}

KR_LABELS = {
    'kr_treasury_3y': '한국 국채 3Y',
    'kr_base_rate': '한국 기준금리',
    'us_treasury_10y': '미국 국채 10Y',
    'us_fed_funds': '미국 기준금리(FFR)',
    'us_breakeven_10y': '미국 BEI 10Y\n(기대인플레)',
    'vix': 'VIX\n(변동성 지수)',
    'kospi': 'KOSPI',
    'sp500': 'S&P 500',
    'dxy': '달러 인덱스\n(DXY)',
    'spread_10y_t1': '한미 금리차\n(t-1)',
    'delta_us10y_t1': 'Δ미국10Y\n(t-1)',
    'delta_vix_t1': 'ΔVIX\n(t-1)',
    'delta_dxy_t1': 'ΔDXY\n(t-1)',
    'crisis_dummy': '위기 더미',
}

CATEGORY = {
    'kr_treasury_3y': '국내 금리',
    'kr_base_rate': '국내 금리',
    'us_treasury_10y': '미국 금리',
    'us_fed_funds': '미국 금리',
    'us_breakeven_10y': '미국 금리',
    'vix': '위험지표',
    'kospi': '주식시장',
    'sp500': '주식시장',
    'dxy': '환율/달러',
    'spread_10y_t1': '파생 변수',
    'delta_us10y_t1': '파생 변수',
    'delta_vix_t1': '파생 변수',
    'delta_dxy_t1': '파생 변수',
    'crisis_dummy': '파생 변수',
}

CAT_COLORS = {
    '국내 금리': '#E53935',
    '미국 금리': '#1565C0',
    '위험지표': '#FF9800',
    '주식시장': '#4CAF50',
    '환율/달러': '#9C27B0',
    '파생 변수': '#607D8B',
}

BASE_VARS = [bv for bv in KR_LABELS.keys() if bv != 'kospi']  # v3: kospi 제외


def get_base_var(col_name):
    """피처명에서 기본 변수명 추출 (lag/rolling 제거)"""
    for bv in sorted(BASE_VARS, key=len, reverse=True):
        if col_name == bv or col_name.startswith(bv + '__'):
            return bv
    return col_name


def get_feature_type(col_name):
    """피처 유형 분류: 현재값, lag, rolling"""
    if '__lag' in col_name:
        return 'lag'
    elif '__rmean' in col_name or '__rstd' in col_name:
        return 'rolling'
    else:
        return 'current'


# ═══════════════════════════════════════════════════════════════════════
# 데이터 + 모델 학습
# ═══════════════════════════════════════════════════════════════════════

print('=' * 60)
print('31 — SHAP 변수 영향도 분석 (XGBoost v2, fold3)')
print('=' * 60)

df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp' and 'kospi' not in c.lower()]
print(f'피처 수: {len(FEATURE_COLS)} (kospi 제외 = v3 변수셋)')

X_tr = df.loc[FOLD3['train'][0]:FOLD3['train'][1]][FEATURE_COLS]
X_val = df.loc[FOLD3['val'][0]:FOLD3['val'][1]][FEATURE_COLS]
X_te = df.loc[FOLD3['test'][0]:FOLD3['test'][1]][FEATURE_COLS]
y_tr = df.loc[FOLD3['train'][0]:FOLD3['train'][1]]['delta_y_bp']
y_val = df.loc[FOLD3['val'][0]:FOLD3['val'][1]]['delta_y_bp']
y_te = df.loc[FOLD3['test'][0]:FOLD3['test'][1]]['delta_y_bp']

print(f'Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}')

print('XGBoost q50 모델 학습...')
model = xgb.XGBRegressor(
    objective='reg:quantileerror', quantile_alpha=0.5,
    n_estimators=XGB_Q50['n_estimators'],
    max_depth=XGB_Q50['max_depth'],
    learning_rate=XGB_Q50['learning_rate'],
    early_stopping_rounds=50,
    verbosity=0, tree_method='hist', random_state=42,
)
model.fit(X_tr.values, y_tr.values,
          eval_set=[(X_val.values, y_val.values)], verbose=False)
print(f'  Best iteration: {model.best_iteration}')

# ═══════════════════════════════════════════════════════════════════════
# SHAP 계산
# ═══════════════════════════════════════════════════════════════════════

print('SHAP 값 계산...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_te.values)
print(f'  SHAP shape: {shap_values.shape}')

shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLS, index=X_te.index)

# 기본 변수별 그룹핑
group_map = {col: get_base_var(col) for col in FEATURE_COLS}
type_map = {col: get_feature_type(col) for col in FEATURE_COLS}

# 변수 그룹별 평균 |SHAP|
group_importance = {}
for bv in BASE_VARS:
    cols = [c for c in FEATURE_COLS if group_map[c] == bv]
    group_importance[bv] = float(np.mean(np.abs(shap_df[cols].values)))

gi_sorted = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
print('\n변수 그룹별 평균 |SHAP| (top → bottom):')
for bv, val in gi_sorted:
    print(f'  {KR_LABELS[bv]:20s}  {val:.4f}  ({CATEGORY[bv]})')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 1: 변수 그룹별 SHAP 중요도 (수평 막대)
# ═══════════════════════════════════════════════════════════════════════

print('\n시각화 생성 중...')

fig, ax = plt.subplots(figsize=(12, 8))

names = [bv for bv, _ in reversed(gi_sorted)]
vals = [v for _, v in reversed(gi_sorted)]
colors = [CAT_COLORS[CATEGORY[bv]] for bv in names]
kr_names = [KR_LABELS[bv] for bv in names]

bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white',
               linewidth=0.5, alpha=0.85, height=0.7)

for i, (bar, v) in enumerate(zip(bars, vals)):
    ax.text(v + max(vals) * 0.01, i, f'{v:.4f}', va='center', fontsize=10)

ax.set_yticks(range(len(names)))
ax.set_yticklabels(kr_names, fontsize=11)
ax.set_xlabel('mean |SHAP value| (bp)', fontsize=12)
ax.set_title('변수 그룹별 SHAP 중요도 — XGBoost v2 q50 (fold3: 2023~2025)',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# 범주 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat) for cat, c in CAT_COLORS.items()]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
          title='변수 범주', title_fontsize=11, framealpha=0.9)

plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_group_importance.png')
plt.close(fig)
print(f'  저장: shap_group_importance.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 2: 개별 피처 Top 20 Beeswarm
# ═══════════════════════════════════════════════════════════════════════

# 개별 피처 중요도
feat_importance = np.mean(np.abs(shap_values), axis=0)
top20_idx = np.argsort(feat_importance)[-20:][::-1]

fig, ax = plt.subplots(figsize=(12, 9))

top20_names = [FEATURE_COLS[i] for i in top20_idx]
top20_vals = feat_importance[top20_idx]
top20_colors = [CAT_COLORS[CATEGORY[get_base_var(c)]] for c in top20_names]

# 한글 레이블 + 피처 유형 표시
def format_feat_name(col):
    bv = get_base_var(col)
    kr = KR_LABELS[bv].replace('\n', ' ')
    ft = get_feature_type(col)
    if ft == 'current':
        return f'{kr} (현재값)'
    suffix = col.replace(bv + '__', '')
    return f'{kr} ({suffix})'

top20_kr = [format_feat_name(c) for c in top20_names]

bars = ax.barh(range(len(top20_names)), top20_vals[::-1],
               color=[top20_colors[i] for i in range(len(top20_names))][::-1],
               edgecolor='white', linewidth=0.5, alpha=0.85, height=0.7)

for i, (bar, v) in enumerate(zip(bars, top20_vals[::-1])):
    ax.text(v + max(top20_vals) * 0.01, i, f'{v:.4f}', va='center', fontsize=9)

ax.set_yticks(range(len(top20_names)))
ax.set_yticklabels(top20_kr[::-1], fontsize=10)
ax.set_xlabel('mean |SHAP value| (bp)', fontsize=12)
ax.set_title('개별 피처 Top 20 SHAP 중요도 — XGBoost v2 q50',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_summary_top20.png')
plt.close(fig)
print(f'  저장: shap_summary_top20.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 3: SHAP Beeswarm (색상=피처값) — Top 15
# ═══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 9))

top15_idx = np.argsort(feat_importance)[-15:][::-1]
top15_names = [FEATURE_COLS[i] for i in top15_idx]
top15_kr = [format_feat_name(c) for c in top15_names]

shap_top15 = shap_values[:, top15_idx]
X_top15 = X_te.values[:, top15_idx]

n_feat = len(top15_names)
for i in range(n_feat):
    fi = n_feat - 1 - i
    sv = shap_top15[:, i]
    fv = X_top15[:, i]

    fv_norm = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-8)
    colors_scatter = plt.cm.coolwarm(fv_norm)

    jitter = np.random.normal(0, 0.12, len(sv))
    ax.scatter(sv, fi + jitter, c=colors_scatter, s=5, alpha=0.4, rasterized=True)

ax.set_yticks(range(n_feat))
ax.set_yticklabels(top15_kr[::-1], fontsize=10)
ax.axvline(0, color='grey', ls='-', lw=0.8)
ax.set_xlabel('SHAP value (bp에 대한 기여)', fontsize=12)
ax.set_title('SHAP Beeswarm — 피처값 크기별 기여 방향 (Top 15)',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm,
                            norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['낮음', '높음'])
cbar.set_label('피처값 크기', fontsize=11)

plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_beeswarm_top15.png')
plt.close(fig)
print(f'  저장: shap_beeswarm_top15.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 4: SHAP Dependence Plot — 상위 6개 기본변수
# ═══════════════════════════════════════════════════════════════════════

top6_base = [bv for bv, _ in gi_sorted[:6]]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax_idx, bv in enumerate(top6_base):
    ax = axes[ax_idx]

    # 해당 변수 그룹에서 가장 중요한 개별 피처 선택
    cols_bv = [c for c in FEATURE_COLS if group_map[c] == bv]
    col_imp = {c: float(np.mean(np.abs(shap_df[c].values))) for c in cols_bv}
    best_col = max(col_imp, key=col_imp.get)
    best_idx = FEATURE_COLS.index(best_col)

    feat_val = X_te[best_col].values
    sv = shap_values[:, best_idx]

    scatter = ax.scatter(feat_val, sv, c=feat_val, cmap='coolwarm',
                        s=8, alpha=0.5, rasterized=True)
    ax.axhline(0, color='grey', ls='--', lw=0.5)

    # 트렌드 라인 (lowess 대신 다항식)
    valid = ~(np.isnan(feat_val) | np.isnan(sv))
    if valid.sum() > 10:
        z = np.polyfit(feat_val[valid], sv[valid], 3)
        p = np.poly1d(z)
        x_sorted = np.sort(feat_val[valid])
        ax.plot(x_sorted, p(x_sorted), color='black', lw=2, alpha=0.7)

    kr_label = KR_LABELS[bv].replace('\n', ' ')
    feat_type = format_feat_name(best_col).split('(')[-1].rstrip(')')
    ax.set_title(f'{kr_label}\n(best: {feat_type})', fontsize=12, fontweight='bold')
    ax.set_xlabel('피처값', fontsize=10)
    ax.set_ylabel('SHAP value (bp)', fontsize=10)
    ax.grid(alpha=0.3)

fig.suptitle('SHAP Dependence — 상위 6개 변수의 비선형 관계',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_dependence_top6.png')
plt.close(fig)
print(f'  저장: shap_dependence_top6.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 5: 피처 유형별 기여도 (현재값 vs lag vs rolling)
# ═══════════════════════════════════════════════════════════════════════

type_contribution = {bv: {'current': 0, 'lag': 0, 'rolling': 0} for bv in BASE_VARS}
for col in FEATURE_COLS:
    bv = group_map[col]
    ft = type_map[col]
    type_contribution[bv][ft] += float(np.mean(np.abs(shap_df[col].values)))

# 총 중요도 순으로 정렬
tc_sorted = sorted(type_contribution.items(),
                   key=lambda x: sum(x[1].values()), reverse=True)

fig, ax = plt.subplots(figsize=(14, 8))

bv_names = [bv for bv, _ in tc_sorted]
kr_labels = [KR_LABELS[bv] for bv in bv_names]
current_vals = [tc_sorted[i][1]['current'] for i in range(len(tc_sorted))]
lag_vals = [tc_sorted[i][1]['lag'] for i in range(len(tc_sorted))]
rolling_vals = [tc_sorted[i][1]['rolling'] for i in range(len(tc_sorted))]

x = np.arange(len(bv_names))
w = 0.6

bars1 = ax.bar(x, current_vals, w, label='현재값 (t)', color='#E53935', alpha=0.85)
bars2 = ax.bar(x, lag_vals, w, bottom=current_vals, label='시차값 (lag)', color='#1565C0', alpha=0.85)
bottom2 = [c + l for c, l in zip(current_vals, lag_vals)]
bars3 = ax.bar(x, rolling_vals, w, bottom=bottom2, label='이동통계 (rolling)', color='#4CAF50', alpha=0.85)

# 총합 레이블
for i, (c, l, r) in enumerate(zip(current_vals, lag_vals, rolling_vals)):
    total = c + l + r
    ax.text(i, total + max([c+l+r for c,l,r in zip(current_vals,lag_vals,rolling_vals)]) * 0.01,
            f'{total:.3f}', ha='center', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(kr_labels, fontsize=10, rotation=45, ha='right')
ax.set_ylabel('mean |SHAP| 합계 (bp)', fontsize=12)
ax.set_title('피처 유형별 SHAP 기여도 — 현재값 vs 시차 vs 이동통계',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_feature_type.png')
plt.close(fig)
print(f'  저장: shap_feature_type.png')


# ═══════════════════════════════════════════════════════════════════════
# 시각화 6: 시간에 따른 SHAP 기여도 변화 (상위 5개 변수)
# ═══════════════════════════════════════════════════════════════════════

top5_base = [bv for bv, _ in gi_sorted[:5]]

fig, ax = plt.subplots(figsize=(16, 7))

dates_te = X_te.index

# 변수 그룹별 SHAP 합산 시계열
for bv in top5_base:
    cols_bv = [c for c in FEATURE_COLS if group_map[c] == bv]
    group_shap = shap_df[cols_bv].sum(axis=1)
    ma20 = group_shap.rolling(20, min_periods=1).mean()
    kr_label = KR_LABELS[bv].replace('\n', ' ')
    ax.plot(dates_te, ma20, lw=2, label=kr_label, alpha=0.85)

ax.axhline(0, color='grey', ls='--', lw=1)
ax.set_ylabel('SHAP 기여도 합계 (bp, 20일 MA)', fontsize=12)
ax.set_title('상위 5개 변수의 SHAP 기여도 시계열 — 시간에 따른 영향력 변화',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(alpha=0.3)

import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
fig.savefig(FIG_DIR / 'shap_temporal_top5.png')
plt.close(fig)
print(f'  저장: shap_temporal_top5.png')


# ═══════════════════════════════════════════════════════════════════════
# 해석 요약 출력
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 60)
print('SHAP 변수 해석 요약')
print('=' * 60)

print('\n[변수 그룹 중요도 순위]')
for rank, (bv, val) in enumerate(gi_sorted, 1):
    kr = KR_LABELS[bv].replace('\n', ' ')
    cat = CATEGORY[bv]
    tc = type_contribution[bv]
    total = sum(tc.values())
    pct_current = tc['current'] / total * 100 if total > 0 else 0
    pct_lag = tc['lag'] / total * 100 if total > 0 else 0
    pct_rolling = tc['rolling'] / total * 100 if total > 0 else 0
    print(f'  {rank:2d}. {kr:22s} [{cat}]  '
          f'|SHAP|={val:.4f}  '
          f'현재={pct_current:.0f}% lag={pct_lag:.0f}% rolling={pct_rolling:.0f}%')

# 개별 피처 Top 10
print('\n[개별 피처 Top 10]')
for rank, idx in enumerate(top20_idx[:10], 1):
    col = FEATURE_COLS[idx]
    val = feat_importance[idx]
    kr = format_feat_name(col)
    print(f'  {rank:2d}. {kr:35s}  |SHAP|={val:.4f}')

# 범주별 합산
cat_total = {}
for bv, val in group_importance.items():
    cat = CATEGORY[bv]
    cat_total[cat] = cat_total.get(cat, 0) + val
cat_sorted = sorted(cat_total.items(), key=lambda x: x[1], reverse=True)
print('\n[범주별 SHAP 합산]')
total_all = sum(v for _, v in cat_sorted)
for cat, val in cat_sorted:
    print(f'  {cat:12s}  {val:.4f}  ({val/total_all*100:.1f}%)')

print('\n완료!')

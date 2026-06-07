# -*- coding: utf-8 -*-
"""
46 — 전체 후보 풀(28개)에서 13변수 무작위 조합 성능 분포 (진짜 v3 full 모델 기준)

목적: "우리 예측셋(13변수)이 전체 후보에서 무작위로 뽑은 13변수 조합 대비
       통계적으로 상위인가?" — 진짜 v3 파이프라인(0.61대) 기준으로 검증.

⚠️ 공정·누수안전 규칙:
  - 모든 거시 raw에 동일하게 shift(1) (전일 정보만) → 발표시차 누수 0
  - 파생 5개(이미 t-1 설계)는 그대로
  - rolling은 .shift(1) (14_features_v2와 동일 CL-03)
  - 우리 셋(SET_A)도 같은 규칙으로 재계산 → 동일 조건 비교
  - 이 보수적 규칙 때문에 절대 dir_acc는 발표용 0.614와 다를 수 있음.
    중요한 건 "상대 백분위". (분포는 선택의 사후검증, cherry-pick 금지)

사용: python scripts/46_fullpool_subset_distribution.py [N]   (기본 N=500)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import RobustScaler
import warnings; warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT  = ROOT / 'reports' / 'no_leak_v2'
FIG  = ROOT / 'reports' / 'figures' / 'v3'
TARGET = 'kr_treasury_10y'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
SEED = 42
LAGS = [1, 5, 10, 20, 30]; ROLLS = [5, 10, 20]

# ── 데이터 ──
wide = pd.read_csv(DATA/'interim'/'wide_daily_filled.csv', index_col='date', parse_dates=['date']).sort_index()
feat_v2 = pd.read_csv(DATA/'processed'/'features_v2_no_leak.csv', index_col='date', parse_dates=['date']).sort_index()

DERIVED = ['spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy']
MACRO = [c for c in wide.columns if c != TARGET and wide[c].notna().mean() > 0.9]
POOL = MACRO + DERIVED
print(f'거시 raw {len(MACRO)} + 파생 {len(DERIVED)} = 후보 풀 {len(POOL)}')
print(' MACRO:', MACRO)

# ── base 구성 (14_features_v2 방식: dropna 먼저 → 행기준 연속 → shift/lag/roll) ──
base = pd.DataFrame(index=wide.index)
base['__tgt'] = wide[TARGET]
for v in MACRO:   base[v] = wide[v]
for v in DERIVED: base[v] = feat_v2[v].reindex(wide.index)
base = base.dropna()                                  # 28변수+타깃 공통 유효행만 (결측 증폭 차단)
y_bp = (base['__tgt'].diff() * 100).rename('delta_y_bp')   # 행기준 Δy(bp)
for v in MACRO:   base[v] = base[v].shift(1)          # 거시 누수처리: 직전 영업일(행기준)
base = base.drop(columns='__tgt')

# ── lag/rolling 확장 (14_features_v2 동일 스키마) ──
cols = {}
for c in POOL:
    s = base[c]; cols[c] = s
    for k in LAGS: cols[f'{c}__lag{k}'] = s.shift(k)
    if c == 'crisis_dummy': continue
    for w in ROLLS:
        cols[f'{c}__rmean{w}'] = s.rolling(w).mean().shift(1)
        cols[f'{c}__rstd{w}']  = s.rolling(w).std().shift(1)
master = pd.DataFrame(cols)
master['delta_y_bp'] = y_bp
master = master.dropna()
ALLF = [c for c in master.columns if c != 'delta_y_bp']
print(f'master {master.shape}  {master.index.min().date()} ~ {master.index.max().date()}  | 확장피처 {len(ALLF)}')

# ── 34_top_combos와 동일한 full 평가 ──
FOLDS = [('2010-01-01','2017-12-31','2018-01-01','2019-12-31','2020-01-01','2020-12-31'),
         ('2010-01-01','2019-12-31','2020-01-01','2020-12-31','2021-01-01','2022-12-31'),
         ('2010-01-01','2021-12-31','2022-01-01','2022-12-31','2023-01-01','2025-12-31')]
P = dict(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=400, max_depth=4,
         learning_rate=0.05, early_stopping_rounds=50, tree_method='hist', random_state=42, verbosity=0)
def expand(rs):
    out = []
    for v in rs: out += [x for x in ALLF if x == v or x.startswith(v+'__')]
    return out
def ev(rs):
    f = expand(rs); accs = []
    for a,b,c,d,e,g in FOLDS:
        Xtr,Xv,Xte = master.loc[a:b][f], master.loc[c:d][f], master.loc[e:g][f]
        ytr,yv,yte = master.loc[a:b]['delta_y_bp'], master.loc[c:d]['delta_y_bp'], master.loc[e:g]['delta_y_bp']
        sc = RobustScaler().fit(Xtr); m = xgb.XGBRegressor(**P)
        m.fit(sc.transform(Xtr), ytr, eval_set=[(sc.transform(Xv), yv)], verbose=False)
        pr = m.predict(sc.transform(Xte)); yvl = yte.values
        msk = (np.sign(pr) != 0) & (np.sign(yvl) != 0)
        accs.append(float((np.sign(pr[msk]) == np.sign(yvl[msk])).mean()))
    return float(np.mean(accs))

SET_A = ['kr_treasury_3y','us_treasury_10y','us_breakeven_10y','dxy','kr_base_rate','us_fed_funds',
         'vix','sp500','spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy']
SET_A = [v for v in SET_A if v in POOL]
our = ev(SET_A)
print(f'\n[우리 예측셋 SET_A, {len(SET_A)}변수] dir_acc(동일규칙) = {our:.4f}\n')

# ── 무작위 13변수 sweep ──
rng = np.random.default_rng(SEED)
scores = []; t0 = time.time()
for i in range(N):
    combo = list(rng.choice(POOL, size=13, replace=False))
    s = ev(combo)
    if not np.isnan(s): scores.append(s)
    if (i+1) % 50 == 0:
        el = time.time()-t0
        print(f'  {i+1}/{N}  경과 {el/60:.1f}분  (1회 {el/(i+1):.2f}s, 남은 ~{el/(i+1)*(N-i-1)/60:.1f}분)')
scores = np.array(scores)
pct = float((scores < our).mean()*100)

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'dir_acc': scores}).to_csv(OUT/f'fullpool_subset_scores_N{N}.csv', index=False)

print(f'\n=== 전체 풀 13변수 무작위 {len(scores)}개 분포 ===')
print(f'  중앙값={np.median(scores):.4f}  평균={scores.mean():.4f}  std={scores.std():.4f}')
print(f'  최소={scores.min():.4f}  90%={np.quantile(scores,0.9):.4f}  최대={scores.max():.4f}')
print(f'  우리 셋={our:.4f} → 상위 {100-pct:.1f}% (백분위 {pct:.1f})')
se = float(np.sqrt((pct/100)*(1-pct/100)/len(scores))*100)
print(f'  백분위 95% 오차 ±{1.96*se:.1f}%p  (N={len(scores)})')

# ── 히스토그램 ──
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.hist(scores, bins=30, color='lightgray', edgecolor='gray')
    ax.axvline(our, color='seagreen', lw=2.2, label=f'우리 예측셋 = {our:.3f} (상위 {100-pct:.0f}%)')
    ax.axvline(np.median(scores), color='black', ls=':', lw=1.2, label=f'무작위 중앙값 {np.median(scores):.3f}')
    ax.set_xlabel('walk-forward 방향정확도 (v3 full)'); ax.set_ylabel('빈도')
    ax.set_title(f'전체 후보({len(POOL)})에서 13변수 무작위 {len(scores)}개 vs 우리 선택')
    ax.legend()
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/f'fullpool_subset_hist_N{N}.png', dpi=130, bbox_inches='tight')
    print(f'  💾 그림: reports/figures/v3/fullpool_subset_hist_N{N}.png')
except Exception as ex:
    print('  (그림 생략:', ex, ')')

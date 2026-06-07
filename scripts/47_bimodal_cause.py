# -*- coding: utf-8 -*-
"""
47 — 이봉분포 원인 규명 (scripts/46 확장)
46 분포가 두 봉우리(왼~0.50 신호없음 / 오른~0.585 신호있음)로 갈리는 원인 검증.
가설: 오른쪽 봉우리 = 핵심 채널 변수(간밤 US 금리·한미 스프레드 등) 포함 조합.
→ 각 조합의 변수 구성 + dir_acc 저장 후:
  (1) 변수별 포함/미포함 dir_acc 차이 (봉우리 결정력 순위)
  (2) 최상위 변수로 분포 분할 히스토그램 (이봉이 그 변수로 설명되는지)
  (3) 좌/우 봉우리 그룹의 변수 출현율 비교
N=5000, seed=42 → 46과 동일 분포 재현.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import RobustScaler
import warnings; warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'; OUT = ROOT / 'reports' / 'no_leak_v2'; FIG = ROOT / 'reports' / 'figures' / 'v3'
TARGET = 'kr_treasury_10y'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SEED = 42; LAGS = [1, 5, 10, 20, 30]; ROLLS = [5, 10, 20]

# ── master 구성 (46과 동일) ──
wide = pd.read_csv(DATA/'interim'/'wide_daily_filled.csv', index_col='date', parse_dates=['date']).sort_index()
feat_v2 = pd.read_csv(DATA/'processed'/'features_v2_no_leak.csv', index_col='date', parse_dates=['date']).sort_index()
DERIVED = ['spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy']
MACRO = [c for c in wide.columns if c != TARGET and wide[c].notna().mean() > 0.9]
POOL = MACRO + DERIVED
base = pd.DataFrame(index=wide.index); base['__tgt'] = wide[TARGET]
for v in MACRO:   base[v] = wide[v]
for v in DERIVED: base[v] = feat_v2[v].reindex(wide.index)
base = base.dropna()
y_bp = (base['__tgt'].diff() * 100).rename('delta_y_bp')
for v in MACRO:   base[v] = base[v].shift(1)
base = base.drop(columns='__tgt')
cols = {}
for c in POOL:
    s = base[c]; cols[c] = s
    for k in LAGS: cols[f'{c}__lag{k}'] = s.shift(k)
    if c == 'crisis_dummy': continue
    for w in ROLLS:
        cols[f'{c}__rmean{w}'] = s.rolling(w).mean().shift(1)
        cols[f'{c}__rstd{w}']  = s.rolling(w).std().shift(1)
master = pd.DataFrame(cols); master['delta_y_bp'] = y_bp; master = master.dropna()
ALLF = [c for c in master.columns if c != 'delta_y_bp']
print(f'후보 풀 {len(POOL)} | master {master.shape}')

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

# ── sweep: 조합 변수 + dir_acc 저장 ──
rng = np.random.default_rng(SEED)
combos = []; scores = []; t0 = time.time()
for i in range(N):
    combo = sorted(rng.choice(POOL, size=13, replace=False).tolist())
    s = ev(combo)
    if not np.isnan(s): combos.append(combo); scores.append(s)
    if (i+1) % 100 == 0:
        el = time.time()-t0
        print(f'  {i+1}/{N}  {el/60:.1f}분  남은~{el/(i+1)*(N-i-1)/60:.1f}분')
scores = np.array(scores)
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'vars': ['+'.join(c) for c in combos], 'dir_acc': scores}).to_csv(
    OUT/f'bimodal_combos_N{N}.csv', index=False)

# ── (1) 변수별 포함/미포함 dir_acc 차이 ──
print('\n=== (1) 변수별 봉우리 결정력 (포함 - 미포함 평균 dir_acc) ===')
rows = []
for v in POOL:
    inc = np.array([s for c, s in zip(combos, scores) if v in c])
    exc = np.array([s for c, s in zip(combos, scores) if v not in c])
    if len(inc) == 0 or len(exc) == 0: continue
    rows.append({'변수': v, 'n_포함': len(inc), '포함평균': inc.mean(), '미포함평균': exc.mean(),
                 'Δ': inc.mean()-exc.mean()})
rank = pd.DataFrame(rows).sort_values('Δ', ascending=False).reset_index(drop=True)
pd.set_option('display.width', 140); pd.set_option('display.max_rows', 40)
print(rank.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
rank.to_csv(OUT/f'bimodal_varrank_N{N}.csv', index=False)

# ── (2) 좌/우 봉우리 그룹 변수 출현율 (경계 = 두 봉 사이 골 ≈ 0.54) ──
GAP = 0.54
left = [c for c, s in zip(combos, scores) if s < GAP]
right = [c for c, s in zip(combos, scores) if s >= GAP]
print(f'\n=== (2) 봉우리 분리 (경계 {GAP}) — 좌봉 {len(left)}개 / 우봉 {len(right)}개 ===')
from collections import Counter
cl = Counter(); cr = Counter()
for c in left: cl.update(c)
for c in right: cr.update(c)
occ = []
for v in POOL:
    lr = cl[v]/len(left) if left else 0
    rr = cr[v]/len(right) if right else 0
    occ.append({'변수': v, '좌봉출현율': lr, '우봉출현율': rr, '우-좌': rr-lr})
occ = pd.DataFrame(occ).sort_values('우-좌', ascending=False).reset_index(drop=True)
print(occ.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# ── (3) 최상위 변수로 분할 히스토그램 ──
KEY = rank.iloc[0]['변수']
print(f'\n최상위 결정 변수 = {KEY}')
inc_s = np.array([s for c, s in zip(combos, scores) if KEY in c])
exc_s = np.array([s for c, s in zip(combos, scores) if KEY not in c])
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Malgun Gothic'; plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    bins = np.linspace(scores.min(), scores.max(), 32)
    ax.hist(exc_s, bins=bins, color='#b0b0b0', alpha=0.85, label=f'{KEY} 미포함 (n={len(exc_s)})')
    ax.hist(inc_s, bins=bins, color='#2ca02c', alpha=0.7, label=f'{KEY} 포함 (n={len(inc_s)})')
    ax.axvline(0.610, color='black', lw=1.8, ls='-', label='우리 선택 0.610')
    ax.set_xlabel('walk-forward 방향정확도 (v3 full)'); ax.set_ylabel('빈도')
    ax.set_title(f'이봉분포 원인: 핵심 변수 "{KEY}" 포함 여부가 봉우리를 가른다')
    ax.legend(fontsize=9)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG/f'bimodal_cause_N{N}.png', dpi=140, bbox_inches='tight')
    print(f'💾 그림: reports/figures/v3/bimodal_cause_N{N}.png')
except Exception as ex:
    print('(그림 생략:', ex, ')')

print(f'\n핵심변수 포함 평균={inc_s.mean():.4f} vs 미포함 평균={exc_s.mean():.4f} (차이 {inc_s.mean()-exc_s.mean():+.4f})')

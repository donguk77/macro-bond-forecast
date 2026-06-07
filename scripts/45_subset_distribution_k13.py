# -*- coding: utf-8 -*-
"""
45 — 랜덤 서브셋 성능 분포 재검증 (nb09 §6 확장)
질문: "전체 풀(~25개)에서 13개를 뽑은 분포에서 우리 예측셋(13개)이 우측 꼬리에 있나?"
nb09 §6은 k=5(그룹대표 수)로만 돌렸음 → 예측셋(13개)과 단위 불일치.
여기서 k=5와 k=13 둘 다 돌려 SET_A(13)/SET_B/SET_C(5)의 백분위를 함께 본다.
프록시: nb09와 동일(단일 XGBoost 점예측 + t-1 lag, walk-forward dir acc).
"""
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings; warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
TARGET = 'kr_treasury_10y'
SEED = 42; np.random.seed(SEED)

# --- 데이터 (nb09 §1과 동일) ---
wide = pd.read_csv(DATA/'interim'/'wide_daily_filled.csv', index_col='date', parse_dates=['date'])
feat_v2 = pd.read_csv(DATA/'processed'/'features_v2_no_leak.csv', index_col='date', parse_dates=['date'])
DERIVED = ['spread_10y_t1','delta_us10y_t1','delta_vix_t1','delta_dxy_t1','crisis_dummy']
derived_avail = [c for c in DERIVED if c in feat_v2.columns]
master = wide.join(feat_v2[derived_avail], how='left')
target_delta = wide[TARGET].diff().rename('target_dy')

# --- 셋 정의 (nb09 §2,§3과 동일) ---
GROUPS = {
    'G1':['us_treasury_10y','delta_us10y_t1','us_fed_funds','us_breakeven_10y'],
    'G2':['spread_10y_t1','dxy','delta_dxy_t1'],
    'G3':['kr_treasury_3y','kr_base_rate'],
    'G4':['vix','delta_vix_t1','sp500'],
    'G5':['crisis_dummy'],
}
REP = {'G1':'delta_us10y_t1','G2':'spread_10y_t1','G3':'kr_treasury_3y','G4':'delta_vix_t1','G5':'crisis_dummy'}
BASE_V2 = ['kr_treasury_3y','us_treasury_10y','us_breakeven_10y','dxy','kr_base_rate','us_fed_funds','vix','sp500']
SET_A = [c for c in BASE_V2 + derived_avail if c in master.columns]            # 13 (kospi 제외 예측셋)
SET_B = SET_A + [c for c in ['kr_treasury_5y','kr_treasury_1y','kospi'] if c in master.columns]
SET_C = [REP[g] for g in GROUPS if REP[g] in master.columns]                   # 5 (그룹대표)

def eval_dir_acc(cols, n_splits=3, seed=SEED):
    cols = [c for c in cols if c in master.columns]
    if not cols: return np.nan
    X = master[cols].shift(1)
    df = pd.concat([X, target_delta], axis=1).dropna()
    Xv, yv = df[cols].values, df['target_dy'].values
    tscv = TimeSeriesSplit(n_splits=n_splits); accs=[]
    for tr, te in tscv.split(Xv):
        m = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.05,
                         subsample=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1)
        m.fit(Xv[tr], yv[tr]); pred = m.predict(Xv[te])
        accs.append(float(np.mean(np.sign(pred)==np.sign(yv[te]))))
    return float(np.mean(accs))

CAND_POOL = [c for c in master.columns if c != TARGET and master[c].notna().mean() > 0.9]
print('후보 풀 크기:', len(CAND_POOL))
print('  ', CAND_POOL)
print('SET_A (예측셋):', len(SET_A), SET_A)
print('SET_B        :', len(SET_B))
print('SET_C (그룹대표):', len(SET_C), SET_C)
acc_A, acc_B, acc_C = eval_dir_acc(SET_A), eval_dir_acc(SET_B), eval_dir_acc(SET_C)
print(f'\ndir_acc(proxy): A={acc_A:.4f}  B={acc_B:.4f}  C={acc_C:.4f}\n')

N = 500
rng = np.random.default_rng(SEED)
def sweep(k):
    s=[]
    for _ in range(N):
        cols = list(rng.choice(CAND_POOL, size=k, replace=False))
        v = eval_dir_acc(cols)
        if not np.isnan(v): s.append(v)
    return np.array(s)

for k, label_sets in [(5, [('C',acc_C),('A',acc_A)]), (13, [('A',acc_A),('B',acc_B)])]:
    dist = sweep(k)
    print(f'=== k={k} 무작위 {len(dist)}개 분포 ===')
    print(f'  중앙값={np.median(dist):.4f}  90%분위={np.quantile(dist,0.9):.4f}  최대={dist.max():.4f}')
    for name, acc in label_sets:
        pct = float((dist < acc).mean()*100)
        print(f'  SET_{name} (acc={acc:.4f}) → 상위 {100-pct:.1f}% (백분위 {pct:.1f})')
    print()

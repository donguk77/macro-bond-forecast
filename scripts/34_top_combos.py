"""34_top_combos.py — C(13,5) 전수 조합의 dir acc 순위 + 변수 확인 (히스토그램 이상치 규명)"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_csv(ROOT/'data'/'processed'/'features_v3_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
TARGET='delta_y_bp'
ALLF=[c for c in DF.columns if c!=TARGET]
CAND=[c for c in ALLF if '__' not in c]   # 13 raw
print('후보 raw:', len(CAND), CAND)
FOLDS=[('2010-01-01','2017-12-31','2018-01-01','2019-12-31','2020-01-01','2020-12-31'),
       ('2010-01-01','2019-12-31','2020-01-01','2020-12-31','2021-01-01','2022-12-31'),
       ('2010-01-01','2021-12-31','2022-01-01','2022-12-31','2023-01-01','2025-12-31')]
P=dict(objective='reg:quantileerror',quantile_alpha=0.5,n_estimators=400,max_depth=4,
       learning_rate=0.05,early_stopping_rounds=50,tree_method='hist',random_state=42,verbosity=0)
def expand(rs):
    c=[]
    for v in rs: c+=[x for x in ALLF if x==v or x.startswith(v+'__')]
    return c
def ev(rs):
    f=expand(rs); accs=[]
    for a,b,c,d,e,g in FOLDS:
        Xtr,Xv,Xte=DF.loc[a:b][f],DF.loc[c:d][f],DF.loc[e:g][f]
        ytr,yv,yte=DF.loc[a:b][TARGET],DF.loc[c:d][TARGET],DF.loc[e:g][TARGET]
        sc=RobustScaler().fit(Xtr); m=xgb.XGBRegressor(**P)
        m.fit(sc.transform(Xtr),ytr,eval_set=[(sc.transform(Xv),yv)],verbose=False)
        pr=m.predict(sc.transform(Xte)); yvl=yte.values
        msk=(np.sign(pr)!=0)&(np.sign(yvl)!=0)
        accs.append((np.sign(pr[msk])==np.sign(yvl[msk])).mean())
    return float(np.mean(accs)), [round(x,3) for x in accs]

GROUPREP=['delta_us10y_t1','spread_10y_t1','kr_treasury_3y','delta_vix_t1','crisis_dummy']
rows=[]
for j,combo in enumerate(combinations(CAND,5)):
    s,folds=ev(list(combo))
    rows.append({'score':s,'fold1':folds[0],'fold2':folds[1],'fold3':folds[2],'vars':'+'.join(combo)})
    if (j+1)%200==0: print(f'{j+1}/1287 ...')
res=pd.DataFrame(rows).sort_values('score',ascending=False).reset_index(drop=True)
res.to_csv(ROOT/'reports'/'no_leak_v2'/'combo_enum_v3.csv',index=False)
our=ev(GROUPREP)[0]
print(f'\n우리 그룹대표 dir acc = {our:.4f} (순위 {int((res.score>our).sum())+1}/{len(res)})')
print('\n=== 상위 8개 조합 ===')
print(res.head(8).to_string(index=False))
print('\n=== 분포 요약 ===')
print(f'max={res.score.max():.4f} mean={res.score.mean():.4f} std={res.score.std():.4f}')
print(f'상위 5개 평균={res.score.head(5).mean():.4f} | 2등과 1등 격차={res.score.iloc[0]-res.score.iloc[1]:.4f}')
top=res.iloc[0]
print(f'\n1등 변수: {top.vars}')
print(f'1등 fold별: {top.fold1}/{top.fold2}/{top.fold3}  (전체 {top.score:.4f})')
# 변수 출현 빈도 (상위 20)
from collections import Counter
cnt=Counter()
for v in res.head(20)['vars']: cnt.update(v.split('+'))
print('\n상위 20개 조합 변수 출현빈도:', dict(cnt.most_common()))

"""
36_horizon_sweep.py — 예측 호라이즌별 방향정확도 (일별 노이즈 vs 누적 신호 검증)

가설: 일별 Δy는 노이즈(RMSE≈naive)지만, h일 누적 방향(향후 h일 금리 등락 합의 부호)은
       추세 누적·노이즈 상쇄로 더 예측 가능할 것.
타깃_h(t) = sum(Δy_{t..t+h-1})  (피처는 이미 shift(1) → 누수 없음)
모델: v3(13변수, 150피처) 동일 XGBoost q50, walk-forward 3-fold.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import RobustScaler
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DF = pd.read_csv(ROOT/'data'/'processed'/'features_v3_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
TARGET='delta_y_bp'; FEAT=[c for c in DF.columns if c!=TARGET]
FOLDS=[('2010-01-01','2017-12-31','2018-01-01','2019-12-31','2020-01-01','2020-12-31'),
       ('2010-01-01','2019-12-31','2020-01-01','2020-12-31','2021-01-01','2022-12-31'),
       ('2010-01-01','2021-12-31','2022-01-01','2022-12-31','2023-01-01','2025-12-31')]
P=dict(objective='reg:quantileerror',quantile_alpha=0.5,n_estimators=400,max_depth=4,
       learning_rate=0.05,early_stopping_rounds=50,tree_method='hist',random_state=42,verbosity=0)

def diracc(y,p):
    m=(np.sign(p)!=0)&(np.sign(y)!=0)
    return float((np.sign(p[m])==np.sign(y[m])).mean()) if m.any() else np.nan, int(m.sum())

print(f'{"h":>3} | {"fold1":>13} {"fold2":>13} {"fold3":>13} | {"평균dir":>7} {"풀dir":>7} {"풀n":>6}')
results=[]
for h in [1,2,3,5,10]:
    # 향후 h일 누적 타깃 (forward rolling sum)
    yh = DF[TARGET].rolling(h).sum().shift(-(h-1))
    accs=[]; pool_p=[]; pool_y=[]
    for a,b,c,d,e,g in FOLDS:
        Xtr,Xv,Xte=DF.loc[a:b][FEAT],DF.loc[c:d][FEAT],DF.loc[e:g][FEAT]
        ytr,yv=yh.loc[a:b],yh.loc[c:d]
        # train/val에서 타깃 NaN(끝부분) 제거
        tr=ytr.notna(); vl=yv.notna()
        sc=RobustScaler().fit(Xtr[tr.values])
        m=xgb.XGBRegressor(**P)
        m.fit(sc.transform(Xtr[tr.values]), ytr[tr], eval_set=[(sc.transform(Xv[vl.values]), yv[vl])], verbose=False)
        yte=yh.loc[e:g]; te=yte.notna()
        pr=m.predict(sc.transform(Xte[te.values])); yt=yte[te].values
        da,_=diracc(yt,pr); accs.append(da)
        pool_p.append(pr); pool_y.append(yt)
    pp=np.concatenate(pool_p); py=np.concatenate(pool_y)
    pooled,pn=diracc(py,pp)
    results.append({'h':h,'fold1':accs[0],'fold2':accs[1],'fold3':accs[2],'mean':np.mean(accs),'pooled':pooled,'pool_n':pn})
    print(f'{h:>3} | {accs[0]:>13.4f} {accs[1]:>13.4f} {accs[2]:>13.4f} | {np.mean(accs):>7.4f} {pooled:>7.4f} {pn:>6}')

pd.DataFrame(results).to_csv(ROOT/'reports'/'no_leak_v2'/'horizon_diracc_v3.csv', index=False)
print('\nSaved horizon_diracc_v3.csv')

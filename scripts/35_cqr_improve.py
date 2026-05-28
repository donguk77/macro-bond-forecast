"""
35_cqr_improve.py — CQR 캘리브레이션 개선 실험 (fold2 레짐시프트 미달 해결 시도)

비교:
  static_6mo   : 현재 방식 (cal=val 마지막 6개월, 고정 Q)
  static_full  : cal=val 전체 (더 큰 고정 Q)
  online_W     : 롤링/온라인 conformal — 최근 W일 실현 conformity로 Q 매일 갱신
                 (Gibbs&Candes 2021 류; 과거 실현값만 사용 → 누수 없음)
목표: 마진/조건부 coverage를 90%에 근접시키되 폭 과증가 억제.
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
ALPHA=0.10; QS=[0.05,0.5,0.95]
FOLDS=[
 {'name':'fold1','train':('2010-01-01','2017-12-31'),'val':('2018-01-01','2019-12-31'),'cal6':('2019-07-01','2019-12-31'),'test':('2020-01-01','2020-12-31')},
 {'name':'fold2','train':('2010-01-01','2019-12-31'),'val':('2020-01-01','2020-12-31'),'cal6':('2020-07-01','2020-12-31'),'test':('2021-01-01','2022-12-31')},
 {'name':'fold3','train':('2010-01-01','2021-12-31'),'val':('2022-01-01','2022-12-31'),'cal6':('2022-07-01','2022-12-31'),'test':('2023-01-01','2025-12-31')},
]
def qp(a): return dict(objective='reg:quantileerror',quantile_alpha=a,n_estimators=400,max_depth=4,
                       learning_rate=0.05,early_stopping_rounds=50,tree_method='hist',random_state=42,verbosity=0)
def qhat(s,a=ALPHA):
    s=np.sort(s); k=min(int(np.ceil((len(s)+1)*(1-a))),len(s)); return float(s[k-1])
def cov(y,lo,hi): return float(((y>=lo)&(y<=hi)).mean())
def wid(lo,hi): return float(np.mean(hi-lo))
def e0(lo,hi): return float(((lo>0)|(hi<0)).mean())

def run():
    agg={m:{'y':[],'lo':[],'hi':[]} for m in ['static_6mo','static_full','online_125','online_250']}
    perfold={}
    for fd in FOLDS:
        nm=fd['name']
        def sl(a,b): return DF.loc[a:b]
        Xtr=sl(*fd['train'])[FEAT]; Xval=sl(*fd['val'])[FEAT]
        sc=RobustScaler().fit(Xtr)
        ytr=sl(*fd['train'])[TARGET].values; yval=sl(*fd['val'])[TARGET].values
        # 모델 학습
        models={}
        for q in QS:
            m=xgb.XGBRegressor(**qp(q)); m.fit(sc.transform(Xtr),ytr,eval_set=[(sc.transform(Xval),yval)],verbose=False); models[q]=m
        def predict(idx):
            X=sc.transform(DF.loc[idx][FEAT])
            P=np.column_stack([models[q].predict(X) for q in QS]); P=np.sort(P,axis=1)
            return P[:,0],P[:,1],P[:,2]
        # cal6 / fullval conformity
        for cal_key,label in [('cal6','static_6mo'),('val','static_full')]:
            ci=DF.loc[fd[cal_key][0]:fd[cal_key][1]].index
            q05c,_,q95c=predict(ci); yc=DF.loc[ci][TARGET].values
            Q=qhat(np.maximum(q05c-yc, yc-q95c))
            ti=DF.loc[fd['test'][0]:fd['test'][1]].index
            q05t,_,q95t=predict(ti); yt=DF.loc[ti][TARGET].values
            lo,hi=q05t-Q, q95t+Q
            agg[label]['y']+=list(yt); agg[label]['lo']+=list(lo); agg[label]['hi']+=list(hi)
            perfold.setdefault(label,{})[nm]=(cov(yt,lo,hi),wid(lo,hi),e0(lo,hi),Q)
        # online rolling: cal6 으로 buffer 시작 → test 진행하며 매일 갱신
        ci=DF.loc[fd['cal6'][0]:fd['cal6'][1]].index
        q05c,_,q95c=predict(ci); yc=DF.loc[ci][TARGET].values
        buf=list(np.maximum(q05c-yc, yc-q95c))   # 초기 conformity buffer
        ti=DF.loc[fd['test'][0]:fd['test'][1]].index
        q05t,_,q95t=predict(ti); yt=DF.loc[ti][TARGET].values
        for W,label in [(125,'online_125'),(250,'online_250')]:
            b=list(buf); los=[];his=[]
            for i in range(len(ti)):
                Q=qhat(np.array(b[-W:]))           # 최근 W일(과거 실현)로 Q
                los.append(q05t[i]-Q); his.append(q95t[i]+Q)
                b.append(max(q05t[i]-yt[i], yt[i]-q95t[i]))   # 오늘 실현 후 buffer 추가
            los=np.array(los);his=np.array(his)
            agg[label]['y']+=list(yt); agg[label]['lo']+=list(los); agg[label]['hi']+=list(his)
            perfold.setdefault(label,{})[nm]=(cov(yt,los,his),wid(los,his),e0(los,his),np.nan)
    return agg,perfold

agg,perfold=run()
print('\n=== fold별 coverage / 폭 / 0배제 ===')
for label in ['static_6mo','static_full','online_125','online_250']:
    print(f'\n[{label}]')
    for nm in ['fold1','fold2','fold3']:
        c,w,e,Q=perfold[label][nm]; print(f'  {nm}: cov={c:.3f} 폭={w:6.2f} 0배제={e:.3f}'+(f' Q={Q:.2f}' if Q==Q else ''))
    y=np.array(agg[label]['y']);lo=np.array(agg[label]['lo']);hi=np.array(agg[label]['hi'])
    print(f'  POOLED cov={cov(y,lo,hi):.3f} 폭={wid(lo,hi):.2f} 0배제={e0(lo,hi):.3f}')

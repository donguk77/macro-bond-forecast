"""
33_quantile_v3_cqr.py — v3(13변수) XGBoost 분위수 + CQR 4종 (plan §3.1)

q05/q50/q95 독립 XGBoost(quantile) → 비교차 정렬 → CQR 변형:
  raw    : 보정 없음 (등분산 → 평탄 밴드)
  sym    : 대칭 CQR (단일 Q_hat 양쪽 동일) — 과보정 baseline
  asym   : 비대칭 CQR (상/하한 conformity 각각, Romano 2019)
  local  : 국소적응 CQR (과거 실현변동성으로 정규화 → 가변폭, 누수 없음)

날짜별 lo/hi 저장 + 지표(coverage·sharpness·IS·0배제율·레짐 coverage) 요약.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'
ALPHA = 0.10
QUANTILES = [0.05, 0.5, 0.95]

FOLDS = [
    {'name':'fold1','train':('2010-01-01','2017-12-31'),'val':('2018-01-01','2019-12-31'),'cal':('2019-07-01','2019-12-31'),'test':('2020-01-01','2020-12-31')},
    {'name':'fold2','train':('2010-01-01','2019-12-31'),'val':('2020-01-01','2020-12-31'),'cal':('2020-07-01','2020-12-31'),'test':('2021-01-01','2022-12-31')},
    {'name':'fold3','train':('2010-01-01','2021-12-31'),'val':('2022-01-01','2022-12-31'),'cal':('2022-07-01','2022-12-31'),'test':('2023-01-01','2025-12-31')},
]

DF = pd.read_csv(DATA_DIR/'processed'/'features_v3_candidate.csv', index_col='date', parse_dates=['date']).sort_index()
TARGET = 'delta_y_bp'
FEAT = [c for c in DF.columns if c != TARGET]
# 누수 없는 변동성 프록시: 과거 20일 실현변동성 (오늘 타깃 제외 → shift(1))
SIGMA = DF[TARGET].rolling(20).std().shift(1)
# VIX 원자료 (레짐 분할용; v3 raw 변수에 포함)
VIX = DF['vix'] if 'vix' in DF.columns else None
print(f'v3 features: {len(FEAT)} | sigma median={SIGMA.median():.2f}')

def qparams(alpha):
    return dict(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=400,
                max_depth=4, learning_rate=0.05, early_stopping_rounds=50,
                tree_method='hist', random_state=42, verbosity=0)

def fit_predict_q(Xtr, ytr, Xval, yval, *Xs):
    out = {}
    for q in QUANTILES:
        m = xgb.XGBRegressor(**qparams(q))
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        out[q] = [m.predict(X) for X in Xs]
    return out  # out[q] = [pred_cal, pred_test]

rows = []
for fd in FOLDS:
    name = fd['name']
    def sl(p): return DF.loc[fd[p][0]:fd[p][1]]
    Xtr_raw, Xval_raw = sl('train')[FEAT], sl('val')[FEAT]
    Xcal_raw, Xte_raw = sl('cal')[FEAT], sl('test')[FEAT]
    sc = RobustScaler().fit(Xtr_raw)
    Xtr, Xval = sc.transform(Xtr_raw), sc.transform(Xval_raw)
    Xcal, Xte = sc.transform(Xcal_raw), sc.transform(Xte_raw)
    ytr, yval = sl('train')[TARGET].values, sl('val')[TARGET].values
    ycal, yte = sl('cal')[TARGET].values, sl('test')[TARGET].values
    pred = fit_predict_q(Xtr, ytr, Xval, yval, Xcal, Xte)

    # 비교차 정렬 (q05<=q50<=q95)
    def noncross(qd, idx):
        arr = np.column_stack([qd[q][idx] for q in QUANTILES])
        arr = np.sort(arr, axis=1)
        return arr[:,0], arr[:,1], arr[:,2]
    q05c, q50c, q95c = noncross(pred, 0)   # cal
    q05t, q50t, q95t = noncross(pred, 1)   # test

    sig_cal = SIGMA.reindex(sl('cal').index).values
    sig_te  = SIGMA.reindex(sl('test').index).values
    sig_cal = np.where(np.isfinite(sig_cal)&(sig_cal>0), sig_cal, np.nanmedian(SIGMA))
    sig_te  = np.where(np.isfinite(sig_te)&(sig_te>0), sig_te, np.nanmedian(SIGMA))

    # --- CQR 변형 ---
    n = len(ycal)
    def qhat(scores, a=ALPHA):
        k = min(int(np.ceil((len(scores)+1)*(1-a))), len(scores))
        return float(np.sort(scores)[k-1])
    # 대칭
    Qs = qhat(np.maximum(q05c - ycal, ycal - q95c))
    # 비대칭 (상/하한 독립)
    Qlo = qhat(q05c - ycal); Qhi = qhat(ycal - q95c)
    # 국소적응 (정규화 점수)
    s_norm = np.maximum(q05c - ycal, ycal - q95c) / sig_cal
    Qn = qhat(s_norm)

    methods = {
        'raw':   (q05t,            q95t),
        'sym':   (q05t - Qs,       q95t + Qs),
        'asym':  (q05t - Qlo,      q95t + Qhi),
        'local': (q05t - Qn*sig_te, q95t + Qn*sig_te),
    }
    dates = sl('test').index
    vix_te = VIX.reindex(dates).values if VIX is not None else np.full(len(dates), np.nan)
    for i, dt in enumerate(dates):
        r = {'date': dt, 'fold': name, 'y_true': float(yte[i]), 'q50': float(q50t[i]),
             'sigma': float(sig_te[i]), 'vix': float(vix_te[i])}
        for mth,(lo,hi) in methods.items():
            r[f'{mth}_lo'] = float(lo[i]); r[f'{mth}_hi'] = float(hi[i])
        rows.append(r)
    print(f'{name}: cal n={n} Qs={Qs:.2f} Qlo={Qlo:.2f} Qhi={Qhi:.2f} Qn={Qn:.3f}')

out = pd.DataFrame(rows)
out.to_csv(REPORT_DIR/'predictions_xgb_v3_intervals.csv', index=False)
print(f'\nSaved predictions_xgb_v3_intervals.csv {out.shape}')

# --- 요약 지표 ---
def metrics(y, lo, hi):
    cov = float(((y>=lo)&(y<=hi)).mean())
    width = float(np.mean(hi-lo))
    IS = (hi-lo) + (2/ALPHA)*(lo-y)*(y<lo) + (2/ALPHA)*(y-hi)*(y>hi)
    excl0 = float(((lo>0)|(hi<0)).mean())     # 0 배제율 = 방향 신호 비율
    return cov, width, float(np.mean(IS)), excl0

y = out['y_true'].values
print('\n=== 전체 풀 ===')
print(f'{"method":7s} {"cov":>6s} {"width":>7s} {"IS":>7s} {"0배제율":>8s}')
summ=[]
for m in ['raw','sym','asym','local']:
    cov,w,IS,e0 = metrics(y, out[f'{m}_lo'].values, out[f'{m}_hi'].values)
    summ.append({'method':m,'coverage':cov,'sharpness_bp':w,'interval_score':IS,'excl0_rate':e0})
    print(f'{m:7s} {cov:6.3f} {w:7.2f} {IS:7.2f} {e0:8.3f}')
pd.DataFrame(summ).to_csv(REPORT_DIR/'cqr_v3_summary.csv', index=False)

# 레짐별 coverage (VIX 중앙값 기준 고/저변동)
if VIX is not None:
    med = np.nanmedian(out['vix'])
    print(f'\n=== 레짐별 coverage (VIX median={med:.1f}) ===')
    for m in ['sym','asym','local']:
        hi_mask = out['vix']>med; lo_mask=~hi_mask
        cov_hi = float(((y>=out[f'{m}_lo'])&(y<=out[f'{m}_hi']))[hi_mask].mean())
        cov_lo = float(((y>=out[f'{m}_lo'])&(y<=out[f'{m}_hi']))[lo_mask].mean())
        print(f'{m:7s} 고변동 cov={cov_hi:.3f} | 저변동 cov={cov_lo:.3f}')

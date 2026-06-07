"""
38_live_oos_2026.py — §3.2 2026 라이브 OOS (진짜 검증대)

동결 v3 모델로 2026 신규 데이터를 포워드 예측 → 방향정확도·DM·백테스트.
모델은 2026을 학습/검증/보정 어디에도 쓰지 않는다 (완전 OOS).

동결 설계 (fold3 구조를 2026으로 한 칸 전진):
  train 2010-01-01~2024-12-31 → val(early stop) 2025 → cal(conformal) 2025-07~12 → test 2026
모델 레시피: scripts/33_quantile_v3_cqr.py 와 동일 (독립 q05/q50/q95, RobustScaler, early_stopping=50)
지표 정의: nb11(DM HLN), nb13(백테스트 D=8/C=85/캐리/1bp/블록부트스트랩) 그대로.

사용: python scripts/38_live_oos_2026.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from scipy.stats import t as t_dist

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
REP = ROOT / 'reports' / 'no_leak_v2'
REP.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
QUANTILES = [0.05, 0.5, 0.95]
TARGET = 'delta_y_bp'
SEED = 42

SPLIT = {
    'train': ('2010-01-01', '2024-12-31'),
    'val':   ('2025-01-01', '2025-12-31'),
    'cal':   ('2025-07-01', '2025-12-31'),
    'test':  ('2026-01-01', '2026-12-31'),   # 2026 라이브 OOS
}

DF = pd.read_csv(DATA / 'processed' / 'features_v3_candidate.csv',
                 index_col='date', parse_dates=['date']).sort_index()
FEAT = [c for c in DF.columns if c != TARGET]
SIGMA = DF[TARGET].rolling(20).std().shift(1)   # 누수없는 변동성 프록시
print('=' * 72)
print(f'38_live_oos_2026 — v3 features={len(FEAT)}')
for k, (s, e) in SPLIT.items():
    g = DF.loc[s:e]
    print(f'  {k:5s} {s}~{e}  n={len(g)}  ({g.index.min().date() if len(g) else "-"} ~ {g.index.max().date() if len(g) else "-"})')
print('=' * 72)

def qparams(alpha):
    return dict(objective='reg:quantileerror', quantile_alpha=alpha, n_estimators=400,
                max_depth=4, learning_rate=0.05, early_stopping_rounds=50,
                tree_method='hist', random_state=SEED, verbosity=0)

def sl(p): return DF.loc[SPLIT[p][0]:SPLIT[p][1]]

Xtr_raw, Xval_raw, Xcal_raw, Xte_raw = sl('train')[FEAT], sl('val')[FEAT], sl('cal')[FEAT], sl('test')[FEAT]
sc = RobustScaler().fit(Xtr_raw)
Xtr, Xval, Xcal, Xte = (sc.transform(x) for x in (Xtr_raw, Xval_raw, Xcal_raw, Xte_raw))
ytr, yval = sl('train')[TARGET].values, sl('val')[TARGET].values
ycal, yte = sl('cal')[TARGET].values, sl('test')[TARGET].values
test_idx = sl('test').index

# --- q05/q50/q95 학습 (train), early stop (val), 예측 (cal, test) ---
preds = {}
for q in QUANTILES:
    m = xgb.XGBRegressor(**qparams(q))
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    preds[q] = {'cal': m.predict(Xcal), 'test': m.predict(Xte)}

def noncross(split):
    arr = np.column_stack([preds[q][split] for q in QUANTILES])
    arr = np.sort(arr, axis=1)
    return arr[:, 0], arr[:, 1], arr[:, 2]
q05c, q50c, q95c = noncross('cal')
q05t, q50t, q95t = noncross('test')

# ─────────────────────────────────────────────────────────────────
# 1. 방향정확도 (sign q50) + 부트스트랩 CI  — nb11 정의
# ─────────────────────────────────────────────────────────────────
def diracc(y, p):
    m = (np.sign(p) != 0) & (np.sign(y) != 0)
    return float((np.sign(p[m]) == np.sign(y[m])).mean()) if m.any() else np.nan

dacc = diracc(yte, q50t)
n_eff = int(((np.sign(q50t) != 0) & (np.sign(yte) != 0)).sum())
# 방향 적중 부트스트랩 CI
rng = np.random.default_rng(1)
mask = (np.sign(q50t) != 0) & (np.sign(yte) != 0)
hits = (np.sign(q50t[mask]) == np.sign(yte[mask])).astype(float)
boot = [rng.choice(hits, size=len(hits), replace=True).mean() for _ in range(5000)]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f'\n[방향정확도] 2026 라이브 OOS = {dacc:.4f}  (n={n_eff}일)')
print(f'  부트스트랩 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]   (동전 0.50 기준)')
print(f'  ※ 비교: in-sample 워크포워드 pooled(2020-2025) = 0.621 (nb11)')

# ─────────────────────────────────────────────────────────────────
# 2. DM test vs Naive (제곱오차) — nb11 HLN 그대로
# ─────────────────────────────────────────────────────────────────
def dm_test_hln(e1, e2, lag=6):
    d = np.asarray(e1) - np.asarray(e2); n = len(d)
    d_mean = d.mean(); var = np.var(d, ddof=0)
    for k in range(1, lag + 1):
        wk = 1 - k / (lag + 1)
        var += 2 * wk * np.mean((d[:-k] - d_mean) * (d[k:] - d_mean))
    var = max(var, 1e-12)
    dm = d_mean / np.sqrt(var / n)
    h = 1; fac = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * fac
    p = 2 * (1 - t_dist.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)

rmse = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
e_naive = yte ** 2
e_xgb = (yte - q50t) ** 2
dmv, pv = dm_test_hln(e_xgb, e_naive, lag=6)
print(f'\n[DM test] XGB v3 q50 vs Naive (2026)')
print(f'  RMSE: XGB={rmse(yte,q50t):.4f}  Naive={rmse(yte,np.zeros_like(yte)):.4f} bp')
print(f'  DM_HLN={dmv:.4f}  p={pv:.4f}  → {"모델 유의 ✓" if dmv<0 and pv<0.05 else "유의하지 않음 (tie)"}')

# ─────────────────────────────────────────────────────────────────
# 3. 백테스트 (1일 방향신호) — nb13 정의 그대로
# ─────────────────────────────────────────────────────────────────
D, C, TXN_BP = 8.0, 85.0, 1.0
kr = pd.read_csv(DATA / 'raw' / 'raw_ecos.csv', parse_dates=['date'])
kr = kr[kr['variable'] == 'kr_treasury_10y'][['date', 'value']].rename(columns={'value': 'y10'})
bt = pd.DataFrame({'date': test_idx, 'y_true': yte, 'q50': q50t}).merge(kr, on='date', how='left')
dy = bt['y_true'].values / 10000.0; ylv = bt['y10'].values

def pnl(pos): return pos * D * dy - pos * 0.5 * C * dy ** 2 - pos * (ylv / 100) / 252
def sharpe(p): return float(p.mean() / p.std() * np.sqrt(252)) if p.std() > 0 else np.nan
def mdd(p):
    cum = np.cumsum(p); peak = np.maximum.accumulate(cum); return float((cum - peak).min() * 100)
def block_boot_sharpe(p, block=15, B=2000, seed=1):
    r = np.random.default_rng(seed); n = len(p); nb = int(np.ceil(n / block)); out = []
    for _ in range(B):
        st = r.integers(0, n - block + 1, nb)
        s = np.concatenate([p[i:i + block] for i in st])[:n]
        out.append(sharpe(pd.Series(s)))
    return np.percentile(out, [2.5, 97.5])

pos_x = np.sign(bt['q50']).values.astype(float)
gross = pnl(pos_x)
turnover = np.abs(np.diff(pos_x, prepend=pos_x[0]))
cost = turnover * TXN_BP * D / 10000
net = gross - cost
bh = pnl(-np.ones(len(bt)))
print(f'\n[백테스트] 2026 ({bt.date.min().date()}~{bt.date.max().date()}, {len(bt)}일)')
print(f'  포지션변경 {int(turnover.sum())}회 (회전율 {turnover.mean()*100:.0f}%)')
for nm, p in [('Buy&Hold', bh), ('XGB v3 무비용', gross), ('XGB v3 1bp비용', net)]:
    line = f'  {nm:14s} 누적={p.sum()*100:6.2f}%  Sharpe={sharpe(pd.Series(p)):6.3f}  MDD={mdd(p):6.2f}%  승률={(p>0).mean()*100:4.1f}%'
    if nm == 'XGB v3 1bp비용':
        lo, hi = block_boot_sharpe(net)
        line += f'  Sharpe95%CI=[{lo:.2f},{hi:.2f}]'
    print(line)

# 비용 민감도 sweep (Q&A 방어)
print('  비용 sweep:', end=' ')
for cbp in [0.5, 1.0, 2.0, 3.0]:
    nn = gross - turnover * cbp * D / 10000
    print(f'{cbp}bp→Sh {sharpe(pd.Series(nn)):.2f}', end='  ')
print()

# ─────────────────────────────────────────────────────────────────
# 4. 보너스: 온라인 conformal 90% 구간 coverage (2026)
# ─────────────────────────────────────────────────────────────────
def qhat(scores, a=ALPHA):
    k = min(int(np.ceil((len(scores) + 1) * (1 - a))), len(scores))
    return float(np.sort(scores)[k - 1])
W = 125
buf = list(np.maximum(q05c - ycal, ycal - q95c))   # cal(2025H2) conformity로 초기화
on_lo = np.empty(len(yte)); on_hi = np.empty(len(yte))
for i in range(len(yte)):
    Qo = qhat(np.array(buf[-W:]))
    on_lo[i] = q05t[i] - Qo; on_hi[i] = q95t[i] + Qo
    buf.append(max(q05t[i] - yte[i], yte[i] - q95t[i]))
cov = float(((yte >= on_lo) & (yte <= on_hi)).mean())
width = float(np.mean(on_hi - on_lo))
excl0 = float(((on_lo > 0) | (on_hi < 0)).mean())
print(f'\n[구간 (온라인 conformal)] 2026 coverage={cov:.3f} (목표 0.90)  폭={width:.1f}bp  0배제율={excl0:.3f}')

# ─────────────────────────────────────────────────────────────────
# 5. 저장
# ─────────────────────────────────────────────────────────────────
bt_out = bt.copy()
bt_out['gross'] = gross; bt_out['net'] = net; bt_out['on_lo'] = on_lo; bt_out['on_hi'] = on_hi
bt_out.to_csv(REP / 'live_oos_2026_xgb.csv', index=False)
print(f'\nSaved {REP.relative_to(ROOT)}/live_oos_2026_xgb.csv  {bt_out.shape}')

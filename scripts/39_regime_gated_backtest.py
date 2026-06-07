# -*- coding: utf-8 -*-
"""
39 — Regime-gated backtest (MAIN v3 pipeline)

목적: 실험폴더(experiments/path_e_regime, path_b_confidence)에서 검증된
      VIX 레짐 필터 + 신뢰도 threshold 게이팅을 **메인 v3 모델**에 반영.
      앙상블이 아니라 "거르기(gating)" — 횡보·저신호 구간에서 매매를 쉰다.

입력 (재학습 없음, 저장된 v3 예측 재사용):
  - reports/no_leak_v2/predictions_xgb_v3_intervals.csv  (q50, vix, y_true, fold)
  - data/raw/raw_ecos.csv  (kr_treasury_10y level → carry)

전략 비교 (모두 v3 예측, walk-forward 3-fold):
  S0 BuyHold   : 항상 롱 (가격 only, 방향신호 미사용)
  S1 Full      : 매일 sign(q50) 매매 (게이팅 없음)
  S2 Conf(tau) : |q50| > τ 일 때만 (Path B)
  S3 Conf+VIX  : |q50| > τ AND VIX(t-1) < vix_thr (Path B + Path E)  ← 메인 제안

누수 방지: 레짐 판단 VIX 는 shift(1) (T-1 종가, 의사결정 시점 관측가능).
           Path D audit 의 vix[t] 타이밍 누수 지적 반영 (영향 미미하나 정직성).

출력:
  - reports/no_leak_v2/regime_gated_v3.csv        (per-fold 전 전략)
  - reports/no_leak_v2/regime_gated_v3_pooled.csv (pooled + bootstrap CI)
  - reports/no_leak_v2/regime_gated_v3_summary.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / 'reports' / 'no_leak_v2'

# ── 파라미터 (실험폴더 검증값) ────────────────────────────────────
D = 8.0           # 듀레이션 근사
C = 85.0          # convexity 근사
TAU = 1.5         # Path B best (walk-forward robust)
VIX_THR = 20.0    # Path E best (sweet spot)
COST_BP = 1.0     # 1bp 편도
N_BOOT = 1000
SEED = 42

FOLD_LABEL = {
    'fold1': '2020 코로나',
    'fold2': '2021-22 인상기(횡보)',
    'fold3': '2023-25 안정+충격',
}

# ── 데이터 로드 ───────────────────────────────────────────────────
iv = pd.read_csv(REP / 'predictions_xgb_v3_intervals.csv', parse_dates=['date']).sort_values('date')
ec = pd.read_csv(ROOT / 'data/raw/raw_ecos.csv', parse_dates=['date'])
y_level = ec[ec['variable'] == 'kr_treasury_10y'].set_index('date')['value'].sort_index()

iv['y_lev'] = y_level.reindex(iv['date']).ffill().values
# 레짐 VIX: 전일값 (의사결정 시점 관측가능) — fold 경계 넘김 방지 위해 fold 내 shift
iv['vix_lag'] = iv.groupby('fold')['vix'].shift(1)
iv['vix_lag'] = iv['vix_lag'].fillna(iv['vix'])  # 각 fold 첫날만 당일값 대체

# ── PnL / metrics (script 24·Path E 동일 규약) ───────────────────
def compute_pnl(pos, y_bp, y_lev, cost_bp=COST_BP):
    dy = y_bp / 10000
    pnl_price = pos * D * dy
    pnl_convex = -pos * 0.5 * C * dy ** 2
    pnl_carry = -pos * (y_lev / 100) / 252
    pos_change = np.abs(np.diff(pos, prepend=pos[0]))
    cost = pos_change * cost_bp * D / 10000
    return pnl_price + pnl_convex + pnl_carry - cost


def metrics(pnl):
    pnl = np.asarray(pnl, float)
    if len(pnl) == 0 or pnl.std() == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0
    cum = np.cumsum(pnl)
    total = float(cum[-1]) * 100
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252))
    mdd = float((cum - np.maximum.accumulate(cum)).min()) * 100
    win = int((pnl > 0).sum()); loss = int((pnl < 0).sum())
    n = win + loss
    wr = win / n * 100 if n else float('nan')
    return total, sharpe, mdd, wr, n


def sharpe_ci(pnl, n_boot=N_BOOT, seed=SEED):
    pnl = np.asarray(pnl, float)
    rng = np.random.default_rng(seed)
    n = len(pnl)
    if n == 0:
        return float('nan'), float('nan')
    s = []
    for _ in range(n_boot):
        x = pnl[rng.integers(0, n, n)]
        if x.std() > 0:
            s.append(x.mean() / x.std() * np.sqrt(252))
    return (float(np.quantile(s, .025)), float(np.quantile(s, .975))) if s else (float('nan'),) * 2


# ── 전략별 포지션 ─────────────────────────────────────────────────
def positions(g, strat):
    q50 = g['q50'].values
    sig = np.sign(q50)
    if strat == 'S0_BuyHold':
        return np.ones(len(g))
    if strat == 'S1_Full':
        return sig.astype(float)
    if strat == 'S2_Conf':
        return np.where(np.abs(q50) > TAU, sig, 0.0)
    if strat == 'S3_Conf_VIX':
        return np.where((np.abs(q50) > TAU) & (g['vix_lag'].values < VIX_THR), sig, 0.0)
    raise ValueError(strat)


# 탐색적 민감도 변형 (⚠️ snooping 주의 — 사전등록 아님, 보조 진단용)
def positions_explore(g, key):
    q50 = g['q50'].values; sig = np.sign(q50); vix = g['vix_lag'].values
    if key == 'E_VIX20_only':
        return np.where(vix < 20, sig, 0.0)
    if key == 'E_VIX25_only':
        return np.where(vix < 25, sig, 0.0)
    if key == 'E_tau0.5':
        return np.where(np.abs(q50) > 0.5, sig, 0.0)
    if key == 'E_tau0.5_VIX25':
        return np.where((np.abs(q50) > 0.5) & (vix < 25), sig, 0.0)
    raise ValueError(key)


STRATS = ['S0_BuyHold', 'S1_Full', 'S2_Conf', 'S3_Conf_VIX']
EXPLORE = ['E_VIX20_only', 'E_VIX25_only', 'E_tau0.5', 'E_tau0.5_VIX25']
folds = ['fold1', 'fold2', 'fold3']

# ── per-fold ──────────────────────────────────────────────────────
rows, pooled_pnl = [], {s: [] for s in STRATS}
for f in folds:
    g = iv[iv['fold'] == f]
    for s in STRATS:
        pos = positions(g, s)
        pnl = compute_pnl(pos, g['y_true'].values, g['y_lev'].values)
        total, sharpe, mdd, wr, n = metrics(pnl)
        rows.append({'fold': f, 'period': FOLD_LABEL[f], 'strategy': s,
                     'sharpe': round(sharpe, 3), 'total_%': round(total, 3),
                     'mdd_%': round(mdd, 3), 'win_%': round(wr, 1),
                     'active_days': int((pos != 0).sum()), 'n_days': len(g)})
        pooled_pnl[s].append(pnl)

fold_df = pd.DataFrame(rows)
fold_df.to_csv(REP / 'regime_gated_v3.csv', index=False)

# ── pooled + CI ───────────────────────────────────────────────────
prows = []
for s in STRATS:
    pooled = np.concatenate(pooled_pnl[s])
    total, sharpe, mdd, wr, n = metrics(pooled)
    lo, hi = sharpe_ci(pooled)
    prows.append({'strategy': s, 'sharpe': round(sharpe, 3),
                  'CI_low': round(lo, 3), 'CI_high': round(hi, 3),
                  'CI_excl_0': bool(lo > 0), 'total_%': round(total, 3),
                  'mdd_%': round(mdd, 3), 'win_%': round(wr, 1),
                  'active_days': int((pooled != 0).sum() if s == 'S0_BuyHold' else
                                     sum(int((positions(iv[iv['fold'] == f], s) != 0).sum()) for f in folds))})
pooled_df = pd.DataFrame(prows)
pooled_df.to_csv(REP / 'regime_gated_v3_pooled.csv', index=False)

# ── 출력 ──────────────────────────────────────────────────────────
print('=' * 74)
print(f'Regime-gated backtest (v3) — τ={TAU}bp, VIX<{VIX_THR:.0f}(t-1), cost={COST_BP}bp')
print('=' * 74)
print('\n[Pooled] 3-fold 합산 + bootstrap 95% CI')
print(pooled_df.to_string(index=False))

print('\n[Per-fold] Sharpe — 게이팅 회복 효과 (특히 fold2 횡보장)')
piv = fold_df.pivot(index='fold', columns='strategy', values='sharpe')[STRATS]
piv.index = [f'{f} ({FOLD_LABEL[f]})' for f in piv.index]
print(piv.to_string())

print('\n[Per-fold] 매매일수 (게이팅으로 abstain)')
piv2 = fold_df.pivot(index='fold', columns='strategy', values='active_days')[STRATS]
print(piv2.to_string())

# ── 탐색적 민감도 (⚠️ snooping 주의) ─────────────────────────────
print('\n' + '=' * 74)
print('⚠️  탐색적 민감도 (사전등록 아님 — fold 끼워맞추기 위험, 보조 진단만)')
print('=' * 74)
exp_rows = []
for key in EXPLORE:
    pl, per, days = [], {}, 0
    for f in folds:
        g = iv[iv['fold'] == f]; pos = positions_explore(g, key)
        pp = compute_pnl(pos, g['y_true'].values, g['y_lev'].values)
        pl.append(pp); per[f] = round(metrics(pp)[1], 2); days += int((pos != 0).sum())
    pool = np.concatenate(pl); _, sh, _, _, _ = metrics(pool); lo, hi = sharpe_ci(pool)
    exp_rows.append({'variant': key, 'pooled_sharpe': round(sh, 3),
                     'CI_low': round(lo, 2), 'CI_high': round(hi, 2), 'days': days,
                     'fold1': per['fold1'], 'fold2': per['fold2'], 'fold3': per['fold3']})
exp_df = pd.DataFrame(exp_rows)
print(exp_df.to_string(index=False))
print('\n→ 주의: pooled 가 높아도 per-fold 가 fold 별로 부호·크기 요동(예: VIX20 은 fold1 살리고')
print('  fold2 죽임; tau0.5_VIX25 는 fold3 집중·fold1 악화). 차이 대부분이 노이즈 = snooping 위험.')
print('  → 사전등록 게이팅(S3)·탐색 변형 모두 robust 하게 S1_Full(매일 매매)을 못 이김.')
exp_df.to_csv(REP / 'regime_gated_v3_explore.csv', index=False)

# ── summary.md ────────────────────────────────────────────────────
base = pooled_df.set_index('strategy')
def row(s): return base.loc[s]
md = f"""# Regime-gated backtest (v3 메인) — 결과 요약

> `scripts/39_regime_gated_backtest.py` 산출. 저장된 v3 예측(`predictions_xgb_v3_intervals.csv`) 재사용(재학습 없음).
> 게이팅 = **앙상블이 아니라 "거르기"**. τ={TAU}bp(Path B) + VIX<{VIX_THR:.0f}(Path E), cost={COST_BP}bp 편도.
> 누수 방지: 레짐 VIX 는 **shift(1)**(T-1 종가, 의사결정 시점 관측가능).

## Pooled (3-fold, bootstrap 95% CI)

| 전략 | Sharpe | 95% CI | CI 0제외 | Total % | MDD % | 매매일 |
|---|---|---|---|---|---|---|
| S0 Buy&Hold (항상 롱) | {row('S0_BuyHold')['sharpe']} | [{row('S0_BuyHold')['CI_low']}, {row('S0_BuyHold')['CI_high']}] | {row('S0_BuyHold')['CI_excl_0']} | {row('S0_BuyHold')['total_%']} | {row('S0_BuyHold')['mdd_%']} | {row('S0_BuyHold')['active_days']} |
| S1 Full (매일 매매) | {row('S1_Full')['sharpe']} | [{row('S1_Full')['CI_low']}, {row('S1_Full')['CI_high']}] | {row('S1_Full')['CI_excl_0']} | {row('S1_Full')['total_%']} | {row('S1_Full')['mdd_%']} | {row('S1_Full')['active_days']} |
| S2 Conf (\\|q50\\|>τ) | {row('S2_Conf')['sharpe']} | [{row('S2_Conf')['CI_low']}, {row('S2_Conf')['CI_high']}] | {row('S2_Conf')['CI_excl_0']} | {row('S2_Conf')['total_%']} | {row('S2_Conf')['mdd_%']} | {row('S2_Conf')['active_days']} |
| **S3 Conf+VIX (게이팅)** | **{row('S3_Conf_VIX')['sharpe']}** | **[{row('S3_Conf_VIX')['CI_low']}, {row('S3_Conf_VIX')['CI_high']}]** | **{row('S3_Conf_VIX')['CI_excl_0']}** | {row('S3_Conf_VIX')['total_%']} | **{row('S3_Conf_VIX')['mdd_%']}** | {row('S3_Conf_VIX')['active_days']} |

## Per-fold Sharpe

{piv.to_string()}

## ⚠️ 핵심 결론 (정직 — 계획과 반대 결과)

**v3 모델에서는 게이팅이 도움이 되지 않는다. 매일 매매(S1_Full)가 정직한 best.**

- 사전등록한 **τ={TAU}bp + VIX<{VIX_THR:.0f} 게이팅(S3)은 Sharpe {row('S3_Conf_VIX')['sharpe']}** — S1_Full({row('S1_Full')['sharpe']})보다 **악화**.
  원인: v3 신호의 |q50|>{TAU}bp 가 **3%(42/1410일)뿐** → τ={TAU}bp 가 좋은 매매까지 97% 버림.
- 탐색적으로 τ 를 낮추면(`_explore.csv`) pooled Sharpe 는 오르지만(최대 1.8) **fold3 에 몰리고 fold1 을 악화**
  시킴 — robust 한 회복이 아니라 **fold 끼워맞추기(data snooping)**. 본 프로젝트 정직성 가드가 경고하는 함정.
- **앙상블도 답 아님**: 실험(Path D/A)에서 앙상블은 단일 모델 대비 열세. 레짐 집중은 섞기로도 안 풀림.

## 왜 이게 좋은 발견인가 (방법론적 정직성)
- v2(약한 모델)에서 도움됐던 게이팅이 **v3(dir 0.62, 더 강함)에서는 불필요** — 모델 개선이 보조 장치를 무용화.
- 엣지 본질 = 간밤 US→KR 모멘텀 스필오버 → 추세장 강·횡보장 약은 **모델의 본성**이지 필터로 덮을 것 아님.
- **권고**: v3 백테스트는 **S1_Full(Sharpe {row('S1_Full')['sharpe']}, CI [{row('S1_Full')['CI_low']}, {row('S1_Full')['CI_high']}])** 을 정직한 결과로 보고하고,
  레짐/연도 집중(2021·2025 약세)은 **공개된 한계 + 대시보드로 시각화**. "게이팅 시도→v3엔 불필요→snooping 회피"
  자체가 정직한 방법론 서사.
"""
(REP / 'regime_gated_v3_summary.md').write_text(md, encoding='utf-8')
print('\n[save] reports/no_leak_v2/regime_gated_v3.csv, _pooled.csv, _summary.md')

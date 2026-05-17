"""
25_honest_walkforward.py — 점검 발견 P0 이슈 정정판 walk-forward

수정 사항 (vs 16_walkforward.py):
  [P0-1] Per-fold HP grid search (기존: single-split HP 재사용 → 미래정보 누수)
  [P0-2] cal ∩ val = ∅ 로 재정의 (기존: cal ⊂ val → early stopping leak)
  [P0-3] 3 시드 평균 (기존: seed=42 단일)
  [P1-3] dir_acc bootstrap CI (block bootstrap, B=1000)

수정 폴드 정의:
  fold1: train 2010-01~2017-12, val 2018-01~2019-06, cal 2019-07~12, test 2020
  fold2: train 2010-01~2019-12, val 2020-01~06,      cal 2020-07~12, test 2021-22
  fold3: train 2010-01~2021-12, val 2022-01~06,      cal 2022-07~12, test 2023-25

  → 각 fold 내 val/cal/test 모두 disjoint (DM·CQR exchangeability 보장)

출력: reports/no_leak_v2_honest/
  honest_xgb_eval.csv
  honest_walkforward_summary.md
  honest_dm_pool_bootstrap.csv
  honest_grid_per_fold.csv
"""
from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import t as t_dist
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2_honest'
REPORT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
SEEDS = [42, 123, 2024]

# [P0-2] cal disjoint from val
FOLDS = [
    {
        'name': 'fold1',
        'train': ('2010-01-01', '2017-12-31'),
        'val':   ('2018-01-01', '2019-06-30'),
        'cal':   ('2019-07-01', '2019-12-31'),
        'test':  ('2020-01-01', '2020-12-31'),
    },
    {
        'name': 'fold2',
        'train': ('2010-01-01', '2019-12-31'),
        'val':   ('2020-01-01', '2020-06-30'),
        'cal':   ('2020-07-01', '2020-12-31'),
        'test':  ('2021-01-01', '2022-12-31'),
    },
    {
        'name': 'fold3',
        'train': ('2010-01-01', '2021-12-31'),
        'val':   ('2022-01-01', '2022-06-30'),
        'cal':   ('2022-07-01', '2022-12-31'),
        'test':  ('2023-01-01', '2025-12-31'),
    },
]

# [P0-1] per-fold grid (작게: 2x2 = 4 combos x 3 quantile = 12 fit/fold)
GRID = {
    'max_depth':     [4, 6],
    'learning_rate': [0.03, 0.05],
    'n_estimators':  [600],  # early stopping
}

print('=' * 72)
print('25_honest_walkforward.py — P0 이슈 정정 honest 재실행')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date']).sort_index()
FEATURE_COLS = [c for c in df.columns if c != 'delta_y_bp']
print(f'[load] shape={df.shape}, features={len(FEATURE_COLS)}')


# ─────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────
def pinball(y, p, q):
    diff = y - p
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def dir_acc(y, p):
    mask = (np.sign(p) != 0) & (np.sign(y) != 0)
    if mask.sum() == 0:
        return float('nan')
    return float((np.sign(p[mask]) == np.sign(y[mask])).mean())


def sort_qs(p):
    arr = np.column_stack([p[q] for q in QUANTILES])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate(QUANTILES)}


def eval_q(p, y, label):
    out = {'split': label}
    for q in QUANTILES:
        out[f'pinball_q{int(q*100):02d}'] = pinball(y, p[q], q)
    out['coverage_90'] = float(np.mean((y >= p[0.05]) & (y <= p[0.95])))
    out['sharpness_bp'] = float(np.mean(p[0.95] - p[0.05]))
    err = y - p[0.5]
    out['rmse_q50_bp'] = float(np.sqrt(np.mean(err ** 2)))
    out['dir_acc_q50'] = dir_acc(y, p[0.5])
    return out


def dm_test_hln(e1, e2, lag=None):
    d = e1 - e2
    n = len(d)
    if n < 10:
        return float('nan'), float('nan'), 0
    if lag is None:
        lag = max(1, int(4 * (n / 100) ** (2 / 9)))  # Newey-West rule
    d_mean = float(d.mean())
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, lag + 1):
        wk = 1.0 - k / (lag + 1)
        cov = float(np.mean((d[:-k] - d_mean) * (d[k:] - d_mean)))
        var += 2.0 * wk * cov
    var = max(var, 1e-12)
    dm = d_mean / np.sqrt(var / n)
    h = 1
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln_factor
    p_val = float(2.0 * (1.0 - t_dist.cdf(abs(dm_hln), df=n - 1)))
    return float(dm_hln), p_val, lag


def block_bootstrap_dir_ci(y, p, block=10, B=1000, alpha=0.05, seed=42):
    """dir_acc 의 block bootstrap CI ([P1-3])"""
    rng = np.random.default_rng(seed)
    n = len(y)
    mask = (np.sign(p) != 0) & (np.sign(y) != 0)
    correct = (np.sign(p[mask]) == np.sign(y[mask])).astype(float)
    n_eff = len(correct)
    if n_eff < block * 2:
        return float('nan'), float('nan')
    n_blocks = n_eff // block + 1
    stats = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, n_eff - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n_eff]
        stats[b] = correct[idx].mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fit_xgb(q, params, X_tr, y_tr, X_val, y_val, seed):
    m = xgb.XGBRegressor(
        objective='reg:quantileerror',
        quantile_alpha=q,
        n_estimators=GRID['n_estimators'][0],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        early_stopping_rounds=50,
        verbosity=0, tree_method='hist', random_state=seed,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return m


# ─────────────────────────────────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────────────────────────────────
all_eval = []
all_grid = []
all_dm = []
pooled_y, pooled_p_byseed = [], {s: [] for s in SEEDS}

for fold in FOLDS:
    name = fold['name']
    print(f'\n{"="*72}\n{name}: tr {fold["train"]} | val {fold["val"]} | cal {fold["cal"]} | test {fold["test"]}')
    print('=' * 72)

    def sl(p): return df.loc[fold[p][0]:fold[p][1]]

    X_tr_raw = sl('train')[FEATURE_COLS]
    X_va_raw = sl('val')[FEATURE_COLS]
    X_ca_raw = sl('cal')[FEATURE_COLS]
    X_te_raw = sl('test')[FEATURE_COLS]
    y_tr = sl('train')['delta_y_bp']
    y_va = sl('val')['delta_y_bp']
    y_ca = sl('cal')['delta_y_bp']
    y_te = sl('test')['delta_y_bp']

    # [P0-2] disjoint check
    assert sl('val').index.intersection(sl('cal').index).empty, 'val/cal must be disjoint'
    assert sl('cal').index.intersection(sl('test').index).empty, 'cal/test must be disjoint'
    print(f'  shapes: tr={X_tr_raw.shape} val={X_va_raw.shape} cal={X_ca_raw.shape} te={X_te_raw.shape}')

    scaler = RobustScaler().fit(X_tr_raw)
    def s(X): return pd.DataFrame(scaler.transform(X), index=X.index, columns=FEATURE_COLS)
    X_tr, X_va, X_ca, X_te = s(X_tr_raw), s(X_va_raw), s(X_ca_raw), s(X_te_raw)

    # ────────────────────────────────────────────────────────
    # [P0-1] per-fold grid search (분위수별 best HP, val pinball 최소)
    # ────────────────────────────────────────────────────────
    print(f'  [P0-1] per-fold grid search ({len(GRID["max_depth"]) * len(GRID["learning_rate"])} combos × 3 q)')
    best_per_q = {}
    for q in QUANTILES:
        best_v, best_p, best_m = float('inf'), None, None
        for md, lr in product(GRID['max_depth'], GRID['learning_rate']):
            m = fit_xgb(q, {'max_depth': md, 'learning_rate': lr},
                        X_tr.values, y_tr.values, X_va.values, y_va.values, seed=42)
            pred_v = m.predict(X_va.values)
            vp = pinball(y_va.values, pred_v, q)
            all_grid.append({'fold': name, 'q': q, 'max_depth': md, 'lr': lr,
                             'val_pinball': vp,
                             'best_iter': int(m.best_iteration or GRID['n_estimators'][0])})
            if vp < best_v:
                best_v, best_p, best_m = vp, {'max_depth': md, 'learning_rate': lr}, m
        best_per_q[q] = best_p
        print(f'    q={q} best={best_p}  val_pinball={best_v:.4f}')

    # ────────────────────────────────────────────────────────
    # [P0-3] 3 시드 평균
    # ────────────────────────────────────────────────────────
    for seed in SEEDS:
        preds_te = {}
        preds_ca = {}
        for q in QUANTILES:
            m = fit_xgb(q, best_per_q[q],
                        X_tr.values, y_tr.values, X_va.values, y_va.values, seed=seed)
            preds_te[q] = m.predict(X_te.values)
            preds_ca[q] = m.predict(X_ca.values)
        preds_te = sort_qs(preds_te)
        preds_ca = sort_qs(preds_ca)

        # CQR (cal 이 val 과 disjoint → conformity exchangeable)
        ca_y = y_ca.values
        sc = np.maximum(preds_ca[0.05] - ca_y, ca_y - preds_ca[0.95])
        n_c = len(sc)
        k = min(int(np.ceil((n_c + 1) * (1 - ALPHA))), n_c)
        Q_hat = float(np.sort(sc)[k - 1])

        preds_te_cqr = {
            0.05: preds_te[0.05] - Q_hat,
            0.5:  preds_te[0.5],
            0.95: preds_te[0.95] + Q_hat,
        }

        raw_e = eval_q(preds_te, y_te.values, 'test')
        raw_e.update({'fold': name, 'seed': seed, 'stage': 'raw', 'Q_hat': Q_hat})
        cqr_e = eval_q(preds_te_cqr, y_te.values, 'test')
        cqr_e.update({'fold': name, 'seed': seed, 'stage': 'CQR', 'Q_hat': Q_hat})
        all_eval.extend([raw_e, cqr_e])
        print(f'    seed={seed}: dir={raw_e["dir_acc_q50"]:.4f}, cov_raw={raw_e["coverage_90"]:.4f} → CQR={cqr_e["coverage_90"]:.4f} (Q={Q_hat:+.3f})')

        # pool
        pooled_p_byseed[seed].append(preds_te[0.5])
        if seed == SEEDS[0]:
            pooled_y.append(y_te.values)

# ─────────────────────────────────────────────────────────────────────────
# Pooled + Bootstrap CI + DM
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('Pooled (3 fold test 합쳐) + Bootstrap CI + DM')
print('=' * 72)

y_pool = np.concatenate(pooled_y)
n_pool = len(y_pool)
print(f'  pooled n = {n_pool}')

dir_byseed, ci_byseed = {}, {}
dm_byseed = {}
for seed in SEEDS:
    p_pool = np.concatenate(pooled_p_byseed[seed])
    d = dir_acc(y_pool, p_pool)
    lo, hi = block_bootstrap_dir_ci(y_pool, p_pool, block=10, B=1000, seed=seed)
    err_x = (y_pool - p_pool) ** 2
    err_n = y_pool ** 2
    dm, pv, used_lag = dm_test_hln(err_x, err_n)
    dir_byseed[seed] = d
    ci_byseed[seed] = (lo, hi)
    dm_byseed[seed] = (dm, pv, used_lag)
    print(f'  seed={seed}: dir={d:.4f} [{lo:.4f}, {hi:.4f}], DM={dm:.3f} (p={pv:.4f}, lag={used_lag})')

mean_dir = float(np.mean(list(dir_byseed.values())))
std_dir = float(np.std(list(dir_byseed.values()), ddof=1))
print(f'\n  Pooled dir_acc (3 seed avg): {mean_dir:.4f} ± {std_dir:.4f}')

# 표본평균에 대한 conservative CI: seed별 CI 의 min/max 합집합
ci_lo = min(c[0] for c in ci_byseed.values())
ci_hi = max(c[1] for c in ci_byseed.values())
print(f'  Pooled dir 95% CI (union of seeds): [{ci_lo:.4f}, {ci_hi:.4f}]')

# ─────────────────────────────────────────────────────────────────────────
# 저장
# ─────────────────────────────────────────────────────────────────────────
pd.DataFrame(all_eval).to_csv(REPORT_DIR / 'honest_xgb_eval.csv', index=False)
pd.DataFrame(all_grid).to_csv(REPORT_DIR / 'honest_grid_per_fold.csv', index=False)

dm_rows = []
for seed in SEEDS:
    dm, pv, lag = dm_byseed[seed]
    lo, hi = ci_byseed[seed]
    dm_rows.append({
        'seed': seed,
        'dir_acc_pooled': dir_byseed[seed],
        'dir_ci_lo': lo, 'dir_ci_hi': hi,
        'DM_HLN': dm, 'p_value': pv, 'nw_lag': lag,
        'bonf_alpha_0.0167': 'OK' if pv < 0.0167 else 'NO',
    })
pd.DataFrame(dm_rows).to_csv(REPORT_DIR / 'honest_dm_pool_bootstrap.csv', index=False)

# Per-fold per-seed CQR summary
ev_df = pd.DataFrame(all_eval)
cqr_df = ev_df[ev_df['stage'] == 'CQR']
fold_mean = cqr_df.groupby('fold').agg(
    dir_mean=('dir_acc_q50', 'mean'), dir_std=('dir_acc_q50', 'std'),
    cov_mean=('coverage_90', 'mean'), cov_std=('coverage_90', 'std'),
    Q_mean=('Q_hat', 'mean'),
).round(4)
print('\n=== Per-fold CQR (3-seed avg) ===')
print(fold_mean.to_string())

# 마크다운
md = [
    '# Honest walk-forward (25_honest_walkforward.py)',
    '',
    '> P0-1/P0-2/P0-3/P1-3 정정판. 기존 16_walkforward.py 결과와 직접 비교용.',
    '',
    '## 정정 내역',
    '- [P0-1] Per-fold HP grid search (기존: single-split HP 고정 → 미래정보 누수)',
    '- [P0-2] cal/val/test 모두 disjoint (기존: cal ⊂ val → CQR exchangeability 위반)',
    '- [P0-3] 3 시드 평균 (기존: seed=42 단일)',
    '- [P1-3] dir_acc block bootstrap 95% CI (기존: 점추정만)',
    '',
    '## Fold (honest)',
    '',
    '| fold | train | val | cal | test |',
    '|---|---|---|---|---|',
]
for f in FOLDS:
    md.append(f"| {f['name']} | {f['train'][0]}~{f['train'][1]} | {f['val'][0]}~{f['val'][1]} | {f['cal'][0]}~{f['cal'][1]} | {f['test'][0]}~{f['test'][1]} |")

md.extend([
    '',
    '## Per-fold CQR (3 seed avg)',
    '',
    '| fold | dir_acc | Coverage 90% | Q_hat (bp) |',
    '|---|---|---|---|',
])
for fname, row in fold_mean.iterrows():
    md.append(f"| {fname} | {row['dir_mean']:.4f} ± {row['dir_std']:.4f} | {row['cov_mean']:.4f} ± {row['cov_std']:.4f} | {row['Q_mean']:+.3f} |")

md.extend([
    '',
    '## Pooled (3 fold test) + Bootstrap CI',
    '',
    f'- dir_acc (3 seed avg): **{mean_dir:.4f} ± {std_dir:.4f}**',
    f'- 95% block bootstrap CI (B=1000, block=10, union of seeds): **[{ci_lo:.4f}, {ci_hi:.4f}]**',
    f'- 학술 합격선 0.53 통계 우위: **{"✅ 입증" if ci_lo > 0.53 else "❌ CI 가 0.53 포함"}**',
    '',
    '| seed | dir_acc | CI 95% | DM_HLN | p-value | lag | Bonf α=0.0167 |',
    '|---|---|---|---|---|---|---|',
])
for seed in SEEDS:
    dm, pv, lag = dm_byseed[seed]
    lo, hi = ci_byseed[seed]
    bonf = '✅' if pv < 0.0167 else '❌'
    md.append(f"| {seed} | {dir_byseed[seed]:.4f} | [{lo:.4f}, {hi:.4f}] | {dm:.3f} | {pv:.4f} | {lag} | {bonf} |")

md.extend([
    '',
    '## 비교 — 기존 v2 (16_walkforward) vs honest (25)',
    '',
    '| 지표 | v2 기존 (16) | honest (25) | 변화 |',
    '|---|---|---|---|',
    f'| Pooled dir_acc | 0.6178 (single seed) | {mean_dir:.4f} ± {std_dir:.4f} (3-seed) | {(mean_dir-0.6178)*100:+.2f}%p |',
])

md_path = REPORT_DIR / 'honest_walkforward_summary.md'
md_path.write_text('\n'.join(md), encoding='utf-8')
print(f'\n[save] {md_path.relative_to(PROJECT_ROOT)}')
print(f'[save] reports/no_leak_v2_honest/honest_xgb_eval.csv')
print(f'[save] reports/no_leak_v2_honest/honest_grid_per_fold.csv')
print(f'[save] reports/no_leak_v2_honest/honest_dm_pool_bootstrap.csv')
print('\n=== 25 honest walkforward 완료 ===')

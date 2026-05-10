"""
20_verify_package_a_direction.py — 패키지 A' (ACI + rolling vol) 방향 검증

verification-before-completion: 추측 금지, 실제 데이터로 측정.

검증 4가지:
  1. v2 features 에 rolling vol/std 가 이미 있는지 → 추가 필요성 판단
  2. fold1 (2020 코로나) 시점에 모델이 실제로 어떻게 행동했는지 정량 분석
  3. ACI 시뮬레이션 — 우리 데이터에서 정말 Coverage 회복하는가
  4. rolling vol 입력 추가의 ablation 효과 (1 fold 만 빠르게)

출력: reports/no_leak_v2/package_a_verification.md
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
REPORT_V2 = PROJECT_ROOT / 'reports' / 'no_leak_v2'
MODELS_DIR = PROJECT_ROOT / 'models'

print('=' * 72)
print('20_verify_package_a_direction.py — 패키지 A 방향 검증')
print('=' * 72)

evidence = []  # 모든 검증 증거를 마크다운으로 모음


def log(msg):
    print(msg)
    evidence.append(msg)


# ─────────────────────────────────────────────────────────────────────────
# 검증 1: v2 features 에 rolling vol/std 가 이미 있는지
# ─────────────────────────────────────────────────────────────────────────
log('\n## 검증 1: v2 features 에 rolling vol/std 존재 여부')
log('')

v2 = pd.read_csv(DATA_DIR / 'processed' / 'features_v2_no_leak.csv',
                 index_col='date', parse_dates=['date'])

all_cols = v2.columns.tolist()
rmean_cols = [c for c in all_cols if 'rmean' in c]
rstd_cols = [c for c in all_cols if 'rstd' in c]
crisis_col_present = 'crisis_dummy' in all_cols
spread_present = 'spread_10y_t1' in all_cols
delta_us10y = 'delta_us10y_t1' in all_cols

log(f'- 전체 컬럼 수: {len(all_cols)}')
log(f'- rolling mean (rmean) 컬럼 수: {len(rmean_cols)}')
log(f'- rolling std (rstd) 컬럼 수: {len(rstd_cols)}')
log(f'- crisis_dummy 존재: {crisis_col_present}')
log(f'- spread_10y_t1 존재: {spread_present}')
log(f'- delta_us10y_t1 존재: {delta_us10y}')
log('')

# 타겟(kr_treasury_10y) 의 rolling std 가 있는지
target_rstd = [c for c in rstd_cols if 'kr_treasury_10y' in c]
log(f'- 타겟 kr_treasury_10y 의 rolling std 컬럼: {target_rstd}')
log('')

# 만약 타겟 rstd 가 없다면 진짜 vol of target 정보 부재
if target_rstd:
    log(f'  → ✅ 타겟 변동성 직접 입력 이미 있음 (rolling std). 추가 필요 약함.')
else:
    log(f'  → ❌ 타겟 변동성 직접 입력 없음. crisis_dummy 만 있음 (binary).')
    log(f'    rolling vol(continuous) 추가 시 첫 충격 보호 강화 가능.')

# 다른 변수의 rstd도 확인
log('\nrolling std 컬럼 sample (앞 8개):')
for c in rstd_cols[:8]:
    log(f'  - {c}')

# ─────────────────────────────────────────────────────────────────────────
# 검증 2: fold1 (2020 코로나) 시점 모델 실제 행동 분석
# ─────────────────────────────────────────────────────────────────────────
log('\n## 검증 2: fold1 (2020 코로나) 시점 모델 행동 분석')
log('')

# Fold1 정의 (16_walkforward 와 동일)
FOLD1 = {
    'train': ('2010-01-01', '2017-12-31'),
    'val':   ('2018-01-01', '2019-12-31'),
    'cal':   ('2019-07-01', '2019-12-31'),
    'test':  ('2020-01-01', '2020-12-31'),
}

FEATURE_COLS = [c for c in v2.columns if c != 'delta_y_bp']

def sl(period):
    s, e = FOLD1[period]
    return v2.loc[s:e]

X_tr_raw = sl('train')[FEATURE_COLS]
X_cal_raw = sl('cal')[FEATURE_COLS]
X_val_raw = sl('val')[FEATURE_COLS]
X_te_raw = sl('test')[FEATURE_COLS]

scaler = RobustScaler().fit(X_tr_raw)
def s_(X):
    return pd.DataFrame(scaler.transform(X), index=X.index, columns=FEATURE_COLS)

X_tr = s_(X_tr_raw); X_val = s_(X_val_raw); X_cal = s_(X_cal_raw); X_te = s_(X_te_raw)
y_tr = sl('train')['delta_y_bp']
y_val = sl('val')['delta_y_bp']
y_cal = sl('cal')['delta_y_bp']
y_te = sl('test')['delta_y_bp']

# 빠른 XGBoost q05/q50/q95 학습 (best params 재사용)
XGB_BEST = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 200},
    0.5:  {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 400},
    0.95: {'max_depth': 4, 'learning_rate': 0.1,  'n_estimators': 100},
}

print('  fold1 XGB q05/q50/q95 학습...')
preds_te = {}
preds_cal = {}
for q in [0.05, 0.5, 0.95]:
    p = XGB_BEST[q]
    m = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=q,
        n_estimators=p['n_estimators'], max_depth=p['max_depth'],
        learning_rate=p['learning_rate'],
        early_stopping_rounds=50, verbosity=0, tree_method='hist', random_state=42,
    )
    m.fit(X_tr.values, y_tr.values, eval_set=[(X_val.values, y_val.values)], verbose=False)
    preds_te[q] = m.predict(X_te.values)
    preds_cal[q] = m.predict(X_cal.values)

# Sort
def sort_q(p):
    arr = np.column_stack([p[q] for q in [0.05, 0.5, 0.95]])
    arr_s = np.sort(arr, axis=1)
    return {q: arr_s[:, i] for i, q in enumerate([0.05, 0.5, 0.95])}

p_te = sort_q(preds_te)
p_cal = sort_q(preds_cal)

# CQR Q_hat
sc = np.maximum(p_cal[0.05] - y_cal.values, y_cal.values - p_cal[0.95])
n_c = len(sc)
k = min(int(np.ceil((n_c + 1) * 0.9)), n_c)
Q_hat = float(np.sort(sc)[k - 1])

p_te_cqr = {0.05: p_te[0.05] - Q_hat, 0.5: p_te[0.5], 0.95: p_te[0.95] + Q_hat}

# 시점별 분석 — 코로나 첫 충격 (2020-02-21~04-10)
te_dates = sl('test').index
y_te_arr = y_te.values
in_covid = (te_dates >= '2020-02-21') & (te_dates <= '2020-04-10')
covid_dates = te_dates[in_covid]
covid_y = y_te_arr[in_covid]
covid_q05 = p_te_cqr[0.05][in_covid]
covid_q50 = p_te_cqr[0.5][in_covid]
covid_q95 = p_te_cqr[0.95][in_covid]
covid_hit = (covid_y >= covid_q05) & (covid_y <= covid_q95)
covid_width = covid_q95 - covid_q05

log(f'- 코로나 충격기 (2020-02-21 ~ 2020-04-10) 분석:')
log(f'  - 일수: {len(covid_dates)} 영업일')
log(f'  - 실제 |Δy| 평균: {np.mean(np.abs(covid_y)):.2f} bp')
log(f'  - 실제 |Δy| 최대: {np.max(np.abs(covid_y)):.2f} bp')
log(f'  - 평소 |Δy| 평균 (test 전체): {np.mean(np.abs(y_te_arr)):.2f} bp')
log(f'  - 모델 PI 평균 폭: {np.mean(covid_width):.2f} bp')
log(f'  - PI 평균 폭 (test 전체): {np.mean(p_te_cqr[0.95] - p_te_cqr[0.05]):.2f} bp')
log(f'  - **코로나 hit rate: {covid_hit.mean()*100:.1f}%** (목표 90%)')
log(f'  - 충격 첫날 (2020-02-21~03-09, 12일) hit rate: {covid_hit[:12].mean()*100:.1f}%')
log(f'  - 후반 (3-10 이후) hit rate: {covid_hit[12:].mean()*100:.1f}%')
log('')

# 일별 분석 (앞 10일)
log(f'  코로나 충격 시점별 (앞 12일):')
log(f'  | 날짜 | 실제 Δy (bp) | q05 | q50 | q95 | 폭 | hit |')
log(f'  |---|---|---|---|---|---|---|')
for i in range(min(12, len(covid_dates))):
    d = pd.Timestamp(covid_dates[i]).strftime('%Y-%m-%d')
    log(f'  | {d} | {covid_y[i]:+.2f} | {covid_q05[i]:+.2f} | {covid_q50[i]:+.2f} '
        f'| {covid_q95[i]:+.2f} | {covid_width[i]:.2f} | {"✅" if covid_hit[i] else "❌"} |')
log('')

# ─────────────────────────────────────────────────────────────────────────
# 검증 3: ACI 시뮬레이션 — 우리 데이터에서 효과 측정
# ─────────────────────────────────────────────────────────────────────────
log('\n## 검증 3: ACI (Adaptive Conformal Inference) 시뮬레이션')
log('')

# 단순 ACI: γ = 0.005, α_0 = 0.10
def aci_sim(p_q05, p_q50, p_q95, y, alpha_0=0.10, gamma=0.005):
    """매 시점 t 에서 직전 hit/miss 로 alpha 갱신.
    PI 보정: q05 -= Q_t, q95 += Q_t. Q_t는 직전까지 conformity score 의 (1-alpha_t) 분위수.
    """
    n = len(y)
    alpha_t = alpha_0
    alphas = []
    Q_ts = []
    hits = []
    pi_low = np.zeros(n)
    pi_high = np.zeros(n)

    score_history = []  # conformity score 누적

    for t in range(n):
        # Q_t: 누적 score 의 (1-alpha_t) 분위수
        if len(score_history) > 5:
            sorted_s = np.sort(score_history)
            k = min(int(np.ceil((len(sorted_s) + 1) * (1 - alpha_t))), len(sorted_s))
            k = max(1, k)
            Q_t = float(sorted_s[k - 1])
        else:
            Q_t = 0.0  # 초기에는 보정 없음
        Q_ts.append(Q_t)
        alphas.append(alpha_t)

        # 보정된 PI
        pi_low[t] = p_q05[t] - Q_t
        pi_high[t] = p_q95[t] + Q_t

        # 실제 관측 → score 갱신, alpha 갱신
        score_t = max(p_q05[t] - y[t], y[t] - p_q95[t])  # raw conformity
        score_history.append(score_t)

        is_hit = (y[t] >= pi_low[t]) and (y[t] <= pi_high[t])
        hits.append(is_hit)

        # ACI 갱신
        if is_hit:
            alpha_t = alpha_t - gamma * (alpha_0 / (1 - alpha_0))
        else:
            alpha_t = alpha_t + gamma
        alpha_t = max(0.001, min(0.5, alpha_t))  # safety

    return np.array(pi_low), np.array(pi_high), np.array(hits), np.array(alphas), np.array(Q_ts)


# fold1 test 에 ACI 적용
aci_low, aci_high, aci_hits, aci_alphas, aci_Q = aci_sim(
    p_te[0.05], p_te[0.5], p_te[0.95], y_te_arr, alpha_0=0.10, gamma=0.005
)

baseline_hit = float(np.mean((y_te_arr >= p_te_cqr[0.05]) & (y_te_arr <= p_te_cqr[0.95])))
aci_hit = float(aci_hits.mean())
baseline_width = float(np.mean(p_te_cqr[0.95] - p_te_cqr[0.05]))
aci_width = float(np.mean(aci_high - aci_low))

log(f'- fold1 (2020 전체) Coverage 비교:')
log(f'  - Split-CQR (현재 v2): {baseline_hit:.4f}')
log(f'  - ACI (γ=0.005): {aci_hit:.4f}')
log(f'  - 목표 0.90 대비: split-CQR {(baseline_hit-0.9)*100:+.1f}%p / ACI {(aci_hit-0.9)*100:+.1f}%p')
log('')
log(f'- fold1 PI 평균 폭 비교:')
log(f'  - Split-CQR: {baseline_width:.2f} bp')
log(f'  - ACI: {aci_width:.2f} bp')
log(f'  - ACI 가 {aci_width / baseline_width:.2f}배 (작아짐 = sharper, 커짐 = wider)')
log('')

# 코로나 충격기 ACI 효과
covid_aci_hit = aci_hits[in_covid]
log(f'- 코로나 충격기 (2020-02-21~04-10) Coverage:')
log(f'  - Split-CQR: {covid_hit.mean()*100:.1f}%')
log(f'  - ACI: {covid_aci_hit.mean()*100:.1f}%')
log(f'  - 첫 12일 ACI: {covid_aci_hit[:12].mean()*100:.1f}%')
log(f'  - 13~끝 ACI: {covid_aci_hit[12:].mean()*100:.1f}%')
log('')

# alpha 의 시간 추이 (코로나 시점)
covid_alpha_path = aci_alphas[in_covid]
log(f'- ACI alpha 추이 (코로나 시점):')
log(f'  - 시작: α_0 = {aci_alphas[0]:.4f}')
log(f'  - 코로나 진입: α = {covid_alpha_path[0]:.4f}')
log(f'  - 코로나 첫 12일 후: α = {covid_alpha_path[12] if len(covid_alpha_path)>12 else covid_alpha_path[-1]:.4f}')
log(f'  - 코로나 종료: α = {covid_alpha_path[-1]:.4f}')
log(f'  - 최대 α: {aci_alphas.max():.4f}')
log('')

# fold2 음수 Q_hat 케이스도 시뮬레이션 (간단히)
log('## 검증 3b: fold2 음수 Q̂ 문제에 ACI 적용')
log('')
log(f'(fold2 학습 생략, fold1 결과로 추론)')
log(f'- fold2 음수 Q̂ 원인: cal 변동성 << test 변동성')
log(f'- ACI 가 자동으로 α_t 증가시켜 음수 Q̂ 문제 회피')
log(f'- 단점: 첫 며칠 학습이 필요')
log('')

# ─────────────────────────────────────────────────────────────────────────
# 검증 4: rolling vol 입력 추가 ablation (1 fold 빠른 비교)
# ─────────────────────────────────────────────────────────────────────────
log('\n## 검증 4: rolling vol(타겟 변동성) 입력 추가의 효과 (fold1 ablation)')
log('')

# 현재 v2 에 타겟 자체의 rolling std 가 없을 가능성
# 추가: kr_treasury_10y rolling std 5d, 20d (단 raw 가 v2 features 에 없음)
# 우리는 features_v1_candidate.csv 에서 가져옴 (raw 타겟 보존)
raw = pd.read_csv(DATA_DIR / 'processed' / 'features_v1_candidate.csv',
                  index_col='date', parse_dates=['date']).sort_index()

# 타겟의 일별 변화량 변동성 (rolling std)
delta_y = (raw['kr_treasury_10y'].diff() * 100)  # bp
target_rstd_5 = delta_y.rolling(5).std().shift(1)  # CL-03 강제
target_rstd_20 = delta_y.rolling(20).std().shift(1)

# fold1 train 시기에만 fit, test 에 적용
common = v2.index.intersection(target_rstd_5.index)
v2_aug = v2.loc[common].copy()
v2_aug['target_dy_rstd5'] = target_rstd_5.loc[common]
v2_aug['target_dy_rstd20'] = target_rstd_20.loc[common]
v2_aug = v2_aug.dropna(subset=['target_dy_rstd5', 'target_dy_rstd20'])

FEATURE_COLS_AUG = [c for c in v2_aug.columns if c != 'delta_y_bp']

X_tr_aug = v2_aug.loc[FOLD1['train'][0]:FOLD1['train'][1], FEATURE_COLS_AUG]
X_val_aug = v2_aug.loc[FOLD1['val'][0]:FOLD1['val'][1], FEATURE_COLS_AUG]
X_cal_aug = v2_aug.loc[FOLD1['cal'][0]:FOLD1['cal'][1], FEATURE_COLS_AUG]
X_te_aug = v2_aug.loc[FOLD1['test'][0]:FOLD1['test'][1], FEATURE_COLS_AUG]

y_tr_aug = v2_aug.loc[FOLD1['train'][0]:FOLD1['train'][1], 'delta_y_bp']
y_val_aug = v2_aug.loc[FOLD1['val'][0]:FOLD1['val'][1], 'delta_y_bp']
y_cal_aug = v2_aug.loc[FOLD1['cal'][0]:FOLD1['cal'][1], 'delta_y_bp']
y_te_aug = v2_aug.loc[FOLD1['test'][0]:FOLD1['test'][1], 'delta_y_bp']

scaler_aug = RobustScaler().fit(X_tr_aug)
X_tr_aug_s = pd.DataFrame(scaler_aug.transform(X_tr_aug), index=X_tr_aug.index, columns=FEATURE_COLS_AUG)
X_val_aug_s = pd.DataFrame(scaler_aug.transform(X_val_aug), index=X_val_aug.index, columns=FEATURE_COLS_AUG)
X_cal_aug_s = pd.DataFrame(scaler_aug.transform(X_cal_aug), index=X_cal_aug.index, columns=FEATURE_COLS_AUG)
X_te_aug_s = pd.DataFrame(scaler_aug.transform(X_te_aug), index=X_te_aug.index, columns=FEATURE_COLS_AUG)

print('  fold1 with rolling vol 추가 학습...')
preds_te_aug = {}
preds_cal_aug = {}
for q in [0.05, 0.5, 0.95]:
    p = XGB_BEST[q]
    m = xgb.XGBRegressor(
        objective='reg:quantileerror', quantile_alpha=q,
        n_estimators=p['n_estimators'], max_depth=p['max_depth'],
        learning_rate=p['learning_rate'],
        early_stopping_rounds=50, verbosity=0, tree_method='hist', random_state=42,
    )
    m.fit(X_tr_aug_s.values, y_tr_aug.values,
          eval_set=[(X_val_aug_s.values, y_val_aug.values)], verbose=False)
    preds_te_aug[q] = m.predict(X_te_aug_s.values)
    preds_cal_aug[q] = m.predict(X_cal_aug_s.values)

p_te_aug = sort_q(preds_te_aug)
p_cal_aug = sort_q(preds_cal_aug)
sc_aug = np.maximum(p_cal_aug[0.05] - y_cal_aug.values, y_cal_aug.values - p_cal_aug[0.95])
n_c_aug = len(sc_aug)
k_aug = min(int(np.ceil((n_c_aug + 1) * 0.9)), n_c_aug)
Q_hat_aug = float(np.sort(sc_aug)[k_aug - 1])

p_te_cqr_aug = {
    0.05: p_te_aug[0.05] - Q_hat_aug,
    0.5: p_te_aug[0.5],
    0.95: p_te_aug[0.95] + Q_hat_aug,
}

y_te_aug_arr = y_te_aug.values

# 비교 — 원래 v2 vs v2+vol
def metrics(p_q05, p_q50, p_q95, y):
    cov = float(np.mean((y >= p_q05) & (y <= p_q95)))
    width = float(np.mean(p_q95 - p_q05))
    mask = (np.sign(p_q50) != 0) & (np.sign(y) != 0)
    da = float(np.mean(np.sign(p_q50[mask]) == np.sign(y[mask]))) if mask.sum() > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((y - p_q50) ** 2)))
    return cov, width, da, rmse

cov_b, wid_b, da_b, rm_b = metrics(p_te_cqr[0.05], p_te_cqr[0.5], p_te_cqr[0.95], y_te_arr)
cov_a, wid_a, da_a, rm_a = metrics(p_te_cqr_aug[0.05], p_te_cqr_aug[0.5], p_te_cqr_aug[0.95], y_te_aug_arr)

log(f'- fold1 (2020 코로나기 포함) ablation 결과:')
log(f'')
log(f'| 모델 | dir_acc | Coverage | Sharpness (bp) | RMSE (bp) |')
log(f'|---|---|---|---|---|')
log(f'| v2 (현재) | {da_b:.4f} | {cov_b:.4f} | {wid_b:.2f} | {rm_b:.3f} |')
log(f'| **v2 + 타겟 vol** | {da_a:.4f} | {cov_a:.4f} | {wid_a:.2f} | {rm_a:.3f} |')
log(f'| Δ | {da_a-da_b:+.4f} | {cov_a-cov_b:+.4f} | {wid_a-wid_b:+.2f} | {rm_a-rm_b:+.3f} |')
log('')

# 코로나기 별도
in_covid_aug = (y_te_aug.index >= '2020-02-21') & (y_te_aug.index <= '2020-04-10')
covid_y_aug = y_te_aug_arr[in_covid_aug]
covid_q05_aug = p_te_cqr_aug[0.05][in_covid_aug]
covid_q95_aug = p_te_cqr_aug[0.95][in_covid_aug]
covid_q50_aug = p_te_cqr_aug[0.5][in_covid_aug]
covid_hit_aug = (covid_y_aug >= covid_q05_aug) & (covid_y_aug <= covid_q95_aug)
covid_width_aug = covid_q95_aug - covid_q05_aug

log(f'- 코로나 충격기만 비교:')
log(f'  - v2 hit rate: {covid_hit.mean()*100:.1f}%, 평균 폭 {np.mean(p_te_cqr[0.95][in_covid] - p_te_cqr[0.05][in_covid]):.2f} bp')
log(f'  - v2+vol hit rate: {covid_hit_aug.mean()*100:.1f}%, 평균 폭 {np.mean(covid_width_aug):.2f} bp')
log(f'  - 첫 12일: v2 {covid_hit[:12].mean()*100:.1f}% / v2+vol {covid_hit_aug[:12].mean()*100:.1f}%')
log('')

# ─────────────────────────────────────────────────────────────────────────
# 종합 결론 — 패키지 A 방향 검증 결과
# ─────────────────────────────────────────────────────────────────────────
log('\n## 종합 결론')
log('')
log(f'### 검증 1 — rolling vol 존재 여부')
if target_rstd:
    log(f'- ✅ 타겟 rolling std 가 lag/roll 자동 생성에 있음 → 추가 필요 적음')
else:
    log(f'- ❌ 타겟 변화량의 rolling std 없음 (delta_y_bp 자체가 lag/roll 안 됨)')
    log(f'- → 타겟 vol 입력 추가가 의미 있음')
log('')

log(f'### 검증 2 — fold1 코로나 시점 모델 행동')
log(f'- 코로나기 hit rate: {covid_hit.mean()*100:.1f}% (목표 90% 대비 미달)')
log(f'- 첫 12일 hit rate: {covid_hit[:12].mean()*100:.1f}% (가장 위험)')
if covid_hit.mean() < 0.85:
    log(f'- → 모델이 충격에 대응 못함. 보완 필요.')
else:
    log(f'- → 의외로 모델이 잘 대응')
log('')

log(f'### 검증 3 — ACI 효과')
log(f'- ACI Coverage {aci_hit:.4f} vs Split-CQR {baseline_hit:.4f}')
log(f'- ACI 폭 {aci_width:.2f} bp vs Split-CQR {baseline_width:.2f} bp')
if abs(aci_hit - 0.9) < abs(baseline_hit - 0.9):
    log(f'- → ✅ ACI 가 Split-CQR 보다 90% 에 더 가까움 → 효과 있음')
else:
    log(f'- → ⚠️ ACI 가 Split-CQR 보다 멀어짐 → fold1 에서는 효과 제한적')
log('')

log(f'### 검증 4 — rolling vol 입력 추가 효과')
log(f'- dir_acc Δ: {da_a-da_b:+.4f}')
log(f'- Coverage Δ: {cov_a-cov_b:+.4f}')
log(f'- 코로나 hit rate Δ: {(covid_hit_aug.mean()-covid_hit.mean())*100:+.1f}%p')
if cov_a > cov_b and abs(cov_a - 0.9) < abs(cov_b - 0.9):
    log(f'- → ✅ rolling vol 추가가 Coverage 개선')
else:
    log(f'- → ⚠️ rolling vol 추가 효과 제한적 또는 역효과')
log('')

log(f'### 패키지 A 방향성 평가')
ok_count = 0
total_count = 4
if not target_rstd:
    log(f'- 1) rolling vol 추가 필요성: ✅ 정당')
    ok_count += 1
else:
    log(f'- 1) rolling vol 추가 필요성: ⚠️ 이미 있음 (효과 제한적)')

if covid_hit.mean() < 0.85:
    log(f'- 2) fold1 보완 필요성: ✅ 명확 (코로나 hit {covid_hit.mean()*100:.1f}%)')
    ok_count += 1
else:
    log(f'- 2) fold1 보완 필요성: ⚠️ 의외로 모델이 잘 대응')

if abs(aci_hit - 0.9) < abs(baseline_hit - 0.9):
    log(f'- 3) ACI 효과: ✅ 정량 입증')
    ok_count += 1
else:
    log(f'- 3) ACI 효과: ⚠️ fold1 에서 제한적')

if cov_a > cov_b:
    log(f'- 4) vol 입력 효과: ✅ 정량 입증')
    ok_count += 1
else:
    log(f'- 4) vol 입력 효과: ⚠️ 이번 fold1 에서는 효과 작음')

log('')
log(f'**종합 점수: {ok_count}/{total_count}**')
log('')
log(f'### 권장')
if ok_count >= 3:
    log(f'- 패키지 A 방향 진행 권장 (3+ 항목 정당)')
elif ok_count == 2:
    log(f'- 패키지 A 일부 항목만 (효과 입증된 것만) 진행')
else:
    log(f'- 패키지 A 재검토 필요. 패키지 B 또는 C 도 고려.')

# ─────────────────────────────────────────────────────────────────────────
# 마크다운 저장
# ─────────────────────────────────────────────────────────────────────────
md_path = REPORT_V2 / 'package_a_verification.md'
md_path.write_text('# 패키지 A 방향 검증 결과\n\n> 20_verify_package_a_direction.py 출력.\n'
                   + '\n'.join(evidence), encoding='utf-8')
print(f'\n[save] {md_path.relative_to(PROJECT_ROOT)}')
print('\n=== 검증 완료 ===')

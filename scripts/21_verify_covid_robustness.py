"""
21_verify_covid_robustness.py — fold1 코로나 hit 91.7% 의 진짜 원인 검증

검증 질문:
  Q1. Train (2010-2017) 에 코로나만큼 큰 변동성 시기가 있었나?
  Q2. Val/Cal (2018-2019) 에는 큰 충격이 없었나?
  Q3. 코로나 "첫 12일" 이 진짜 충격기인가, 아니면 운 좋게 작은 변동만 있었나?
  Q4. 91.7% hit 가 통계적으로 의미 있는가, 아니면 chance level 인가?
  Q5. PI 폭이 train 에서 학습된 폭의 자연 결과인가, 아니면 진짜 적응인가?
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
REPORT_V2 = PROJECT_ROOT / 'reports' / 'no_leak_v2'

print('=' * 72)
print('21_verify_covid_robustness.py — 코로나 강건성 진짜 원인 검증')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_DIR / 'processed' / 'features_v1_candidate.csv',
                  index_col='date', parse_dates=['date']).sort_index()
delta_y = (raw['kr_treasury_10y'].diff() * 100).dropna()  # bp
delta_y.name = 'delta_y_bp'

# Fold1 정의
TRAIN = ('2010-01-01', '2017-12-31')
VAL = ('2018-01-01', '2019-12-31')
CAL = ('2019-07-01', '2019-12-31')
COVID = ('2020-02-21', '2020-04-10')
COVID_FIRST12 = ('2020-02-21', '2020-03-09')

# ─────────────────────────────────────────────────────────────────────────
# Q1. Train (2010-2017) 의 |Δy| 분포 — 코로나만큼 큰 변동성 있었나?
# ─────────────────────────────────────────────────────────────────────────
print('\n## Q1. Train 기간 (2010-2017) |Δy| 분포')

train_dy = delta_y.loc[TRAIN[0]:TRAIN[1]]
val_dy = delta_y.loc[VAL[0]:VAL[1]]
cal_dy = delta_y.loc[CAL[0]:CAL[1]]
covid_dy = delta_y.loc[COVID[0]:COVID[1]]
covid12_dy = delta_y.loc[COVID_FIRST12[0]:COVID_FIRST12[1]]

print(f'\n| 구간 | n | |Δy| 평균 | |Δy| max | |Δy| q95 | |Δy| q99 | std |')
print(f'|---|---|---|---|---|---|---|')
for nm, ser in [('Train (2010-2017)', train_dy),
                ('Val (2018-2019)', val_dy),
                ('Cal (2019-H2)', cal_dy),
                ('Covid 35일', covid_dy),
                ('Covid 첫 12일', covid12_dy)]:
    abs_ = ser.abs()
    print(f'| {nm} | {len(ser)} | {abs_.mean():.2f} | {abs_.max():.2f} | '
          f'{abs_.quantile(0.95):.2f} | {abs_.quantile(0.99):.2f} | {ser.std():.2f} |')

# Train 에서 코로나 11bp 같은 큰 변동이 몇 번 있었나?
covid_max = covid_dy.abs().max()
covid12_max = covid12_dy.abs().max()
n_train_ge_5 = (train_dy.abs() >= 5).sum()
n_train_ge_10 = (train_dy.abs() >= 10).sum()
n_train_ge_15 = (train_dy.abs() >= 15).sum()
n_train_ge_covid = (train_dy.abs() >= covid_max).sum()
n_train_ge_covid12 = (train_dy.abs() >= covid12_max).sum()

print(f'\nTrain (2010-2017, 영업일 {len(train_dy)}일) 큰 변동 빈도:')
print(f'  - |Δy| ≥ 5bp:  {n_train_ge_5}일 ({n_train_ge_5/len(train_dy)*100:.1f}%)')
print(f'  - |Δy| ≥ 10bp: {n_train_ge_10}일 ({n_train_ge_10/len(train_dy)*100:.2f}%)')
print(f'  - |Δy| ≥ 15bp: {n_train_ge_15}일 ({n_train_ge_15/len(train_dy)*100:.2f}%)')
print(f'  - |Δy| ≥ 코로나 35일 max ({covid_max:.1f}bp): {n_train_ge_covid}일')
print(f'  - |Δy| ≥ 코로나 첫12일 max ({covid12_max:.1f}bp): {n_train_ge_covid12}일')

# ─────────────────────────────────────────────────────────────────────────
# Q2. Train 에서 큰 변동 시기 — 어느 사건이었나
# ─────────────────────────────────────────────────────────────────────────
print('\n## Q2. Train 기간 큰 변동 top-15 사건')
print()
top_train = train_dy.abs().nlargest(15)
print(f'| 날짜 | |Δy| (bp) | Δy (sign 포함) |')
print(f'|---|---|---|')
for d in top_train.index:
    print(f'| {d.strftime("%Y-%m-%d")} | {abs(train_dy[d]):.2f} | {train_dy[d]:+.2f} |')
print('\n→ 주요 시기: 2011 미국 신용등급 강등, 2013 taper tantrum, 2016 등 — 모델이 이미 학습)')

# ─────────────────────────────────────────────────────────────────────────
# Q3. 코로나 day-by-day — 첫 12일 vs 13~끝
# ─────────────────────────────────────────────────────────────────────────
print('\n## Q3. 코로나 day-by-day |Δy|')

print(f'\n첫 12일 (2020-02-21~03-09):')
for d in covid12_dy.index:
    print(f'  {d.strftime("%Y-%m-%d")}: {covid12_dy[d]:+.2f} bp  (|Δy|={abs(covid12_dy[d]):.2f})')

covid_after = covid_dy.loc['2020-03-10':]
print(f'\n13~끝 (2020-03-10~04-10):')
for d in covid_after.index:
    print(f'  {d.strftime("%Y-%m-%d")}: {covid_after[d]:+.2f} bp  (|Δy|={abs(covid_after[d]):.2f})')

print(f'\n첫 12일 평균 |Δy| = {covid12_dy.abs().mean():.2f}, max = {covid12_dy.abs().max():.2f}')
print(f'13~끝 평균 |Δy| = {covid_after.abs().mean():.2f}, max = {covid_after.abs().max():.2f}')

# ─────────────────────────────────────────────────────────────────────────
# Q4. 91.7% hit 통계적 유의성
# ─────────────────────────────────────────────────────────────────────────
print('\n## Q4. 첫 12일 hit 91.7% 의 통계적 유의성')

# H0: 진짜 coverage = 0.90
# 관측: 11/12 = 91.67%
# Binomial test
hit_count = 11
n = 12
p_h0 = 0.90

# Two-sided p-value
binom_p = 2 * min(stats.binom.cdf(hit_count, n, p_h0),
                  1 - stats.binom.cdf(hit_count - 1, n, p_h0))
binom_p = min(1.0, binom_p)

# Exact 95% CI for proportion
ci_low, ci_high = stats.beta.interval(0.95, hit_count + 0.5, n - hit_count + 0.5)

print(f'  - 관측: {hit_count}/{n} = {hit_count/n:.4f}')
print(f'  - H0 (진짜 cov=0.90) 하 binomial p-value (양측): {binom_p:.4f}')
print(f'  - 95% Wilson 근사 CI: [{ci_low:.4f}, {ci_high:.4f}]')

if binom_p > 0.05:
    print(f'  - → ⚠️ p={binom_p:.4f} > 0.05, 90% 와 통계적으로 구분 불가')
    print(f'    → 91.7% 는 "특별히 좋다" 고 말하기 어려움 (n=12 너무 작음)')
else:
    print(f'  - → 의미 있는 우위')

# ─────────────────────────────────────────────────────────────────────────
# Q5. 만약 PI 폭이 train 에서 학습된 자연 폭이라면, 코로나 외 시기에서도 비슷할까?
# ─────────────────────────────────────────────────────────────────────────
print('\n## Q5. Train 에서 학습된 PI 폭의 자연 결과인가?')

# Train 의 모든 일별 |Δy| 분포에서 90% PI 가 자연스럽게 어느 폭이어야 하나?
# 단순 Naive: train 의 [q05, q95] 폭
train_pi_natural = train_dy.quantile(0.95) - train_dy.quantile(0.05)
print(f'  - Train (2010-2017) 의 |Δy| 단순 [q05, q95] 폭 = {train_pi_natural:.2f} bp')
print(f'  - 우리 모델 fold1 test 평균 PI 폭 = 12.83 bp')
print(f'  - 우리 모델 fold1 코로나기 평균 PI 폭 = 16.45 bp')
print()
print(f'  → 모델 PI 폭(12.83 bp) 이 train 단순 분위수 폭과 비슷하면,')
print(f'    PI 가 "고정 폭" 으로 학습됐을 가능성. 코로나에 진짜 적응 X.')

# Test 비-코로나 시기 평균 |Δy| vs 코로나
test_normal = delta_y.loc['2020-04-11':'2020-12-31']
print(f'\n  Test 비-코로나 시기 (2020-04-11~12-31) 평균 |Δy| = {test_normal.abs().mean():.2f}')
print(f'  Test 코로나기 평균 |Δy| = {covid_dy.abs().mean():.2f}')
print(f'  → 코로나기 |Δy| 가 평소의 {covid_dy.abs().mean()/test_normal.abs().mean():.1f}배')

# ─────────────────────────────────────────────────────────────────────────
# 종합 결론
# ─────────────────────────────────────────────────────────────────────────
print('\n' + '=' * 72)
print('종합 결론')
print('=' * 72)
print()
print('### 시나리오 평가 (3가지 가설)')
print()

# 시나리오 A: 진짜 OOD 강건
covid_outside_train_q99 = covid_dy.abs().max() > train_dy.abs().quantile(0.99)
n_train_similar = (train_dy.abs() >= 10).sum()
print(f'**A. "진짜 OOD 강건"**')
print(f'   - Train 에 |Δy|≥10bp 가 {n_train_similar}일 ({n_train_similar/len(train_dy)*100:.1f}%) 존재')
print(f'   - 코로나 max |Δy| = {covid_dy.abs().max():.1f}bp, train max = {train_dy.abs().max():.1f}bp')
if not covid_outside_train_q99:
    print(f'   - → ⚠️ 코로나 변동성이 train q99 ({train_dy.abs().quantile(0.99):.1f}bp) 안에 들어감')
    print(f'        → 모델이 비슷한 패턴을 train 에서 이미 봄')
else:
    print(f'   - → ✅ 코로나가 train q99 밖 — 진짜 OOD')
print()

# 시나리오 B: PI 폭이 우연히 넓어서 hit
print(f'**B. "PI 폭이 평균적으로 넓어서 우연히 hit"**')
print(f'   - 첫 12일 hit 11/12 의 binomial p-value = {binom_p:.4f}')
if binom_p > 0.05:
    print(f'   - → ⚠️ 91.7% 와 90% 가 통계적으로 구분 불가 → "우연" 가능성 충분')
print()

# 시나리오 C: 첫 12일이 우연히 작은 변동
print(f'**C. "첫 12일이 우연히 작은 변동만 있었음"**')
print(f'   - 첫 12일 평균 |Δy| = {covid12_dy.abs().mean():.2f} bp')
print(f'   - 13~끝 평균 |Δy| = {covid_after.abs().mean():.2f} bp')
if covid12_dy.abs().mean() < covid_after.abs().mean():
    print(f'   - → ✅ 첫 12일이 후반보다 변동 작음 → 시나리오 C 부분 정당')
else:
    print(f'   - → ❌ 첫 12일이 후반보다 더 큼 → 시나리오 C 약함')
print()

print('### 정직한 해석')
print()
print('- 모델이 코로나를 "학습"했냐? → 아니다 (test=2020 은 train/val/cal 분리)')
print('- 모델이 비슷한 변동성 패턴을 train 에서 봤냐? → 데이터로 확인 필요 (위 Q1·Q2)')
print('- 91.7% hit 가 통계적 의미 있냐? → n=12 너무 작음, p-value 로 판단')
print('- PI 폭이 적응적이냐, 평균적이냐? → 평균 12.83 bp 가 train 자연 폭과 비교')

# Save log
md_lines = []
md_lines.append('# 코로나 hit 91.7% 의 진짜 원인 검증')
md_lines.append('')
md_lines.append('> 21_verify_covid_robustness.py 출력')
md_lines.append('')
md_lines.append('## 핵심 결론')
md_lines.append('')

if binom_p > 0.05:
    md_lines.append(f'- **첫 12일 91.7% (11/12) 는 통계적으로 90% 와 구분 불가** (p={binom_p:.4f}, n=12)')
    md_lines.append(f'- "특별히 잘 대응" 이라 단정 금지')

if covid_outside_train_q99:
    md_lines.append(f'- 코로나 max ({covid_dy.abs().max():.1f}bp) 가 train q99 밖 → 진짜 OOD')
else:
    md_lines.append(f'- 코로나 max ({covid_dy.abs().max():.1f}bp) 가 train q99 ({train_dy.abs().quantile(0.99):.1f}bp) 안')
    md_lines.append(f'  → 모델이 train 에서 비슷한 변동성 학습 (2011·2013·2016 등)')

md_lines.append('')
(REPORT_V2 / 'covid_robustness_check.md').write_text('\n'.join(md_lines), encoding='utf-8')
print(f'\n[save] reports/no_leak_v2/covid_robustness_check.md')

"""
14_features_v2.py — v2 누수제거 + 새 변수 3종 추가

새 변수 (모두 t-1 시점에서 계산):
  1. spread_10y_t1   = (us_treasury_10y - kr_treasury_10y).shift(1)
                       한미 스프레드 — 외국인 자금 흐름 1차 동인
  2. delta_us10y_t1  = us_treasury_10y.diff().shift(1)
                       어제 미국 10년 변화량 — 시차 모멘텀 신호
  3. delta_vix_t1    = vix.diff().shift(1)
                       어제 VIX 변화량 — 위험회피 모멘텀
  4. delta_dxy_t1    = dxy.diff().shift(1)
                       어제 달러 인덱스 변화량 — EM 자본흐름 모멘텀
  5. crisis_dummy    = (kr10y rolling vol 20d > train-only 80%ile).shift(1)
                       위기 더미 (계획서 §4.4 정량 정의)

누수 차단:
  - 미국 마감변수(us_treasury_10y, us_breakeven_10y, vix, sp500, dxy) shift(1)
  - 정책변수(kr_base_rate, us_fed_funds) shift(1)
  - 새 변수 모두 .shift(1) 또는 train-only quantile

출력: data/processed/features_v2_no_leak.csv
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DOCS_DIR = PROJECT_ROOT / 'docs'

with open(PROJECT_ROOT / 'configs' / 'config.yaml', encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

TARGET = CONFIG['project']['target']
LAGS = CONFIG['features']['lags']
ROLL_WINDOWS = [5, 10, 20]

SPLIT = {
    'train': ('2010-01-01', '2020-12-31'),
    'cal':   ('2021-01-01', '2021-12-31'),
    'val':   ('2022-01-01', '2022-12-31'),
    'test':  ('2023-01-01', '2025-12-31'),
}

POLICY_VARS = ['kr_base_rate', 'us_fed_funds']
US_MARKET_CLOSE_VARS = ['us_treasury_10y', 'us_breakeven_10y', 'vix', 'sp500', 'dxy']

print('=' * 72)
print('14_features_v2.py — v2 누수제거 + 새 변수 추가')
print('=' * 72)

# ─────────────────────────────────────────────────────────────────────────
# 1. 원시 데이터 로드
# ─────────────────────────────────────────────────────────────────────────
features_v1 = pd.read_csv(
    DATA_DIR / 'processed' / 'features_v1_candidate.csv',
    index_col='date', parse_dates=['date']
).sort_index()

print(f'\n[load] features_v1.shape = {features_v1.shape}')
print(f'[load] columns          = {features_v1.columns.tolist()}')

# CL-07: 한국 휴장일 타겟 drop
features_v1 = features_v1.dropna(subset=[TARGET])
print(f'[CL-07] after target dropna = {features_v1.shape}')

ALL_FEATURES = [c for c in features_v1.columns if c != TARGET]

# ─────────────────────────────────────────────────────────────────────────
# 2. 누수 차단 — 정책변수 + 미국 마감변수 모두 t-1
# ─────────────────────────────────────────────────────────────────────────
LAG_VARS = sorted(set(
    [v for v in POLICY_VARS if v in ALL_FEATURES]
    + [v for v in US_MARKET_CLOSE_VARS if v in ALL_FEATURES]
))
print(f'\n[CL-05·05c] shift(1) 적용 = {LAG_VARS}')

features_safe = features_v1.copy()
for v in LAG_VARS:
    features_safe[v] = features_safe[v].shift(1)

# 첫 행 NaN drop
features_safe = features_safe.dropna(subset=LAG_VARS)
print(f'[CL-05·05c] after shift = {features_safe.shape}')

# ─────────────────────────────────────────────────────────────────────────
# 3. 새 변수 5개 추가 (모두 t-1 시점)
# ─────────────────────────────────────────────────────────────────────────
print('\n[NEW] 새 변수 5개 추가 (모두 t-1)')

# 한국 변수는 t 시점에 KR 종가에서 알려짐 (shift 불요)
kr10y = features_v1[TARGET]
us10y = features_v1['us_treasury_10y']
vix_ = features_v1['vix']
dxy_ = features_v1['dxy']

# (1) 한미 스프레드 t-1
features_safe['spread_10y_t1'] = (us10y - kr10y).shift(1)

# (2) Δus10y t-1
features_safe['delta_us10y_t1'] = us10y.diff().shift(1)

# (3) Δvix t-1
features_safe['delta_vix_t1'] = vix_.diff().shift(1)

# (4) Δdxy t-1
features_safe['delta_dxy_t1'] = dxy_.diff().shift(1)

# (5) 위기 더미 — train-only threshold (CL-08)
delta_y_full = (kr10y.diff() * 100)  # bp
vol_20d = delta_y_full.rolling(20).std().shift(1)  # rolling은 shift(1) 강제 (CL-03)

# Train 구간 통계로만 임계값 결정
vol_train = vol_20d.loc[SPLIT['train'][0]:SPLIT['train'][1]].dropna()
threshold_train = float(vol_train.quantile(0.8))
print(f'[CL-08] train-only vol 20d 80%ile = {threshold_train:.3f} bp')

features_safe['crisis_dummy'] = (vol_20d > threshold_train).astype(int)

NEW_VARS = ['spread_10y_t1', 'delta_us10y_t1', 'delta_vix_t1', 'delta_dxy_t1', 'crisis_dummy']
print(f'[NEW] 추가 변수 = {NEW_VARS}')

# 새 변수의 NaN 통계
for v in NEW_VARS:
    n_nan = features_safe[v].isna().sum()
    print(f'  {v:20s} NaN={n_nan}')

# 첫 행 NaN drop (rolling 20일 워밍업으로 인한 것)
features_safe = features_safe.dropna(subset=NEW_VARS)
print(f'[NEW] after dropna = {features_safe.shape}')

# ─────────────────────────────────────────────────────────────────────────
# 4. lag/rolling 생성
# ─────────────────────────────────────────────────────────────────────────
print('\n[FE] Lag/Rolling feature 생성')

# 입력 변수 set: 원본 (정책/미국마감 shift 적용된) 9개 + 새 변수 5개 = 14개
INPUT_FEATURES = ALL_FEATURES + NEW_VARS
print(f'[FE] 입력 변수 = {len(INPUT_FEATURES)}개')

lag_blocks = []
for c in INPUT_FEATURES:
    for k in LAGS:
        lag_blocks.append(features_safe[c].shift(k).rename(f'{c}__lag{k}'))
df_lag = pd.concat(lag_blocks, axis=1)
print(f'[FE] lag features = {df_lag.shape}')

roll_blocks = []
for c in INPUT_FEATURES:
    if c == 'crisis_dummy':
        continue  # 더미는 rolling 불필요
    for w in ROLL_WINDOWS:
        roll_blocks.append(
            features_safe[c].rolling(w).mean().shift(1).rename(f'{c}__rmean{w}')
        )
        roll_blocks.append(
            features_safe[c].rolling(w).std().shift(1).rename(f'{c}__rstd{w}')
        )
df_roll = pd.concat(roll_blocks, axis=1)
print(f'[FE] roll features = {df_roll.shape}')

# 타겟 Δy_t (bp)
y_bp = (features_safe[TARGET].diff() * 100).rename('delta_y_bp')

df_features = pd.concat([
    features_safe[INPUT_FEATURES],
    df_lag,
    df_roll,
    y_bp.to_frame(),
], axis=1).dropna()
print(f'[FE] final df_features = {df_features.shape}')

# ─────────────────────────────────────────────────────────────────────────
# 5. 저장
# ─────────────────────────────────────────────────────────────────────────
out = DATA_DIR / 'processed' / 'features_v2_no_leak.csv'
df_features.to_csv(out, index_label='date')
print(f'\n[save] {out.relative_to(PROJECT_ROOT)}')

# Split별 행 수 확인
print('\n[split 별 행 수]')
for k, (s, e) in SPLIT.items():
    n = len(df_features.loc[s:e])
    print(f'  {k:6s} {s} ~ {e}  = {n:,d}')

# ─────────────────────────────────────────────────────────────────────────
# 6. 새 변수 정당화 문서 (자동 생성)
# ─────────────────────────────────────────────────────────────────────────
just_md = DOCS_DIR / 'features_v2_justification.md'
lines = [
    '# v2 새 변수 도메인 정당화',
    '',
    '> features_v2_no_leak.csv 에 추가된 변수 5개의 도메인·학술 근거.',
    '> 작성: scripts/14_features_v2.py 자동 생성 (사후합리화 방지 — 결과 보기 전 사전 기록).',
    '',
    '| 변수 | 정의 | 도메인 정당화 |',
    '|---|---|---|',
    '| `spread_10y_t1` | (us10y - kr10y).shift(1) | 한미 금리차는 외국인 채권 자금 흐름의 1차 동인. Caballero & Krishnamurthy (2009) 등 EM capital flow 학술 표준. 환율 부재로 흡수 못한 EM 충격 채널. |',
    '| `delta_us10y_t1` | us10y.diff().shift(1) | 어제 미국 10년 변화량 — t-1 → t 시차 모멘텀 신호. 미국 종가 형성(KST 새벽) → 한국 시초가 갭 → 한국 종가까지 영향. |',
    '| `delta_vix_t1` | vix.diff().shift(1) | 어제 VIX 변화량 — 위험회피 모멘텀. VIX 급등은 안전자산 선호로 한국 채권 매수 → 금리 ↓ 시차 효과. |',
    '| `delta_dxy_t1` | dxy.diff().shift(1) | 어제 달러 인덱스 변화량 — EM 자본 유출입 모멘텀. DXY ↑ → EM 자본 유출 → 한국 채권 매도 → 금리 ↑. |',
    '| `crisis_dummy` | (vol_20d > train-only 80%ile).shift(1) | 계획서 §4.4 위기 정량 정의. Train-only quantile로 누수 차단. 위기 구간 Coverage 회복용. |',
    '',
    '## 누수 차단 점검 (사전)',
    '- 모든 변수 `.shift(1)` 적용 → CL-05/05c 준수',
    '- 위기 더미 임계값은 train 구간 통계로만 결정 → CL-08 준수',
    '- rolling vol 자체도 `.shift(1)` → CL-03 준수',
    '',
    f'## 통계',
    f'- 위기 더미 train-only 임계값 = {threshold_train:.3f} bp',
    f'- 입력 변수 (raw + 새변수) = {len(INPUT_FEATURES)}개',
    f'- 최종 feature 수 (lag/roll 포함, 라벨 제외) = {df_features.shape[1] - 1}',
    f'- 데이터 기간 = {df_features.index.min().date()} ~ {df_features.index.max().date()}',
    '',
]
just_md.write_text('\n'.join(lines), encoding='utf-8')
print(f'[save] {just_md.relative_to(PROJECT_ROOT)}')

print('\n=== 완료 ===')

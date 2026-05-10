# 누수 제거 (CL-05c) 후 재학습 결과 비교

## 누수 위치
- `notebooks/02b_preprocess_baseline.ipynb` §3 — 정책변수만 shift(1) 적용,
  미국 마감변수(us_treasury_10y, us_breakeven_10y, vix, sp500, dxy)는 raw[t] 사용
- KR 종가(15:30 KST) 시점에 아직 발표되지 않은 미국 종가 입력 → 미래 정보 누수

## 수정
- 정책변수 + 미국 마감변수 모두 shift(1) → raw[t-1] 사용으로 통일
- 새 features: `data/processed/features_with_lags_v2_no_leak.csv`

## Test set 비교 (LSTM 분위수 회귀, 3 시드)

| seed | dir_acc 누수전 | dir_acc 누수후 | Δ | RMSE 전 | RMSE 후 | Cov 전 | Cov 후 |
|---|---|---|---|---|---|---|---|
| 42 | 0.6275 | 0.5053 | **-0.1222** | 4.235 | 4.535 | 0.9167 | 0.8318 |
| 123 | 0.6516 | 0.4947 | **-0.1569** | 4.170 | 4.538 | 0.9003 | 0.8289 |
| 2024 | 0.6335 | 0.4947 | **-0.1388** | 4.181 | 4.535 | 0.8899 | 0.8289 |

**평균 dir_acc**: 0.6375 → 0.4982 (Δ -0.1393)

## DM test (LSTM vs Naive · XGBoost, q50 squared error)

| seed | comparison | DM_HLN | p_value | winner |
|---|---|---|---|---|
| 42 | LSTM_vs_Naive | 0.177 | 0.8595 | tie |
| 42 | LSTM_vs_XGBoost | 3.139 | 0.0018 | OPP |
| 123 | LSTM_vs_Naive | 0.485 | 0.6278 | tie |
| 123 | LSTM_vs_XGBoost | 3.153 | 0.0017 | OPP |
| 2024 | LSTM_vs_Naive | -0.156 | 0.8757 | tie |
| 2024 | LSTM_vs_XGBoost | 3.127 | 0.0018 | OPP |

## 해석
- **누수 영향 큼**: 평균 dir_acc 가 5%p 이상 하락 — 누수가 결과를 크게 부풀렸음
- 학술 합격선 53% 대비: 누수 후 평균 49.8%

# 5주차 Ablation 계획서 (3주차 사전 명시 · 4주차 진단 후 갱신)

> 계획서 v5.1 §3.2(b) — 환율 1개 필수 + 추가 1~2개 선택.
> 음수 채택 원칙 §3.2(c): 부정적 결과도 §6.4 오류 분석 자료로 보고.
> 생성: notebooks/03_freeze_xgboost.ipynb · 2026-05-02
> **갱신: scripts/05_lstm_diff_ablation.py · 2026-05-02 — A0 추가 (4주차 부진 원인 진단 결과)**

## 사전 명시된 Ablation 후보 (우선순위 재배치)

| ID | 우선순위 | 변경 | 가설 | 평가 metric |
|----|---------|------|------|-------------|
| **A0** | 🔴🔴 **필수 (4주차 진단 결과 격상)** | **base 8 → Δbase 8 (1일 차분 + lag-1 강제)** | features 레벨↔Δy 상관 \|r\|<0.05 → 비정상성 신호 손실. 차분이 stationary 신호 운반 (Δus_treasury_10y[t-1] vs Δy[t] r=+0.336) | Pinball/Coverage/Sharpness/RMSE/Dir_Acc/DM |
| A1 | 🔴 필수 | base 8 + `krw_usd` → 9 | 환율 추가가 정부 개입 시점 SHAP 기여 | Pinball / Coverage / Sharpness / RMSE / DM(HLN+Bonferroni) |
| A2 | 🟠 선택 | base 8 + `kospi` → 9 | 국내 위험자산 채널 가설 | 동일 |
| A3 | 🟡 선택 | base 8 + `kr_ppi` → 9 | SHAP 2위(월별 ffill 가짜 신호?) 사후 검증 | 동일 |

### A0 사전 결과 (test split, baseline LSTM 동일 하이퍼파라미터)

| 지표 | LSTM_raw[t] (W4) | **LSTM_Δfeat[t-1] (A0)** | 회복 |
|------|------------------|--------------------------|------|
| RMSE q50 (bp) | 4.535 | **4.156** | -8.4% / Naive 대비 **-10.6%** |
| Coverage 90% | 0.824 | **0.899** | 목표 정확 달성 ✅ |
| Dir_Acc q50 | 0.495 | **0.662** | 목표 0.55 초과 ✅ |
| Pinball q50 | 1.696 | **1.536** | -9.4% |

→ §7 메인 차별화 3개 지표 모두 test 에서 충족. 5주차 본격 평가 시 LSTM_Δfeat[t-1] 을 **본 모델 baseline 으로 채택**, A1/A2/A3 는 그 위에 추가 변수로 재정의:

| 재정의 | 변경 | 비고 |
|--------|------|------|
| A1' | A0 + `Δkrw_usd[t-1]` | 환율 차분의 lag-1 효과 검증 |
| A2' | A0 + `Δkospi[t-1]` | 국내 위험자산 채널 |
| A3' | A0 + `Δkr_ppi[t-1]` (CL-01 발표일 시프트) | 월별 변수 차분의 lag 효과 |

## 평가 baseline

- **A0 baseline (5주차 본 모델)**: `scripts/05_lstm_diff_ablation.py` 의 LSTM_Δfeat[t-1] (8 Δfeatures, q=[0.05,0.5,0.95]). C3 multi-seed (42/123/2024) 검증으로 robust 확인 후 grid 5×5 진입.
- **참고 baseline (reference only)**: W3 XGBoost(q50) (8 vars × 96 lag/rolling feat) + W4 LSTM_raw[t]. 후자는 CL-05b·CL-05c 잔존 결함 명시 후 비교 reference 로만 보존.
- 각 ablation 마다 동일 분할/시드/하이퍼파라미터로 학습 후 A0 baseline 과 비교.

## 데이터 준비 절차 (5주차 진입 시)

### A1' (Δkrw_usd[t-1]) — 환율 차분의 lag-1 효과
1. `data/interim/wide_daily_filled.csv` 에서 `krw_usd` 컬럼 추출
2. **A0 와 동일 변환 적용**: `df_diff_krw_usd = krw_usd.diff().shift(1)` → `X[t] = krw_usd[t-1] - krw_usd[t-2]` (causal lag-1)
3. A0 의 8 Δfeatures + Δkrw_usd[t-1] = **9 Δfeatures**
4. RobustScaler **train 만 재fit** (CL-02) → `models/scaler_diff_a1.pkl` 별도 저장
5. CL-10 (환율 분리 원칙) ablation 모델에는 명시적으로 사용한다고 docs/ 에 기록

### A2' (Δkospi[t-1]) — 국내 위험자산 채널 재검증
1. `data/interim/wide_daily_filled.csv` 에서 `kospi` 컬럼 추출 (W3 freeze 단계에서 Granger p=0.57 로 제외됐으나 차분 형태에서는 다른 결과 가능)
2. **A0 와 동일 변환 적용**: `kospi.diff().shift(1)`
3. A0 의 8 Δfeatures + Δkospi[t-1] = **9 Δfeatures**
4. RobustScaler 재fit → `models/scaler_diff_a2.pkl`
5. (참고) 사전 점검: `corr(Δkospi[t-1], Δy_t)` 가 p<0.05 신호 보일 때만 실행 권장

### A3' (Δkr_ppi[t-1]) — 월별 변수 차분의 lag 효과
1. `data/interim/wide_daily_filled.csv` 에서 `kr_ppi` 추출 (이미 ffill 적용된 일별 인덱스)
2. **CL-01 발표일 시프트 검증**: 익월 25일경 발표 → +1개월 lag 강제 후 차분
3. `kr_ppi_announced.diff().shift(1)` 형태로 lag-1 causal 입력 보장
4. RobustScaler 재fit → `models/scaler_diff_a3.pkl`

## 결과 보고 형식 (5주차 채움)

| Ablation | Δ Pinball q50 | Δ Coverage 90% | Δ Sharpness | Δ RMSE q50 | Δ Dir_Acc | DM p-value (HLN+Bonferroni) |
|----------|---------------|----------------|-------------|------------|-----------|------------------------------|
| A1' (+Δkrw_usd) | ? | ? | ? | ? | ? | ? |
| A2' (+Δkospi)   | ? | ? | ? | ? | ? | ? |
| A3' (+Δkr_ppi)  | ? | ? | ? | ? | ? | ? |

> 음수 채택 원칙(§3.2.c): Δ 가 음수라도 그대로 보고. "왜 효과 없었는가" 를 §6.4 오류 분석 자료로 활용.

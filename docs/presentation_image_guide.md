# 중간 발표 — 섹션별 이미지 가이드 (4번~11번)

> 모든 경로는 프로젝트 루트(`macro-bond-forecast/`) 기준 상대경로입니다.
> `improved/` 폴더의 이미지는 발표용으로 새로 제작된 고품질 버전입니다. 가능하면 improved 버전을 우선 사용하세요.

---

## 섹션 4. 베이스라인 모델 비교 — Naive & ARIMA

> **목적**: Naive(Δy=0)와 ARIMA(1,0,1)를 베이스라인으로 잡고, 이것을 넘어야 연구 성과가 있다는 기준점을 명확히 함

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **베이스라인 비교 (추천)** | `reports/figures/improved/baseline_naive_arima_improved.png` | Naive vs ARIMA — RMSE/MAE/방향정확도 3축 비교. 둘 다 50% 수준 → 동전 던지기 → ML 필요 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 2 | 베이스라인 비교 (구버전) | `reports/figures/w2_04_baseline_compare.png` | Naive vs ARIMA 비교 (초기 버전) |

---

## 섹션 5. XGBoost 분위수 회귀 — 8개 변수 결과

> **목적**: XGBoost 분위수 모델(v0, 누수 수정 전)의 결과를 보여주는 슬라이드

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | XGBoost 분위수 예측 밴드 | `reports/figures/w3_01_xgb_quantile_bands_test.png` | **Test 구간에서 q05/q50/q95 예측 밴드 + 실제값** — 90% 커버리지 시각화 |
| 2 | **XGBoost v0 vs 베이스라인 (추천)** | `reports/figures/improved/baseline_with_xgb_v0_improved.png` | XGBoost v0 vs Naive vs ARIMA — RMSE/MAE/방향정확도 3축 비교. XGBoost 51.2%로 Naive와 거의 동일 (누수 상태) |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 3 | 베이스라인 비교 (구버전) | `reports/figures/w3_02_baselines_compare.png` | XGBoost vs Naive vs ARIMA RMSE/MAE 비교 바 차트 (구버전) |

### 관련 데이터 (수치 인용용)
- `reports/xgb_quantile_eval_w3.csv` — XGBoost v0 평가 수치

---

## 섹션 6. LSTM 분위수 회귀 — 결과 설명 + 누수 발견 과정

> **목적**: LSTM 모델 결과를 보여주고, 결과값이 비정상적으로 좋아서 누수를 의심하게 된 과정 설명

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **v0 4모델 비교 (추천)** | `reports/figures/improved/rmse_4models_v0_improved.png` | **핵심!** LSTM 65.2% 방향정확도가 비정상적으로 높음 → SHAP에서 us_treasury_10y 발견 → 시차 누수 발견. RMSE/방향정확도/Coverage 3축으로 누수 의심 근거를 한 장에 보여줌 |
| 2 | LSTM 분위수 예측 밴드 | `reports/figures/w4_02_lstm_quantile_bands_test.png` | LSTM Test 구간 q05/q50/q95 예측 밴드 — XGBoost와 비교 가능 |
| 3 | LSTM SHAP 중요도 | `reports/figures/w4_03_lstm_shap_bar.png` | **SHAP bar 차트** — us_treasury_10y가 비정상적으로 높아서 누수 의심의 근거 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 4 | RMSE 비교 (구버전) | `reports/figures/w4_05_rmse_compare.png` | LSTM vs XGBoost vs Baseline RMSE 비교 (구버전) |
| 5 | LSTM Loss 곡선 | `reports/figures/w4_01_loss_curve.png` | LSTM 학습 loss 곡선 (early stopping 시점) |
| 6 | LSTM SHAP 시계열 히트맵 | `reports/figures/w4_04_lstm_shap_time_heatmap.png` | 변수 중요도의 시간별 변화 |

### 관련 데이터 (수치 인용용)
- `reports/lstm_quantile_eval_w4.csv` — LSTM v0 평가 수치
- `reports/leakage_audit_w2.csv` — 누수 감사 결과

---

## 섹션 7. 누수 수정 후 재학습 결과 — XGBoost & LSTM 변화

> **목적**: 미국 변수 시차 누수를 수정(shift(1))한 후 모델 성능이 어떻게 변했는지 설명

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **v0→v1 누수 전/후 비교 (추천)** | `reports/figures/improved/v0_v1_leakage_fix_improved.png` | **핵심!** LSTM 방향정확도 65%→50%로 폭락 + RMSE/Coverage/Sharpness 지표 변화. "좋은 결과를 의심하라"는 교훈 |
| 2 | **v0→v1→v2 방향 정확도 변화** | `reports/no_leak_v2/figures/01_dir_acc_3stages.png` | 누수(65%)→수정(49.8%)→개선(61.1%) 3단계 변화를 한 눈에 보여줌 |
| 3 | XGBoost vs LSTM v2 비교 | `reports/no_leak_v2/figures/xgb_vs_lstm_v2_comparison.png` | 누수 수정 후 XGBoost와 LSTM Walk-forward 3-fold 성능 비교 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 4 | 모델 비교 종합 | `reports/no_leak_v2/figures/model_comparison_summary_v2.png` | v2 전체 모델 비교 요약 |
| 5 | LSTM v2 시드별 방향정확도 | `reports/no_leak_v2/figures/lstm_v2_dir_acc_by_seed.png` | LSTM v2의 시드별 방향 정확도 비교 |
| 6 | diff ablation 비교 | `reports/figures/w4_06_diff_ablation_compare.png` | 변환 방식별 성능 비교 |

### 관련 데이터 (수치 인용용)
- `reports/no_leak/leakage_fix_comparison.md` — 누수 전/후 상세 비교
- `reports/no_leak/leakage_fix_comparison_test.csv` — 누수 전/후 수치
- `reports/no_leak_v2/comparison_v0_v1_v2.md` — v0→v1→v2 종합 비교

---

## 섹션 8. 파생변수 5개 선정 — 도메인 지식 활용

> **목적**: XGBoost/LSTM이 변수 간 관계를 자동 학습하지 못하므로 도메인 지식으로 파생변수 5개를 만든 이유 설명

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **파생변수 5개 설명표 (추천)** | `reports/figures/improved/derived_features_analysis.png` | **파생변수 5개의 정의(수식), 도메인 근거, 메커니즘**을 표로 정리한 이미지. 모든 변수에 shift(t-1) 적용 + XGBoost가 트리 기반 → 변수 간 차이/비율을 자동 계산 불가 → 명시적 파생변수 필요 설명 포함 |

### 참고 내용 (PPT 표로 추가 가능)

**출처**: `docs/features_v2_justification.md`

| 변수 | 정의 | 도메인 근거 |
|------|------|-------------|
| `spread_10y_t1` | (us10y − kr10y)[t-1] | 한미 금리차 → 외국인 자금 흐름 |
| `delta_us10y_t1` | Δus10y[t-1] | 미국 금리 변화 모멘텀 → 한국 시초가 갭 |
| `delta_vix_t1` | Δvix[t-1] | 위험회피 모멘텀 → 안전자산 선호 |
| `delta_dxy_t1` | Δdxy[t-1] | 달러 강세 모멘텀 → EM 자본유출 |
| `crisis_dummy` | vol > 80%ile [t-1] | 위기 구간 식별 (train-only 임계값) |

---

## 섹션 9. 변수 점검 — 파생변수 포함 변수 선정 근거 보강

> **목적**: 파생변수를 추가한 후 변수 선정의 통계적 근거를 보여줌

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **13개 변수 3단계 검증 (추천)** | `reports/figures/improved/derived_features_validation_13vars.png` | **핵심!** 원본 8개(초록) + 파생 5개(보라) 13변수 전체에 대해 **상관계수·VIF·Granger** 3단계 검증을 한 장에 보여줌. sp500과 kr_base_rate만 상관 기준 미달(X), VIF 전부 10 미만, Granger 전부 유의 (vix만 경계) |
| 2 | 상관 히트맵 (원본 22개) | `reports/figures/w2_01_correlation_heatmap.png` | 22개 변수 상관행렬 히트맵 — 원본 변수 선정 과정 설명시 사용 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 3 | VIF 분석 바 차트 (구버전) | `reports/figures/w2_02_vif.png` | VIF 분석 결과 — 원본 8개 기준 (구버전) |
| 4 | Granger 인과검정 (구버전) | `reports/figures/w2_03_granger.png` | Granger 인과검정 결과 — 원본 8개 기준 (구버전) |
| 5 | SHAP 중요도 (XGBoost) | `reports/figures/auto_05_shap_bar.png` | XGBoost SHAP 변수 중요도 바 차트 |
| 6 | SHAP Summary Plot | `reports/figures/auto_05_shap_summary.png` | SHAP summary plot — 변수별 영향 방향 |
| 7 | 타겟 분석 | `reports/figures/auto_02_target_analysis.png` | 타겟(Δy) 분포·ACF 분석 |

### 관련 데이터 (수치 인용용)
- `docs/feature_validation_w2.md` — 변수 검증 상세 결과 (VIF, Granger, 점수)

---

## 섹션 10. 13개 변수(8+5) XGBoost & LSTM 분석 결과

> **목적**: 파생변수 5개를 추가한 13개(8원본+5파생) 변수로 XGBoost/LSTM v2 결과 발표

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | XGBoost v2 예측 구간 (fold3) | `reports/no_leak_v2/figures/xgb_v2_pred_interval_fold3.png` | **v2 XGBoost의 Test 구간 예측 밴드** — 13개 변수 결과 |
| 2 | LSTM v2 예측 구간 (fold3) | `reports/no_leak_v2/figures/lstm_v2_pred_interval_fold3.png` | **v2 LSTM의 Test 구간 예측 밴드** — 13개 변수 결과 |
| 3 | 예측 구간 비교 (XGB vs LSTM) | `reports/no_leak_v2/figures/page6_7_pred_interval_comparison.png` | **XGBoost vs LSTM 예측 구간 나란히 비교** |
| 4 | CQR 효과 | `reports/no_leak_v2/figures/02_cqr_effect.png` | CQR 보정 전후 Coverage & Sharpness 비교 |
| 5 | CQR 효과 (발표용) | `reports/no_leak_v2/figures/page6_cqr_effect.png` | CQR 효과 발표 슬라이드용 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 6 | 4panel 비교 | `reports/no_leak_v2/figures/page8_4panel_comparison.png` | 4가지 지표 동시 비교 (방향정확도, CQR, Walk-forward, DM) |
| 7 | 구간 폭 비교 | `reports/no_leak_v2/figures/page8_interval_width_comparison.png` | 예측 구간 폭 비교 |
| 8 | Grid Search 히트맵 | `reports/figures/w5_01_grid_heatmap.png` | 하이퍼파라미터 grid search 결과 히트맵 |
| 9 | W5 비교 차트 | `reports/figures/w5_02_w5_compare.png` | 5주차 모델 비교 종합 |
| 10 | DM 검정 결과 | `reports/no_leak_v2/figures/04_dm_test.png` | Diebold-Mariano 검정 결과 |

### 관련 데이터 (수치 인용용)
- `reports/no_leak_v2/xgb_v2_eval.csv` — XGBoost v2 상세 평가
- `reports/no_leak_v2/xgb_grid_v2.csv` — Grid search 결과
- `reports/no_leak_v2/walkforward_xgb_v2.csv` — Walk-forward XGBoost
- `reports/no_leak_v2/walkforward_lstm_v2.csv` — Walk-forward LSTM

---

## 섹션 11. 최종 변수 선정 기준 정리

> **목적**: 어떤 기준에 따라 최종 변수를 선정했는지 종합 설명

### 필수 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 1 | **v2 결과 종합 4panel** | `reports/no_leak_v2/figures/05_summary_4panel.png` | **핵심!** — 방향정확도, CQR 커버리지, Walk-forward, DM 검정 4가지를 한 번에 |
| 2 | Walk-forward 방향 정확도 | `reports/no_leak_v2/figures/03_walkforward_dir_acc.png` | Walk-forward 3-fold: XGBoost vs LSTM 방향 정확도 비교 |
| 3 | XGBoost vs LSTM 비교 | `reports/no_leak_v2/figures/xgb_vs_lstm_v2_comparison.png` | 최종 모델 선택 근거 |

### 선택 이미지

| # | 이미지 | 경로 | 설명 |
|---|--------|------|------|
| 4 | SHAP 분위수별 | `reports/figures/w6_01_shap_quantile.png` | 분위수별(q05/q50/q95) SHAP 중요도 |
| 5 | SHAP 시계열 히트맵 | `reports/figures/w6_02_shap_time_heatmap.png` | 변수 중요도 시간 변화 |
| 6 | 에러 분석 4축 | `reports/figures/w6_03_error_analysis_4axis.png` | 오차 분석 4축 차트 |
| 7 | 모델 비교 종합 v2 | `reports/no_leak_v2/figures/model_comparison_summary_v2.png` | 전체 모델 비교 요약 |

### 관련 데이터 (수치 인용용)
- `reports/no_leak_v2/comparison_v0_v1_v2.md` — 전체 히스토리
- `reports/no_leak_v2/walkforward_summary_v2.md` — 3-fold 요약

---

## 요약: 섹션별 핵심 이미지 Quick Reference

| 섹션 | 핵심 이미지 | 경로 |
|------|------------|------|
| **4** | 베이스라인 Naive vs ARIMA | `reports/figures/improved/baseline_naive_arima_improved.png` |
| **5** | XGBoost v0 vs 베이스라인 | `reports/figures/improved/baseline_with_xgb_v0_improved.png` |
| **6** | v0 4모델 비교 (LSTM 누수 의심) | `reports/figures/improved/rmse_4models_v0_improved.png` |
| **7** | v0→v1 누수 전/후 + v0→v1→v2 3단계 | `improved/v0_v1_leakage_fix_improved.png` + `no_leak_v2/figures/01_dir_acc_3stages.png` |
| **8** | 파생변수 5개 설명표 | `reports/figures/improved/derived_features_analysis.png` |
| **9** | 13변수 3단계 검증 | `reports/figures/improved/derived_features_validation_13vars.png` |
| **10** | XGB vs LSTM 예측 구간 비교 | `reports/no_leak_v2/figures/page6_7_pred_interval_comparison.png` |
| **11** | 종합 4panel | `reports/no_leak_v2/figures/05_summary_4panel.png` |

---

## 참고 사항

1. **`improved/` 폴더**: 발표용으로 새로 제작된 고품질 이미지들입니다. 기존 `w3_`, `w4_` 시리즈 대신 이 이미지들을 우선 사용하세요.
2. **발표 흐름 팁**: 4→5→6은 "베이스라인→XGBoost→LSTM(누수 발견)" 스토리로 이어지고, 7→8→9→10→11은 "정정→개선→검증→결과→결론" 스토리입니다.

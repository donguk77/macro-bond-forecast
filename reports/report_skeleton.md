# 채권금리 예측 — 최종 성능 비교 리포트 (골격)

> 계획서 v5.1 §8 — 2주차 골격, 매주 채워나감.
> 생성: notebooks/02b_preprocess_baseline.ipynb (2주차)

## 1. 요약 (Executive Summary)
- 1~2 문장 결론 — 7주차에 작성.
- 핵심 수치: Test 방향성 정확도, Pinball, Coverage, DM test p-value.

## 2. 데이터
- 기간: 2010-01 ~ 2025-12 (영업일 기준)
- 분할: Train 2010-2020 / Cal 2021 / Val 2022 / Test 2023-2025
- 변수: 1주차 freeze 9개 + (3주차 ablation 후 최종 N개)
- 누수 차단: CL-01~CL-07 통과 (본 2주차 산출).

## 3. 변수 선정 결과
- 1주차 EDA + 1주차 XGBoost-SHAP Hello World
- 2주차 상관·VIF·Granger (`docs/feature_validation_w2.md`)
- 3주차 freeze 최종 + 5주차 환율 ablation

## 4. 베이스라인 — Naive · ARIMA (본 2주차)

| 모델 | 분할 | RMSE (bp) | MAE (bp) | 방향성 |
|------|------|-----------|----------|--------|
| Naive (Δŷ=0) | train | 3.606 | 2.571 | nan |
| Naive (Δŷ=0) | cal | 3.502 | 2.631 | nan |
| Naive (Δŷ=0) | val | 7.101 | 5.483 | nan |
| Naive (Δŷ=0) | test | 4.647 | 3.477 | nan |
| ARIMA(1, 0, 1) | train | 3.603 | 2.579 | 0.515 |
| ARIMA(1, 0, 1) | cal | 3.517 | 2.643 | 0.473 |
| ARIMA(1, 0, 1) | val | 7.114 | 5.5 | 0.431 |
| ARIMA(1, 0, 1) | test | 4.651 | 3.483 | 0.497 |

> Figure: `reports/figures/w2_04_baseline_compare.png`

## 5. XGBoost 분위수 회귀 (3주차 채움)
- `objective="reg:quantileerror"` 분위수 [0.05, 0.5, 0.95]
- Pinball Loss / Coverage / Sharpness 표
- Naive·ARIMA 대비 DM test 결과

## 6. LSTM 분위수 회귀 (4-5주차 채움)
- 구조: 다변량 LSTM, lookback 30, hidden 64, layer 2, dropout 0.3
- 손실: Pinball Loss, monotonicity sort 후처리
- 학습 곡선·early stop
- (선택) Conformal CQR 후처리 — Coverage 미달 시

## 7. 평가 지표 종합 (Naive · ARIMA · XGBoost · LSTM)
| 모델 | Pinball | RMSE | MAE | 방향성 | Coverage 90% | Sharpness | DM vs Naive |
|------|---------|------|-----|--------|--------------|-----------|-------------|
| Naive | … | … | … | 50%* | n/a | n/a | — |
| ARIMA | … | … | … | … | n/a | n/a | … |
| XGBoost | … | … | … | … | … | … | … |
| LSTM | … | … | … | … | … | … | … |
| LSTM+CQR | … | … | … | … | … | … | … |

## 8. DM test (HLN 보정 + Bonferroni)
- Pinball Loss 차이 검정
- HAC(Newey-West) lag = …
- Bonferroni 다중비교 보정

## 9. SHAP 분석 (6주차 채움)
- §6.3 핵심 분석 질문 5개 답변
- 분위수별 SHAP 차분 시각화
- 시차 효과 정량화 (lag k 별 |SHAP| 평균)

## 10. 오류 분석 4축 (6주차 채움)
- (a) 방향성 오답 top-20
- (b) 큰 변동(|Δy|>5bp) 미예측 top-20
- (c) Coverage Miss — 위기 vs 정상, 보정 전후
- (d) 위기 vs 정상 구간 SHAP 평균 차이

## 11. 환율 Ablation (5주차 채움)
- 환율 포함 모델 vs 미포함
- Pinball / Coverage / Sharpness / RMSE 비교
- 정부 개입 시점(예: 2024 계엄, 2025 고환율) SHAP 분석

## 12. 위기구간 평가 (계획서 §4.4)
- 라벨링: 20일 rolling vol 상위 20% ∪ 이벤트 더미 ±1일
- 위기 vs 정상 RMSE 비율
- Coverage 위기 vs 정상

## 13. 결론 및 한계
- 차별화 포인트 10개 중 어느 것이 결과로 입증되었는가
- 미달 항목의 솔직한 한계 + 후속 연구 방향

## 부록 A. 누수 차단 체크리스트 결과
- CL-01 ~ CL-07 자동 검증 — `02b_preprocess_baseline.ipynb` §10

## 부록 B. AI 사용 기록
- `AI_USAGE_LOG.md` — 매주 +2건, 총 14건+
- `VALIDATION_LOG.md` — 검증 사례

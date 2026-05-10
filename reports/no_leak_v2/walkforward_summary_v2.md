# Walk-forward 3-fold v2 결과 (16_walkforward)

> 16_walkforward.py 마지막 라인(LSTM pooled dir_acc shape mismatch)에서 죽었지만
> 모든 학습은 완료됨. 16b_walkforward_save.py 로 결과 csv·md 저장.

## Fold 정의

| fold | train | val | cal | test |
|---|---|---|---|---|
| fold1 | 2010-01 ~ 2017-12 | 2018-01 ~ 2019-12 | 2019-07 ~ 2019-12 | 2020-01 ~ 2020-12 (코로나) |
| fold2 | 2010-01 ~ 2019-12 | 2020-01 ~ 2020-12 | 2020-07 ~ 2020-12 | 2021-01 ~ 2022-12 (인상기) |
| fold3 | 2010-01 ~ 2021-12 | 2022-01 ~ 2022-12 | 2022-07 ~ 2022-12 | 2023-01 ~ 2025-12 (안정+충격) |

## XGBoost CQR per-fold (test set)

| fold | dir_acc | Coverage 90% | CQR Q_hat (bp) |
|---|---|---|---|
| fold1 | 0.5932 | 0.9325 | +1.279 |
| fold2 | 0.5949 | 0.7013 | -1.170 |
| fold3 | 0.6416 | 0.9843 | +6.631 |

**평균**: dir_acc **0.6099 ± 0.0275**, Coverage 0.8727 ± 0.1507
**Pooled** (3 fold test 합쳐 평가): dir_acc **0.6178**, RMSE 4.721 bp (Naive 4.855 bp)

## LSTM per-fold per-seed (raw, sorted)

| fold | seed=42 | seed=123 | seed=2024 | 평균 ± std |
|---|---|---|---|---|
| fold1 | 0.5072 | 0.5604 | 0.5024 | 0.5233 ± 0.0322 |
| fold2 | 0.5864 | 0.6068 | 0.5773 | 0.5902 ± 0.0151 |
| fold3 | 0.6546 | 0.6591 | 0.6697 | 0.6611 ± 0.0078 |

**전체 평균** (9 학습): 0.5915 ± 0.0624

## DM test (XGB vs Naive, q50 squared error)

| fold | DM_HLN | p-value | Bonferroni α=0.0167 | winner |
|---|---|---|---|---|
| fold1 | -1.267 | 0.2066 | NO | tie |
| fold2 | -4.621 | 0.0000 | OK | XGB |
| fold3 | -7.959 | 0.0000 | OK | XGB |
| POOLED | -8.782 | 0.0000 | OK | XGB |

## 핵심 발견

1. **3 fold 평균 dir_acc 0.6099** + **Pooled 0.6178** — 목표 55% 안정 초과 (+5~7%p).
2. **DM XGB vs Naive**: fold2·fold3·POOLED 모두 통계 우위 (p<0.0001, Bonferroni 통과). fold1(코로나기)은 tie — distribution shift 사례.
3. **LSTM 회복**: 누수 후 v1 50%에서 v2 (3 fold 평균) 59.2% 로 회복. fold3에서 66% 수준 — 새 변수 5개 효과.
4. **Coverage**: 3 fold 평균 0.8727. fold2에서 음수 Q_hat (cal 분포 차이로 모델이 over-conservative) → 70%로 떨어짐. fold1·3은 93/98%로 과보장.
5. **단일 분할 의존성 해소**: single-split (fold3 = 우리 원래 분할)에서 우연히 좋은 게 아니라, fold2에서도 60% 유지 → 결과 일반성 입증.

## 주의 — fold2 음수 Q_hat

- Cal (2020-07 ~ 2020-12) 변동성이 매우 낮은 시기 → 모델이 cal 에서 over-conservative 하게 분위수 예측.
- 결과 Q_hat = -1.17 bp (음수) → 구간을 좁히는 보정.
- Test (2021-22) 변동성이 cal 대비 커서 보정 후 Coverage 70%로 떨어짐.
- 학술적 해결: ACI (Adaptive Conformal Inference) — distribution shift 적응 보정.
- 발표: 음수 Q_hat 자체를 "CQR 한계와 ACI 필요성" 사례로 활용 가능.

## 비교 — single-split (15) vs walk-forward (16)

| 지표 | single-split (test 2023~25) | 3-fold 평균 | Pooled |
|---|---|---|---|
| dir_acc (XGB CQR) | 0.6113 | 0.6099 | 0.6178 |
| Coverage (XGB CQR) | 0.8573 | 0.8727 | n/a |
| DM XGB vs Naive | -6.23 (p=0) | 2/3 fold OK | -8.78 (p=0) ✅ |

→ single-split 결과가 3-fold 평균과 거의 일치 → **결과 안정성 입증**.

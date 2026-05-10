# v0 → v1 → v2 누수 history + 개선 종합 비교

> 발표·제출용 단일 종합 보고서.
> 작성: 2026-05-06 · 기반: `reports/`, `reports/no_leak/`, `reports/no_leak_v2/`

---

## 1. 3 단계 history 한 줄 요약

| 단계 | 데이터 처리 | 메인 모델 | 주요 기법 | dir_acc (test) | DM vs Naive |
|---|---|---|---|---|---|
| **v0 (W7)** | 정책 변수만 shift(1) | LSTM 분위수 | tuning + multiseed | 65.2% (s=123), 평균 63.7% | LSTM 우위 (의심) |
| **v1 (13_rerun)** | + 미국 마감변수 5 shift(1) | LSTM 분위수 | (변경 없음) | 49.8% (3시드 평균) | tie (p>0.6) |
| **v2 (14·15·16)** | v1 + 새 변수 5 + grid + CQR + walk-forward | **XGBoost 분위수 + CQR** | grid + CQR + 3-fold | **61.78% (pooled), 60.99% (3-fold)** | **XGB 승 (p<0.001)** |

---

## 2. 발견된 누수 — CL-05c (cross-market timing leak)

### 위치
`notebooks/02b_preprocess_baseline.ipynb` §3 — 정책 변수만 shift(1) 적용, 미국 마감변수(us_treasury_10y, us_breakeven_10y, vix, sp500, dxy)는 raw[t] 사용

### 메커니즘
```
KST 시간선:
  t-1 영업일 15:30          t 영업일 15:30          t+1 새벽 ~05:30
       ↓                        ↓                         ↓
   kr[t-1]                   kr[t]                     us[t]
                 ↑                       ↑
            us[t-1] (이미 알려짐)    ← 한국 종가 시점에 us[t]는
                                       아직 형성 전!

타겟: Δy_t = (kr[t] - kr[t-1]) × 100 bp  (15:30 KST t일에 확정)
모델 입력: us_treasury_10y[t]  ← 미래 정보 누수
```

### 영향 정량화 (W2 audit)
| 변수 | raw[t-1] 일치율 | raw[t] 일치율 |
|---|---|---|
| us_treasury_10y | 7.5% | **100%** ← t 사용 = 누수 |
| us_breakeven_10y | 16.1% | 100% |
| vix | 0.5% | 100% |
| sp500 | 0.0% | 100% |
| dxy | 0.0% | 100% |

→ 5 변수 모두 raw[t] 사용 확인.

---

## 3. 누수 정정 (v0 → v1) 영향 정량화

LSTM 분위수 회귀, 시드 3개 평균, Test set 비교:

| 지표 | v0 (누수) | v1 (정정) | Δ |
|---|---|---|---|
| dir_acc | 63.75% | **49.82%** | **-13.93%p** |
| RMSE (bp) | 4.20 | 4.54 | +0.34 |
| Coverage 90% | 90.2% | 83.0% | -7.2%p |
| Sharpness (bp) | 12.8 | 11.4 | -1.4 |
| DM vs Naive | LSTM 우세 | **tie** | 통계 우위 사라짐 |

### 해석
- **dir_acc -14%p**: 누수 후 인공적 65% → 진짜 50% (랜덤 수준)
- **us_treasury_10y SHAP top 1**이었던 것 → 누수 부산물
- W6 channel validation의 부호 noise도 누수 부산물 가능성

---

## 4. v2 개선 (v1 → v2) — 모델 자체 개선

### 개선 4가지
| # | 개선 | 효과 |
|---|---|---|
| 1 | 새 변수 5개 추가 | dir_acc +5~6%p (12개 → 14개 입력) |
| 2 | XGBoost grid search (분위수별) | dir_acc +1~2%p (depth=4 최적) |
| 3 | CQR 후처리 (Cal 2021) | Coverage 80→86%, +5.6%p |
| 4 | Walk-forward 3-fold 검증 | 결과 안정성·일반성 입증 |

### 새 변수 5개 (모두 t-1 시점, 도메인 사전 정당화)

| 변수 | 정의 | 정당화 |
|---|---|---|
| `spread_10y_t1` | (us10y - kr10y).shift(1) | EM capital flow 1차 동인 (Caballero & Krishnamurthy 2009) |
| `delta_us10y_t1` | us10y.diff().shift(1) | 미국 → 한국 시차 모멘텀 (16시간) |
| `delta_vix_t1` | vix.diff().shift(1) | 위험회피 모멘텀 — 안전자산 채널 |
| `delta_dxy_t1` | dxy.diff().shift(1) | EM 자본유출 모멘텀 |
| `crisis_dummy` | (vol_20d > train-only 80%ile).shift(1) | 계획서 §4.4 정량 위기 정의 |

→ 사전 기록: `docs/features_v2_justification.md` (사후합리화 방지)

---

## 5. v2 단일 분할 결과 (15_xgb_grid_cqr, Test 2023-2025)

### XGBoost CQR
| 지표 | 결과 | 목표 | 상태 |
|---|---|---|---|
| dir_acc | **0.6113** | ≥ 0.55 | ✅ +6.1%p |
| Coverage 90% | 0.8573 (CQR) / 0.8017 (raw) | 0.87~0.93 | △ -1.3%p |
| Sharpness (bp) | 11.73 (CQR) / 10.51 (raw) | 좁을수록 | ✅ |
| RMSE (bp) | 4.48 | <Naive 4.65 | ✅ -3.7% |

### Best params (grid search)
| q | max_depth | learning_rate | n_iter |
|---|---|---|---|
| 0.05 | 4 | 0.03 | 87 |
| 0.50 | 4 | 0.05 | 215 |
| 0.95 | 4 | 0.10 | 53 |

→ 모두 얕은 트리 (depth 4) — 과적합 방지.

### DM test
| 비교 | DM_HLN | p-value | Bonferroni α=0.025 | winner |
|---|---|---|---|---|
| XGBv2 vs Naive | -6.23 | 0.000 | OK | **XGB** |
| XGBv2 vs ARIMA | -6.54 | 0.000 | OK | **XGB** |

---

## 6. v2 Walk-forward 3-fold 결과 (16_walkforward)

### Fold 정의
| fold | train | val | cal | test |
|---|---|---|---|---|
| 1 | 2010-01 ~ 2017-12 | 2018-01 ~ 2019-12 | 2019-07 ~ 2019-12 | 2020 (코로나) |
| 2 | 2010-01 ~ 2019-12 | 2020-01 ~ 2020-12 | 2020-07 ~ 2020-12 | 2021-22 (인상기) |
| 3 | 2010-01 ~ 2021-12 | 2022-01 ~ 2022-12 | 2022-07 ~ 2022-12 | 2023-25 (안정+충격) |

### XGBoost CQR per-fold
| fold | dir_acc | Coverage 90% | CQR Q̂ (bp) |
|---|---|---|---|
| fold1 | 0.5932 | **0.9325** | +1.28 |
| fold2 | 0.5949 | 0.7013 | -1.17 ⚠️ |
| fold3 | **0.6416** | **0.9843** | +6.63 |
| **평균** | **0.6099 ± 0.0269** | 0.8727 ± 0.1450 | - |
| **Pooled** | **0.6178** | n/a | - |

### LSTM per-fold (시드 3개)
| fold | seed=42 | seed=123 | seed=2024 | 평균 ± std |
|---|---|---|---|---|
| fold1 | 0.5072 | 0.5604 | 0.5024 | 0.5233 ± 0.0322 |
| fold2 | 0.5864 | 0.6068 | 0.5773 | 0.5902 ± 0.0151 |
| **fold3** | **0.6546** | **0.6591** | **0.6697** | **0.6611 ± 0.0078** |

→ LSTM도 fold3에서 v0 수준(65~67%) 회복. 새 변수 5개의 효과.

### DM test (XGB vs Naive)
| fold | DM_HLN | p-value | Bonferroni α=0.0167 | winner |
|---|---|---|---|---|
| fold1 (코로나) | -1.27 | 0.207 | NO | tie (distribution shift) |
| fold2 | -4.62 | 0.000 | OK | **XGB** |
| fold3 | -7.96 | 0.000 | OK | **XGB** |
| **POOLED** | **-8.78** | **0.000** | OK | **XGB** |

→ 2/3 fold + Pooled 통과. **단일 분할 결과(0.6113)가 우연이 아님 입증**.

---

## 7. 우리 목표 달성 현황 (최종)

| 목표 | single-split (v2) | 3-fold 평균 | Pooled | 상태 |
|---|---|---|---|---|
| dir_acc ≥ 55% | 0.6113 | 0.6099 | 0.6178 | ✅ **+5~7%p** |
| Coverage 90%±3%p | 0.8573 | 0.8727 | - | △ 거의 (fold2 영향) |
| DM vs Naive 통계 유의 | OK | 2/3 OK | OK | ✅ |
| Sharpness 우위 | 11.7 bp | - | - | ✅ |
| RMSE < Naive | 4.48 < 4.65 | - | 4.72 < 4.86 | ✅ |

---

## 8. 한계 인정 (정직성)

1. **fold1 (2020 코로나) DM tie**: 학습 분포 밖 충격 (distribution shift) 한계. ACI 적용으로 후속 보완 가능.
2. **fold2 음수 Q̂**: Cal 2020 후반이 매우 안정적이라 모델 over-conservative → CQR 보정이 좁히는 방향. ACI로 동적 해결.
3. **3-fold 평균 Coverage 87.3%**: 목표 90%±3%p 거의 도달이지만 fold2 영향으로 약간 미달.
4. **새 변수 5개의 한계**: 외국인 채권 보유잔고는 데이터 수집 부담으로 미포함 (§11 선택 확장 명시).

---

## 8a. v2 → v2+ 시도와 반증 (옵션 3 채택 사유)

v2 결과 후 추가 개선 (패키지 A — ACI + rolling vol)을 시도했으나 데이터로 반증됨.

### 시도한 개선
| # | 시도 | 가설 |
|---|---|---|
| 1 | ACI (Adaptive Conformal Inference) | distribution shift에 동적 적응으로 fold1·fold2 한계 해결 |
| 2 | Rolling volatility(타겟) 입력 추가 | 첫 충격 보호 (volatility clustering 활용) |

### 검증 결과 (`scripts/20_verify_package_a_direction.py`, `21_verify_covid_robustness.py`)

| 가설 | 데이터 결과 | 결론 |
|---|---|---|
| ACI Coverage 회복 | 93.3% → **96.2%** (over-correction, 목표 90%에서 더 멀어짐) | ❌ 반증 |
| ACI PI 폭 합리적 | 12.83 → **16.72 bp (+30%)** | ❌ 반증 |
| Rolling vol → dir_acc 개선 | -0.4%p (악화) | ❌ 반증 |
| Rolling vol → 코로나 첫 12일 보호 | 91.7% → **75.0%** (-16.7%p 악화) | ❌ 반증 |
| 모델이 코로나에 진짜 강건 | 첫 12일 91.7% binomial p=1.0 (n=12 너무 작음) | △ 단정 불가 |
| 코로나가 진짜 OOD | max 18.3bp < train max 24.2bp, train q99 대부분 안 | ❌ 진짜 OOD 아님 |

### 옵션 3 (현 상태 유지 + 발표 정정) 채택 사유
1. **데이터로 반증된 개선안은 채택 X** (끼워맞추기 회피)
2. v2 결과 (61.78% pooled, Coverage 87.3%)가 학술 합격선 53% 대비 +8%p로 충분
3. **검증·반증 history 자체가 안내문 §9 "결과 타당성 자체 검증" 직격 사례** — 발표 자산
4. 누수 차단 + 정직성 + 통계 우위가 이미 입증됨

→ 발표에서는 정직한 한계 인정 (fold1·fold2·코로나 hit n=12 통계 의미 없음 등) + 후속 작업으로 GARCH·regime-switching 명시.

---

## 9. 발표 활용 — 학술 자세 사례 (안내문 §9 직격)

본 v0 → v1 → v2 history는 안내문 §9 "결과의 타당성을 스스로 검증" 원칙의 정확한 사례:

1. **검증을 통한 의심**: 1차 결과 65%가 학술 합격선 53% +12%p로 비현실적 → 의심
2. **체계적 audit**: leakage_audit_w2.csv에서 CL-05c ❌ 발견
3. **즉시 정정**: 미국 마감변수 5개 shift(1) 적용, downstream 전체 재학습
4. **정량 영향 보고**: dir_acc -14%p 정직 보고
5. **추가 개선**: 새 변수·튜닝·CQR로 정직한 61.78% 도달
6. **walk-forward 검증**: 단일 분할 의존성 해소
7. **모든 산출물 공개**: v0(W7) / v1(13_rerun) / v2(15·16) 결과 모두 보존

→ "AI 도구를 적극 활용하되 모든 결과를 직접 검증" 원칙의 살아있는 사례.

---

## 10. 산출물 목록

### v2 코드
- `scripts/14_features_v2.py` — features 재생성
- `scripts/17_full_audit_v2.py` — 누수 자동 audit
- `scripts/15_xgb_grid_cqr.py` — XGB grid + CQR
- `scripts/16_walkforward.py` — walk-forward 3-fold
- `scripts/16b_walkforward_save.py` — 결과 csv 저장 보완
- `scripts/18_build_notebook_08.py` — 통합 노트북 빌드
- `scripts/19_verify_v2.py` — v2 모든 주장 fresh 재검증 (13/13 PASS)
- `scripts/20_verify_package_a_direction.py` — 패키지 A 방향 검증 (2/4 점수, 폐기)
- `scripts/21_verify_covid_robustness.py` — 코로나 강건성 진짜 원인 검증 (n=12 통계 의미 없음)

### v2 결과 csv
- `reports/no_leak_v2/leakage_audit_v2.csv`
- `reports/no_leak_v2/xgb_grid_v2.csv`
- `reports/no_leak_v2/xgb_v2_eval.csv`
- `reports/no_leak_v2/dm_test_xgb_v2.csv`
- `reports/no_leak_v2/walkforward_xgb_v2.csv`
- `reports/no_leak_v2/walkforward_lstm_v2.csv`
- `reports/no_leak_v2/walkforward_dm_v2.csv`

### v2 문서
- `reports/no_leak_v2/walkforward_summary_v2.md` — 3-fold 요약
- `reports/no_leak_v2/presentation_v2_revisions.md` — 발표 자료 정정 가이드 (검증·반증 history 포함)
- `reports/no_leak_v2/comparison_v0_v1_v2.md` — **본 문서**
- `reports/no_leak_v2/verification_v2.csv` — 13/13 검증 결과
- `reports/no_leak_v2/package_a_verification.md` — 패키지 A 방향 검증 결과 (반증)
- `reports/no_leak_v2/covid_robustness_check.md` — 코로나 강건성 진짜 원인 (n=12 의미 없음)
- `docs/features_v2_justification.md` — 새 변수 사전 정당화

### v2 그림
- `reports/no_leak_v2/figures/01_dir_acc_3stages.png`
- `reports/no_leak_v2/figures/02_cqr_effect.png`
- `reports/no_leak_v2/figures/03_walkforward_dir_acc.png`
- `reports/no_leak_v2/figures/04_dm_test.png`
- `reports/no_leak_v2/figures/05_summary_4panel.png`

### v2 노트북
- `notebooks/08_v2_no_leak_pipeline.ipynb` — 통합 결과 demo

### v2 데이터
- `data/processed/features_v2_no_leak.csv` (3725 rows × 163 cols)

### v2 모델
- `models/xgb_v2_q05.json`
- `models/xgb_v2_q50.json`
- `models/xgb_v2_q95.json`

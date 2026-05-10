# 옵션 3 최종 — 정직한 v2 결과 + 검증·반증 history

> 모든 학습·검증·반증 작업 완료. 발표·제출 직전 단일 종합 요약.
> 작성: 2026-05-06

---

## 1. 최종 결과 (변동 없음 — v2 유지)

### 메인 모델: XGBoost 분위수 회귀 + CQR 후처리

| 지표 | single-split | 3-fold 평균 | Pooled | 목표 | 상태 |
|---|---|---|---|---|---|
| 방향성 정확도 | 61.13% | 60.99% ± 2.69% | **61.78%** | ≥ 55% | ✅ +6.78%p |
| Coverage 90% (CQR) | 85.73% | 87.27% | - | 87~93% | △ 거의 |
| DM vs Naive | DM=-6.23 ✅ | 2/3 OK | DM=-8.78 ✅ | Bonferroni 통과 | ✅ |
| Sharpness | 11.7 bp | - | - | 정상 | ✅ |
| RMSE | 4.48 bp (Naive 4.65) | - | 4.72 bp | <Naive | ✅ |

학술 합격선 53% 대비 **+8.78%p (Pooled 기준)**.

### 실무 시뮬레이션 (Backtest v2 정직 보강판)

> 1차 단순 backtest (Sharpe 2.42) 는 carry 미반영 + 단일 split 한계. 2차 보강 (carry + walk-forward + Sharpe CI bootstrap) 으로 정직 결과 산출.

#### Single-split (test 2023-2025, carry+1bp cost)

| 전략 | Total Return | Sharpe (252) | MDD | Win Rate |
|---|---|---|---|---|
| S0 Buy-and-Hold (carry 포함) | +12.42% | 0.76 | -7.21% | 50.6% |
| S1 SignQ50 (매일 매매) | +12.10% | 0.75 | -11.25% | 52.2% |
| **S2 ConfFilter (\|q50\|>1bp만)** | **+13.43%** | **2.07** | **-1.43%** | 40.9% |

→ **신규 발견**: confidence filter 가 단순 매매보다 robust (Sharpe 2.07, MDD -1.43%).

#### Walk-forward 3-fold Pooled (정직 검증)

| 전략 | Sharpe | 95% bootstrap CI | 평가 |
|---|---|---|---|
| S0 Buy-and-Hold | +0.10 | [-0.70, +0.89] | CI 0 포함 |
| S1 SignQ50 | +0.20 | [-0.67, +1.00] | CI 0 포함 |
| **S2 ConfFilter** | **+0.62** | **[-0.18, +1.35]** | CI 0 포함 (가장 좋음) |

→ **백테스트 차원에서는 통계 우위 입증 어려움**. fold1·fold2 (코로나·인상기) 에서 Sharpe 음수.
→ 단, **모델 자체의 DM test 우위 (Pooled DM=-8.78, p<0.0001) 는 유효**.

#### 솔직한 결론

- ✅ **모델 정확도 통계 우위**: DM Bonferroni 통과
- ⚠️ **거래비용 마진 얇음**: carry+cost 흡수 시 alpha 사라짐
- 💡 **가장 robust 전략**: S2 ConfFilter (확신 있을 때만 매매)
- 📌 발표 메시지: "모델은 통계적으로 정확하지만 거래비용 마진은 얇다."

산출물:
- 1차 단순: `reports/no_leak_v2/backtest_v2.csv`, `scripts/23_backtest_v2.py`
- 2차 정직: `reports/no_leak_v2/backtest_v2_advanced.csv`, `backtest_v2_walkforward.csv`, `backtest_v2_sharpe_ci.csv`, `scripts/24_backtest_v2_advanced.py`

---

## 2. 옵션 3 채택 사유 — 검증·반증으로 정직하게 결론

### 시도한 추가 개선 (패키지 A) → 모두 데이터로 반증

| 시도 | 가설 | 검증 결과 |
|---|---|---|
| ACI (Adaptive Conformal) | Coverage 90% 안정 회복 | Coverage 96.2% over-correction, PI 폭 +30% 악화 ❌ |
| Rolling vol 입력 추가 | 첫 충격 보호 | 코로나 첫 12일 hit 91.7% → 75% 악화 ❌ |
| 코로나기 강건성 주장 | "fold1 첫 12일 hit 91.7% 입증" | binomial p=1.0 (n=12 너무 작음), 통계 의미 없음 ❌ |

→ **데이터로 반증된 개선안은 채택 X** (끼워맞추기 회피).
→ 검증 과정 자체(`scripts/19·20·21_verify_*.py`)가 학술 자세 사례로 발표 자산화.

---

## 3. 발표 자료 정정 사항 (적용 완료)

### 본문 변경
- ✅ `presentation_v2_revisions.md` §4 한계 인정에 검증 결과 반영
- ✅ `presentation_v2_revisions.md` §4.1 신설: "검증·반증 history" 차별화
- ✅ `presentation_v2_revisions.md` §5 Q&A에 4개 추가:
  - "fold1 코로나에서 OOD 강건성?" → 통계 의미 없음 솔직 인정
  - "ACI 시도?" → 반증 history 보고
  - "그러면 진짜 강점은?" → 누수 정정·DM 우위·walk-forward 안정성
- ✅ `domain_knowledge.md` Q19a, Q19b 추가
- ✅ `comparison_v0_v1_v2.md` §8a 신설: v2 → v2+ 시도·반증

### "강건성" 과장 표현 제거
- ❌ Before: "fold1 코로나기 첫 12일 hit 91.7%로 OOD 강건성 입증"
- ✅ After: "n=12 통계 의미 도출 불가, train의 비슷한 변동성 사건들(2011·2013·2016) 일반화로 일정 부분 대응"

---

## 4. 발표 핵심 멘트 3줄 요약

### 강점 (말할 것)
> "누수 발견·정정·재평가로 학술 합격선 53% 대비 +8.78%p (Pooled 61.78%)를 달성했고, walk-forward 3-fold + DM test (Bonferroni 통과)로 안정성·통계 우위까지 입증했습니다."

### 한계 (정직하게 말할 것)
> "fold1 코로나기는 모델이 정말 OOD에 강건한지 단정 어렵습니다. ACI·rolling vol 추가 개선을 시도했지만 데이터로 반증돼 채택하지 않았고, 후속 작업으로 GARCH·regime-switching 모델 검토 예정입니다."

### 학술 자세 (안내문 §9 직격)
> "1차 결과 65%가 학술 합격선 +12%p로 비현실적이라 의심하고 검증해서 누수를 발견·정정했고, 추가 개선안도 데이터로 반증해 채택하지 않았습니다. 모든 검증 스크립트와 결과는 저장소에 그대로 남겨뒀습니다."

---

## 5. 마지막 체크리스트 (발표 직전)

- [x] v2 결과 csv 모두 검증 완료 (`verification_v2.csv` 13/13 PASS)
- [x] 패키지 A 시도·반증 history 정리 (`package_a_verification.md`)
- [x] 코로나 강건성 정직 검증 (`covid_robustness_check.md`)
- [x] 발표 자료 정정 가이드 갱신 (`presentation_v2_revisions.md`)
- [x] 종합 비교 보고서 갱신 (`comparison_v0_v1_v2.md`)
- [x] domain_knowledge.md Q&A 추가 (Q19a, Q19b)
- [ ] **PPT 직접 정정** — `presentation_v2_revisions.md` 가이드대로 사용자가 작업
- [ ] **대본 직접 정정** — 동일
- [ ] **노트북 08 마지막 셀 점검** — Jupyter 직접 열어 figures 확인

---

## 6. 산출물 위치 일람

### 발표 자료 작업용
- `reports/no_leak_v2/presentation_v2_revisions.md` — **PPT/대본 정정 가이드 (1순위)**
- `reports/no_leak_v2/figures/05_summary_4panel.png` — **발표용 1페이지 요약 그림**

### 종합 보고서
- `reports/no_leak_v2/FINAL_option3_summary.md` — **본 문서**
- `reports/no_leak_v2/comparison_v0_v1_v2.md` — v0/v1/v2 종합 비교

### 검증·반증 기록
- `reports/no_leak_v2/verification_v2.csv` — v2 13/13 PASS
- `reports/no_leak_v2/package_a_verification.md` — 패키지 A 반증
- `reports/no_leak_v2/covid_robustness_check.md` — 코로나 강건성 검증

### 결과 csv (학술 인용용)
- `reports/no_leak_v2/xgb_v2_eval.csv`
- `reports/no_leak_v2/walkforward_xgb_v2.csv`
- `reports/no_leak_v2/walkforward_dm_v2.csv`
- `reports/no_leak_v2/leakage_audit_v2.csv`

### 코드 (재현성)
- `scripts/14_features_v2.py` ~ `scripts/21_verify_covid_robustness.py` (8개)
- `notebooks/08_v2_no_leak_pipeline.ipynb`

### 문서
- `docs/data_leakage_checklist.md` (CL-05c 추가)
- `docs/features_v2_justification.md` (새 변수 정당화)
- `docs/domain_knowledge.md` (Q19a, Q19b 추가)

---

## 7. 한 줄 결론

> **"누수 발견·정정 + 효과 없는 개선 폐기 + 정직한 한계 인정"의 3중 학술 자세로 학술 합격선 +8.78%p 달성한 v2 모델을 발표한다.**

verification-before-completion 정신: 모든 주장 fresh 명령으로 검증, 반증되면 폐기. 끼워맞추기 회피.

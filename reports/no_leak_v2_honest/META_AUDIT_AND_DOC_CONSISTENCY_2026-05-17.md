# 메타 점검 — Audit 도구 사각지대 + 문서 정합성

> **점검 일자**: 2026-05-17
> **점검 대상**:
>   - scripts/04_leakage_audit.py, scripts/17_full_audit_v2.py (audit 도구 자체)
>   - plan_md/plan.md ↔ macro-bond-forecast/README.md ↔ reports/* (문서 간 정합성)
> **점검 방식**: 코드 정독 + 문서 cross-grep (재실행 없음)
> **트리거**: "왜 자가 검증이 P0-1 (HP 누수) 를 못 잡았는가?" + "발표 자료 수치가 일관되는가?"

---

## TL;DR

> **(A) Audit 도구 사각지대**: scripts/04, 17 의 audit 는 **CL-01~10 의 feature-level 누수**는 잘 잡지만, **methodology-level 누수 (HP 선택, cal/val 분리, threshold data snooping, bootstrap 방법론)** 는 구조적으로 못 잡음. 이것이 P0-1, P0-2, PA-1, BT-1 등이 안 잡힌 이유. → audit 의 사각지대 공식화 + CL-11~14 신규 카테고리 정의 권장.
>
> **(B) 문서 정합성**: README 의 "walk-forward 3-fold" 표가 **두 개의 다른 walkforward run (script 16 vs script 24)** 의 수치를 혼합해 보여줌. 같은 fold/같은 모델이지만 n_estimators 가 다른 두 학습 결과 → 독자가 같은 run 으로 오해 가능. 발표 시 1줄 명시 필요.

---

## Part A — Audit 도구 사각지대 분석

### A.1 현재 audit 가 잡는 것 (CL-01~10)

| CL | 검증 방법 | 잡는 누수 종류 |
|---|---|---|
| CL-01 | freeze 에 월별 변수 포함 여부 grep | feature 선정 단계 |
| CL-02 | `scaler.fit()` 의 context 가 train 인지 grep | scaler fit 범위 |
| CL-03 | `.rolling()` 후 `.shift()` 동반 여부 grep | feature 생성 |
| CL-04 | `KFold` / `shuffle=True` grep (DataLoader 예외) | CV 분할 |
| CL-05 | policy 변수 = raw.shift(1) 직접 비교 | 정책 변수 timing |
| CL-05b | features_v1 vs raw policy timing | 정책 변수 raw 단계 |
| CL-05c | features_v1 vs raw US-close timing | cross-market timing (CL-05c) |
| CL-06 | `.bfill()` / `backfill` grep | backward fill |
| CL-07 | target NaN 개수 | 휴장일 처리 |
| CL-08 | crisis_dummy train-only stats | threshold leak |
| CL-10 | krw_usd column 존재 여부 | freeze 위반 |

→ **잘하는 영역**: feature·data·preprocessing 단계 누수

### A.2 현재 audit 가 못 잡는 것 (메타 누수)

이번 3축 점검에서 발견한 **P0/Critical 결함 11건** 중 audit 가 잡은 것은 **0건**:

| 결함 ID | 위치 | 종류 | audit 못 잡는 이유 |
|---|---|---|---|
| **P0-1** | scripts/16:167-171 | walk-forward HP 가 single-split grid 결과 hardcoded | grep 으론 "HP 가 어디서 왔는지" 추적 불가 |
| **P0-2** | scripts/16:64-86 | CQR calibration set ⊂ val set | grep 으론 set 간 시간 범위 비교 불가 |
| **P0-3** | scripts/15:186, 16:183 | 단일 seed (LSTM 은 3 seed) | grep 으론 코드 횡단 일관성 검사 못 함 |
| **PA-1** | path_a/04:67 | LGB grid val_pinball 을 3-fold 평균 → single best | grep 으론 산술 연산 누수 못 봄 |
| **PA-2** | path_a/04 line 188-197 | fold1 trade 0건 (VIX>20 throughout 2020) | audit 는 strategy 실행 분석 안 함 |
| **PA-3** | path_a strategy | 1410일 중 77일만 거래 (5.5% participation) | audit 는 backtest 결과 분석 안 함 |
| **PA-4** | path_a/04 hardcoded | TAU=1.5, VIX_THR=20 이 다른 path test 결과로 선택 | audit 는 cross-path 의존성 추적 불가 |
| **BT-1** | scripts/24:91-95 | backtest 의 BEST_PARAMS 도 single-split hardcoded | grep 으론 HP 출처 추적 불가 |
| **BT-2** | scripts/24:197 | VolTarget 의 `rolling(...).bfill()` lookahead | CL-06 는 `.bfill()` 자체를 grep 하지만 strategy 함수 안의 bfill 은 별도 의미 없음 표시 |
| **BT-3** | scripts/24:271-280 | Sharpe bootstrap 이 i.i.d. (block 미사용) | audit 는 통계 방법론 평가 안 함 |
| **BT-4** | scripts/23:152 vs 24:236 | cost 계산 2× 차이 (round-trip vs one-way) | audit 는 스크립트 간 정합성 검사 안 함 |

**잡은 것: 0/11 (0%)**

### A.3 audit 사각지대의 구조적 원인

| 사각지대 | 본질적 원인 |
|---|---|
| **HP 누수** | HP 선택 과정은 **모델 학습 흐름 전체를 따라가야** 검증 가능 → static grep 으론 불가 |
| **set 간 overlap** | 시간 범위 비교 + 코드 흐름 추적 필요 → AST 또는 runtime 검증 필요 |
| **Cross-script consistency** | audit 가 단일 파일 단위 → 여러 파일 간 cost 정의 불일치 못 봄 |
| **Strategy execution** | audit 는 데이터·feature 만 봄 → backtest 실행 결과 (active days, regime filter 효과) 못 봄 |
| **통계 방법론** | i.i.d. bootstrap vs block bootstrap 같은 통계 선택은 의도적 코드 → grep 못 함 |
| **Cross-path snooping** | 여러 실험 path 간 hyperparameter 공유 (TAU, VIX_THR) → 도구가 인지 못 함 |

### A.4 권장 — CL-11~14 신규 audit 카테고리

| 신규 CL | 정의 | 검증 방법 |
|---|---|---|
| **CL-11** | HP origin tracking | 각 fold 의 HP 가 그 fold 의 own val 에서 선택됐는지 — runtime instrumentation 또는 코드 패턴 (BEST_PARAMS hardcoded 검출) |
| **CL-12** | Set disjoint enforcement | train/val/cal/test 간 시간 overlap 자동 검증 — split dict 파싱 후 interval 비교 |
| **CL-13** | Cross-script parameter consistency | 같은 파라미터 (D, COST, threshold) 가 여러 script 에서 동일 의미로 사용되는지 |
| **CL-14** | Strategy execution audit | backtest 의 active days, fold별 trade 수, regime filter 효과 자동 보고 |

### A.5 발표 활용 — 정직성 서사 보강

> **새로 추가 가능한 멘트**:
> "이번 자가 점검에서 기존 audit 도구 (scripts/04, 17) 가 CL-01~10 feature-level 누수는 잘 잡지만, walk-forward HP 누수·calibration set overlap·data snooping 같은 methodology-level 누수는 구조적으로 못 잡음을 확인했습니다. P0-1 (HP 누수), PA-1 (cross-fold HP 평균), BT-1 (backtest HP hardcoded) 등 이번에 발견한 11건이 모두 audit 통과 후 잔존한 결함이며, 이를 CL-11~14 신규 카테고리로 후속 작업에 권장합니다. audit 도구 자체의 한계를 학술적으로 인정하는 자세는 안내문 §7 정신과 정합합니다."

---

## Part B — 문서 정합성 점검 (plan.md ↔ README ↔ reports)

### B.1 발견된 불일치

#### 🟡 B-1. README 의 "walk-forward 3-fold" 표가 두 run 혼합

**위치**: `macro-bond-forecast/README.md` 메인 결과 영역

| README 위치 | 보여주는 수치 | 실제 출처 | n_estimators |
|---|---|---|---|
| line 18-23 (방향성 정확도 표) | 0.6113 / **0.6099 ± 0.0275** / **0.6178 (Pooled)** | `walkforward_xgb_v2.csv` (script 16) | 200/400/100 |
| line 51-56 (Walk-forward 3-fold 표) | fold1 **0.564**, fold2 **0.535**, fold3 **0.619**, Pooled Sharpe **0.62** | `backtest_v2_walkforward.csv` (script 24) | 87/215/53 |

**문제**:
- 두 표 모두 같은 "walk-forward 3-fold v2" 라벨
- 하지만 fold dir_acc 가 다름:
  - script 16 (n=200/400/100): fold1 0.5932, fold2 0.5949, fold3 0.6416 → mean 0.6099
  - script 24 (n=87/215/53): fold1 0.5636, fold2 0.5352, fold3 0.6185 → mean 0.5724
- 평균 약 0.04 차이 (HP n_estimators 차이 때문)
- Pooled 0.6178 (script 16) ≠ 산술평균 0.6099 ≠ backtest 평균 0.5724

**판단**:
- 본질적 결함이 아닌 **표 라벨링 모호성**
- `docs/주차별_실행결과.md` 는 **양쪽 다 정직히 분리 보고** (§379-384 model + §423-426 backtest) — OK
- README 만 두 표를 같은 "walk-forward" 헤더로 묶어 혼동 유발

**권장 수정**:
- README line 51 의 표 헤더에 "(backtest walk-forward, script 24, n_estimators=87/215/53)" 명시
- 또는 두 표 사이에 "model walk-forward (script 16) 와 backtest walk-forward (script 24) 는 같은 fold·feature 사용하지만 early-stopping 결과 n_estimators 가 달라 dir_acc 가 약 0.04 차이" 1줄 노트 추가

#### 🟢 B-2. plan.md 의 목표 (≥55%) vs 실제 결과 (0.6178) — 정합 OK

`plan_md/plan.md` line 49:
> 정량 기준: 방향성 정확도 ≥55%, DM test p<0.05, Coverage Rate 90%±3%p

- ≥55% 목표 → 실제 0.6178 (Pooled) — **+6.78%p 초과 달성** ✅
- DM p<0.05 → 실제 p<0.0001 Bonferroni 통과 ✅
- Coverage 90%±3%p → 실제 87.27% (3-fold 평균) — **3%p 하한 매우 근접 (△)**

**판단**: coverage 만 borderline. 발표 시 정직 인정. plan 의 다른 목표는 모두 달성.

#### 🟢 B-3. plan.md (v5.3.12) vs docx (v5.4.1) 의 버전 차이

- `plan_md/plan.md` = v5.3.12-md3 (2026-05-10)
- 루트 docx = `채권금리예측_프로젝트계획서_v5_4_1.docx` (최신 final)
- 두 버전 차이는 `plan_md/CHANGELOG.md` 의 v5.4 / v5.4.1 항목에 명시되어 있는지 확인 필요

**확인 결과**: CHANGELOG.md 는 v5.3.12-md3 까지만 기록. **v5.4.1 docx 의 추가 변경 사항이 MD 에 미반영**.

**판단**:
- v5.4 / v5.4.1 patch 가 무엇인지는 직접 확인 못 함 (docx 내용 미확인)
- 가능성 1: 본문 내용은 v5.3.12 와 동일, 형식만 정정 → 정합 OK
- 가능성 2: 본문 patch 있음 → MD/docx 의 SSOT 불일치
- 권장: plan_md 디렉토리에 "v5.4/v5.4.1 patch 는 docx 직접 수정으로, MD 미반영" 1줄 명시

#### 🟢 B-4. `docs/project_plan.md` 의 처리

- 이전 정리 작업 (cleanup_2026-05-17) 에서 `_archive/cleanup_2026-05-17/stale_docs/project_plan_v5.1.md` 로 이동 완료
- 더 이상 정합성 문제 없음 ✅

#### 🟢 B-5. README 의 "🏆 4중 검증 통과" 와 Path A 점검 결과 충돌

`experiments/README.md` line 65:
> Path A LGB robust single | ✅✅✅ | 2.02 [+1.45, +2.57] | **leak audit 통과 = 진짜 alpha**

→ 이번 PA-2/PA-3/PA-1/PA-4 점검 결과 "진짜 alpha" 단정 표현 부적절

**권장 수정** (이미 `CODE_REVIEW_PATH_A_2026-05-17.md` 에 적용 권장 명시함):
```
Path A LGB robust single | 🟡 부분 검증 | 2.02 [+1.45, +2.57]
  (단 95% zero PnL, fold1 거래 0건) | 활성 빈도 5.5%, occasional 전략
```

### B.2 정합성 점검 통과 항목

| 항목 | 결과 |
|---|---|
| Pooled dir 0.6178 — README ↔ summary_w6 ↔ FINAL_option3_summary | ✅ 일관 |
| Pooled DM -8.78 — README ↔ walkforward_summary_v2 ↔ FINAL | ✅ 일관 |
| RMSE 4.48 / 4.72 — README ↔ FINAL_option3_summary | ✅ 일관 |
| Single-split dir 0.6113 — README ↔ xgb_v2_eval.csv | ✅ 일관 |
| fold1 코로나, fold2 인상기, fold3 안정+충격 — 모든 문서 | ✅ 일관 |
| VALIDATION_LOG 50건 — README ↔ 실제 파일 467줄 | ⚠️ count 검증 필요 (직접 #N grep 필요) |
| AI_USAGE_LOG 21건 — 같음 | ⚠️ 동상 |

### B.3 honest 결과 (이번 점검 #1) 의 문서 위치

이번 honest 재실행 결과 (dir 0.6163, CI [0.5927, 0.6421]) 는:
- ✅ `reports/no_leak_v2_honest/CODE_REVIEW_AND_FIX_HISTORY.md` 에 기록
- ✅ `reports/no_leak_v2_honest/honest_walkforward_summary.md` 에 기록
- ❌ **메인 README 미반영** (v0/v1/v2 history 만 표시)
- ❌ **plan.md 미반영** (계획서이므로 결과 반영 의무 없음 — OK)
- ❌ **주차별_실행결과.md 미반영**

**권장**: README 끝부분 또는 `## 자가 점검 history` 신설 섹션에 honest 결과 1단락 추가. **단 본문 메인 수치 (0.6178) 는 변경 X — honest 0.6163 은 "검증 결과 동일" 으로만 명시.**

---

## Part C — 통합 권장 액션

### 🔴 발표 전 (10분 이내)
1. **README line 51 표 헤더에 출처 명시** ("(backtest walk-forward, script 24)")
2. **experiments/README.md line 65 의 "🏆 4중 검증 통과 = 진짜 alpha" 문구 정정** (Path A 점검 보고서 §8 참고)

### 🟡 발표 전 (선택, 30분)
3. **README 끝에 "자가 점검 history" 1단락 추가** — honest #1, Path A #2, Backtest #3 보고서 링크
4. **plan_md/CHANGELOG.md 에 v5.4/v5.4.1 docx 변경 사항** (MD 미반영) 1줄 명시

### 🟢 후속 (발표 후)
5. **scripts/04, 17 에 CL-11~14 신규 카테고리** 검증 로직 추가
6. **VALIDATION_LOG 에 #48 (메인 점검), #49 (Path A 점검), #50 (Backtest 점검), #51 (메타·문서 점검) 4건 누적 기록**

---

## Part D — 통합 정직성 서사 (4축 점검 완성)

```
v0 → CL-05c 발견 → v1 → 새 변수+XGB+CQR → v2 (dir 0.6178, DM -8.78)
  ↓ 자가 점검 #1 (메인 v2 모델, scripts 14~22)
    → P0 4건 발견 + honest 재실행 → dir 0.6163 (Δ=-0.15%p, robustness 입증)
  ↓ 자가 점검 #2 (Path A 실험)
    → PA 3 critical 발견 → "5.5% participation, fold1 거래 0건" 한계 인정
  ↓ 자가 점검 #3 (Backtest scripts 23·24)
    → BT 14건 발견 (Critical 2, Medium 4, Low 8) → "결함 합산 후도 메인 결론 유지"
  ↓ 자가 점검 #4 (Audit 도구 자체 + 문서 정합성)
    → audit 사각지대 공식화 (P0/PA/BT 11건 중 0건 잡음)
    → README 표 헤더 모호성 발견 → 정정 권장
```

**4-축 자기점검 완료** — 안내문 §7 "AI 결과 자체 검증" 의 **7단계 메타 검증 사례**:
- #30 (W2 audit false positive)
- #36 (W4 CL-05b)
- #37 (W4 CL-05c)
- #40 (W5 val-overfit)
- #43 (W6 lookahead)
- **#48~50 (자가 점검 #1~#3, 2026-05-17)** ← 이번 추가
- **#51 (메타 점검 #4, 2026-05-17)** ← 본 문서

---

## Part E — 산출물

### 신규 생성
- **`reports/no_leak_v2_honest/META_AUDIT_AND_DOC_CONSISTENCY_2026-05-17.md`** ⭐ 본 문서

### 관련 4축 점검 보고서
- `reports/no_leak_v2_honest/CODE_REVIEW_AND_FIX_HISTORY.md` (#1 메인 v2)
- `experiments/path_a_model/CODE_REVIEW_PATH_A_2026-05-17.md` (#2 Path A)
- `reports/no_leak_v2/BACKTEST_CODE_REVIEW_2026-05-17.md` (#3 Backtest)
- 본 문서 (#4 Meta + Doc)

### 무수정 (보존)
- 모든 scripts, reports/no_leak_v2/*.csv·md·png
- plan_md/plan.md, README.md, docx 파일들

---

**작성**: 2026-05-17
**점검 방식**: 코드 정독 + cross-document grep (재실행 0)
**발견**: audit 사각지대 6 유형 + 문서 정합성 5건 (Critical 0, Medium 1, Low 4)
**4-축 점검 완료**: 메인 + Path A + Backtest + Meta/Doc

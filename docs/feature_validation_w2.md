# 2주차 변수 검증 결과 — 1주차 freeze 9개의 데이터 정당화

> 계획서 v5.1 §3.2(a) 안전장치 — 산출물 3건 사전 마감
> 생성: notebooks/02_feature_selection.ipynb

## 1. 상관 분석 (다중공선성 후보)

| 변수 1 | 변수 2 | r |
|--------|--------|------|
| vix | sp500 | -0.734 |

## 2. VIF 분석

| 변수 | VIF | 상태 |
|------|-----|------|
| vix | 2.28 | ✅ 정상 |
| sp500 | 2.24 | ✅ 정상 |
| us_breakeven_10y | 1.38 | ✅ 정상 |
| us_treasury_10y | 1.35 | ✅ 정상 |
| dxy | 1.18 | ✅ 정상 |
| kospi | 1.11 | ✅ 정상 |
| kr_treasury_3y | 1.04 | ✅ 정상 |
| us_fed_funds | 1.01 | ✅ 정상 |
| kr_base_rate | 1.0 | ✅ 정상 |

결론: 0개 다중공선성 의심, 0개 주의.

## 3. Granger 인과성 (max lag 5)

| 변수 | best_lag | p-value | 유의 |
|------|----------|---------|------|
| kr_treasury_3y | 3 | 0.0 | ✅ |
| us_treasury_10y | 3 | 0.0 | ✅ |
| us_breakeven_10y | 1 | 0.0 | ✅ |
| dxy | 1 | 0.0 | ✅ |
| kr_base_rate | 5 | 0.001 | ✅ |
| us_fed_funds | 5 | 0.0039 | ✅ |
| sp500 | 5 | 0.0153 | ✅ |
| vix | 5 | 0.0539 | ❌ |
| kospi | 1 | 0.5705 | ❌ |

결론: 7개 변수가 p<0.05 로 타겟을 선행.

## 4. 종합 점수 (만점 3)

| 변수 | 상관 \|r\| | VIF | best_lag | Granger p | 점수 |
|------|----------|-----|----------|-----------|------|
| kr_treasury_3y | 0.81 | 1.04 | 3 | 0.0 | 3 |
| us_treasury_10y | 0.157 | 1.35 | 3 | 0.0 | 3 |
| us_breakeven_10y | 0.066 | 1.38 | 1 | 0.0 | 3 |
| dxy | 0.063 | 1.18 | 1 | 0.0 | 3 |
| kr_base_rate | 0.016 | 1.0 | 5 | 0.001 | 2 |
| us_fed_funds | 0.016 | 1.01 | 5 | 0.0039 | 2 |
| vix | 0.051 | 2.28 | 5 | 0.0539 | 2 |
| sp500 | 0.026 | 2.24 | 5 | 0.0153 | 2 |
| kospi | 0.043 | 1.11 | 1 | 0.5705 | 1 |

## 5. 3주차 freeze 확정 권고

- **점수 3 (강력 채택)**: 4개 — `kr_treasury_3y`, `us_treasury_10y`, `us_breakeven_10y`, `dxy`
- **점수 1 이하 (제거 검토)**: 1개 — `kospi`

## 6. 5주차 ablation 대상 사전 명시 후보

- 환율 ablation 1개 필수 (계획서 §3.2(b))
- 추가 ablation 후보: 점수 낮은 변수 → `kospi`

## 7. 누수 차단 체크리스트 — 감사 스크립트 강화

> 2주차 후반 자동 감사(`reports/leakage_audit_w2.csv`) 검토 결과, 초기 인라인
> 로직(`02b_preprocess_baseline.ipynb` cell 24)에서 ❌ 4건이 모두 **false positive**
> 였음을 확인. 실제 코드는 모두 누수-안전. `scripts/04_leakage_audit.py` 로 분리하고
> 다음을 강화 후 재실행 → **7/7 ✅ 통과**.

### 7.1 false positive 원인 (수정 전)

| CL | 보고 | 실제 | 원인 |
|----|------|------|------|
| CL-02 | ❌ line 336 | ✅ | grep 이 마크다운 셀 설명문(`` `scaler.fit()` 은 Train 구간에만… ``)을 매치 |
| CL-04 | ❌ line 582 | ✅ | grep 이 감사 코드 자체의 주석(`# CL-04 K-fold / shuffle=True 금지`)을 매치 |
| CL-05 | ❌         | ✅ | 검증 비교가 `features_safe[v].iloc[0] != features_v1[v].iloc[0]`. shift(1)+dropna 후 첫 행은 원본 두 번째 영업일 값. 정책금리는 영업일 단위로 거의 동일값 → 비교 항상 False |
| CL-06 | ❌ line 607 | ✅ | grep 이 감사 코드 자체의 정규식 패턴(`r'\.bfill\('`)을 매치 |

### 7.2 강화 사항 (`scripts/04_leakage_audit.py`)

1. **`.ipynb` JSON 파싱** → `cell_type == 'code'` 만 스캔 (마크다운 제외)
2. **`#` 주석 줄 + `grep_repo(` 메타 코드 줄 + 감사 스크립트 자기 자신** 제외
3. **CL-05 직접 비교**: `features_v1_candidate.csv` 의 정책 변수와
   `features_with_lags_v1.csv` 의 동일 컬럼이 `shift(1)` 관계인지 직접 검증
   (3,726 행 모두 일치)

### 7.3 최종 결과 (수정 후, 2026-05-02)

| CL | 상태 | 비고 |
|----|------|------|
| CL-01 | ✅ | freeze 에 월별 변수 없음 → 본 검증 범위 N/A |
| CL-02 | ✅ | 모든 scaler.fit() 이 X_train/train 변수에 한정 |
| CL-03 | ✅ | 모든 .rolling() 호출이 같은 줄에 .shift() 동반 |
| CL-04 | ✅ | KFold/shuffle=True 사용 없음 |
| CL-05 | ✅ | 정책 변수 2개 모두 features_v1.shift(1) 와 일치 (3,726 rows) |
| CL-06 | ✅ | bfill/backfill/양방향 보간 사용 없음 |
| CL-07 | ✅ | features_v1_candidate.csv 타겟 결측 0건 |

**종합: 7/7 ✅** — 3주차 freeze 진입 가능.

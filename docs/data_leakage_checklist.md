# 데이터 누수 차단 체크리스트

> 시계열 ML 에서 **데이터 누수(data leakage)** 는 결과를 무효로 만드는 가장 큰 위험이다. 본 체크리스트는 모든 PR 머지 전 리뷰어가 점검한다.
>
> 운영 방식:
> 1. 데이터·모델 관련 PR 작성 시 **PR description 에 본 체크리스트 복붙**.
> 2. 작성자가 1차 자가 점검 → 리뷰어가 2차 점검.
> 3. 모든 🔴 항목 통과해야 머지 가능.
> 4. 검증 결과는 `VALIDATION_LOG.md` 에 기록.

---

## 🔴 필수 (안 하면 결과 무효)

### CL-01. 월별 변수 발표일 시프트
- [ ] CPI(`kr_cpi`, `kr_cpi_core`, `us_cpi`), PPI(`kr_ppi`), 산업생산(`kr_industrial_prod`), BSI(`kr_mfg_bsi_outlook`) 모두 **+1개월 이상 lag** 적용?
- **사유**: CPI 1월분은 보통 2월 5일경 발표. 1월 31일에 1월 CPI 를 input 으로 쓰면 발표 전 정보 누수.
- **검증 코드**:
  ```python
  # 발표일 시프트 후 첫 영업일까지 NaN 인지 확인
  assert df["kr_cpi"].iloc[0:25].isna().all(), "CPI shift not applied"
  ```

### CL-02. Scaler train-only fit
- [ ] `scaler.fit(X_train)` 만 호출, val/test 는 `scaler.transform()` 만?
- [ ] 전체 데이터로 fit 후 split 하는 코드 없음?
- **사유**: 전체 fit 시 val/test 평균/분산이 모델 학습에 누설.
- **검증 코드**:
  ```bash
  grep -nE "scaler\.fit\(.*[^_]\)" src/ notebooks/ scripts/
  # 모든 출력이 X_train 또는 train_data 인지 확인
  ```

### CL-03. Lag/Rolling 시 현재 시점 미포함
- [ ] 모든 lag feature 는 `df[col].shift(k)` (k≥1)?
- [ ] 모든 rolling 은 `df[col].rolling(w).agg(...).shift(1)` 패턴?
- **사유**: `rolling(5).mean()` 은 **현재 시점 포함** → 누수.
- **검증 코드**:
  ```bash
  # rolling 사용처 중 shift 안 붙은 것 찾기
  grep -nE "\.rolling\(" src/ notebooks/ | grep -v "\.shift("
  # 출력 없어야 함
  ```

### CL-04. Cross-validation 방식
- [ ] `sklearn.model_selection.KFold` 사용 안 함?
- [ ] `train_test_split(..., shuffle=True)` 사용 안 함?
- [ ] `TimeSeriesSplit` 또는 expanding window walk-forward 사용?
- **사유**: K-fold/random split 은 미래로 과거 예측 = 누수.
- **검증 코드**:
  ```bash
  grep -nE "KFold|shuffle\s*=\s*True" src/ notebooks/
  # 출력 없어야 함 (StratifiedKFold 등도 시계열엔 부적합)
  ```

### CL-05. 정책 변수 lag 1 강제
- [ ] `kr_base_rate`, `us_fed_funds` 가 항상 `t-1` 값으로 사용?
- [ ] feature 생성 시 `df["kr_base_rate"] = df["kr_base_rate"].shift(1)` 적용?
- **사유**: 한은 금통위/FOMC 발표 당일 결과를 input 으로 쓰면 누수. 발표는 보통 장중 또는 마감 후.
- **검증 코드**: feature 생성 코드의 정책 변수 처리 부분 직접 확인.

---

## 🟠 권장 (놓치면 점수 깎임)

### CL-06. Backward fill / 양방향 보간 금지
- [ ] `bfill()`, `backfill()`, `interpolate(limit_direction='both' or 'backward')` 사용 안 함?
- **사유**: 미래 정보로 과거 결측을 채우는 행위 = 누수.
- **검증 코드**:
  ```bash
  grep -nE "\.bfill\(|backfill|limit_direction.*['\"]both['\"]|limit_direction.*['\"]backward['\"]" src/ notebooks/
  # 출력 없어야 함
  ```

### CL-07. 한국 휴장일의 타겟 처리
- [ ] 한국 휴장일(`kr_treasury_10y` 결측일)의 행이 학습/평가 샘플에서 **drop** 되어 있음?
- [ ] forward fill 로 가짜 타겟 생성하지 않음?
- **사유**: 추석·설 등 한국 휴장일에는 국고채 데이터 자체가 없음. 채우면 가짜 라벨로 학습.
- **검증 코드**:
  ```python
  # 학습 샘플에서 타겟이 원본 데이터 존재일에서만 추출되는지 확인
  raw_dates = pd.read_csv("data/raw/raw_ecos.csv")
  target_dates = raw_dates[raw_dates["variable"] == "kr_treasury_10y"]["date"]
  assert set(train_y.index).issubset(set(pd.to_datetime(target_dates))), "Synthetic targets detected"
  ```

---

## 🟡 디테일 (강건성)

### CL-08. 모든 통계량은 train 범위
- [ ] imputation 평균값, 이상치 제거 임계값(IQR 등) 모두 train 통계로만?
- **사유**: val/test 평균을 보면 누수.

### CL-09. 휴장일 직후 분포 차이
- [ ] 추석/설 직후 영업일에 미국 변수가 며칠치 누적되어 있음을 인식하는 feature(`days_since_last_kr_bday`) 추가 검토?
- **사유**: 평소와 분포가 다른 시점을 모델이 인식하지 못하면 일반화 저하.
- **선택**: 7주 일정상 우선순위 낮음. 시간 여유 시 추가.

### CL-10. 환율(`krw_usd`) 분리
- [ ] 환율은 검증용으로만 수집되고, 모델 입력 feature 에 포함되지 않음?
- [ ] 환율 분석은 학습 끝난 모델의 사후 SHAP/상관 분석에서만?
- **사유**: 정부 개입 구조 때문에 입력에서 제외하기로 결정 (계획서 §3.3). 실수로 포함 시 결정 무효화.

---

## PR description 템플릿

```markdown
## 변경 사항
- ...

## 데이터 누수 체크리스트
- [ ] CL-01 월별 변수 발표일 시프트
- [ ] CL-02 Scaler train-only fit
- [ ] CL-03 Lag/Rolling 현재 시점 미포함
- [ ] CL-04 Cross-validation (TimeSeriesSplit)
- [ ] CL-05 정책 변수 lag 1
- [ ] CL-06 Backward fill 금지
- [ ] CL-07 한국 휴장일 타겟 drop
- [ ] CL-08 통계량 train-only
- [ ] CL-09 휴장일 직후 feature (선택)
- [ ] CL-10 환율 분리

## 검증 로그
VALIDATION_LOG.md #N 에 기록함.
```

---

## 한 번에 자동 점검하는 명령

저장소 루트에서 실행:

```bash
# CL-02, CL-03, CL-04, CL-06 동시 점검
{
  echo "=== CL-02 Scaler ==="
  grep -nE "scaler\.fit\(" src/ notebooks/ scripts/ 2>/dev/null
  echo ""
  echo "=== CL-03 Rolling without shift ==="
  grep -nE "\.rolling\(" src/ notebooks/ scripts/ 2>/dev/null | grep -v "\.shift("
  echo ""
  echo "=== CL-04 Forbidden CV ==="
  grep -nE "KFold|shuffle\s*=\s*True" src/ notebooks/ scripts/ 2>/dev/null
  echo ""
  echo "=== CL-06 Backward fill ==="
  grep -nE "\.bfill\(|backfill|limit_direction.*both|limit_direction.*backward" src/ notebooks/ scripts/ 2>/dev/null
  echo ""
  echo "=== Done ==="
} | tee data_leakage_audit.log
```

이 출력이 비어 있어야(또는 의도된 사용처만 나와야) 통과.

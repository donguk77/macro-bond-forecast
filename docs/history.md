# 프로젝트 작업 히스토리

## [2026-05-28] 🧹 폴더 정리 + 🎯 기말 발표 종합 플랜 작성

### 💬 진행 및 결정 사항 (Discussion)
- **폴더 정리** (사용자 승인: _archive 이동 / 최신 PDF만 유지 / 종합 플랜):
  - 루트 `page*.png` 16개(중간 PPT 슬라이드 추출 임시본) → `_archive/extracts/ppt_slides_midterm/`
  - `docs/` 피치덱 PDF 중복 2개(~105MB): 구버전(49MB, 5/26) → `_archive/midterm_assets/pitch_deck_v1_2026-05-26.pdf`, 최신본(55MB) → `docs/pitch_deck_midterm.pdf`로 rename
  - v1 발표 대본 `docs/midterm_presentation_script.md` → `_archive/midterm_assets/midterm_presentation_script_v1.md`
- **기말 플랜**: 교수님 중간 피드백(VIF 무조건 제거 지양 / SHAP 기여 분산 / 도메인 그룹핑 / 예측·해석 목적별 차등) 4개를 축으로 구성
  - 핵심: 변수 선택을 단일 정량 필터 → **「도메인 그룹핑 + 예측/해석 이원 파이프라인」**으로 재설계
  - 기존 3대 과제(예측구간 축소·위기 앙상블·백테스트)와 통합, 주차별 일정·발표 구성안·Q&A·리스크 포함

### 🛠️ 코드 수정 내역 (Code Changes)
- **Added**: `docs/final_presentation_plan.md` — 기말 발표 종합 플랜
- **Moved**: 위 정리 파일들 → `_archive/`
- **Renamed**: 피치덱 최신본 → `docs/pitch_deck_midterm.pdf`

## [2026-05-26 19:45] 🎯 발표 대본 섹션 1, 12, 13 대규모 보강

### 💬 진행 및 결정 사항 (Discussion)
- **섹션 1(선행연구)**: 기존 6편 나열 → **최신 논문 3편(Gao&Hyndman 2025, Joshi 2020, Nunes et al. 2025) 중심**으로 재구성
  - 각 논문의 모델, 목표, 한계점, 본 연구와의 관계를 상세 비교표로 작성
  - 기존 vs 본 연구 비교를 4열(3편 + 본 연구)로 확장
  - 대본에서 각 논문의 한계를 구체적으로 설명하고 본 연구의 차별점과 연결
- **섹션 12(SHAP)**: 1분 30초 → **2분**으로 확장
  - Nunes et al.의 간접 해석(LagLasso) vs 본 연구의 직접 해석(TreeSHAP) 비교 추가
  - 변수 그룹별 재집계 상세 수치(13개 그룹 × 기여도) 추가
  - Beeswarm 비선형 관계 해석 + delta_vix_t1 위험회피 채널 해석 추가
  - 시계열 SHAP 변동 분석(2023.10 위기 구간) 추가
- **섹션 13(향후 계획)**: 1분 → **2분**으로 확장, 제목 변경 "CQR 한계점 & 향후 계획"
  - CQR 보정 등 CQR 관련 방법론 제외 및 단순 "90% 예측 구간"의 문제로 논조 변경
  - GBoost(XGBoost) 단일 모델의 외삽(Extrapolation) 한계 명시
  - 해결 방법론: AsymVar(Method B) + 위기 구간 전용 혼합(앙상블) 모델 구축 설명 추가
  - 시각화 이미지 2종만 남기고 향후 계획 구체화
- **Q&A**: Q1의 CQR 언급 삭제, Q3의 CQR 질문을 "트리 기반 모델의 위기 대응 한계"로 변경하여 앙상블 계획과 연결
- 합계 약 18분 30초, 조절 포인트 5개로 약 15분 30초까지 축소 가능

### 🛠️ 코드 수정 내역 (Code Changes)
- **Changed**: `docs/midterm_presentation_script.md` — 섹션 1(CQR 용어 삭제), 섹션 13(CQR 전면 제외, GBoost 한계 및 앙상블 계획 추가), Q&A(CQR 삭제 및 위기 질문 대체)

## [2026-05-26 19:29] 🎯 발표 대본 섹션 4(ARIMA), 섹션 12(SHAP) 수정

### 💬 진행 및 결정 사항 (Discussion)
- 섹션 4: ARIMA(1,0,1) 단일 모델 → **35가지 (p,q) 조합 grid search + AIC 최적 ARIMA(2,0,3) 선정**으로 변경
  - auto_arima_search_results.csv, auto_arima_baseline.csv 데이터 반영
  - 결론: 35가지 중 최적도 51.7% → 단변량 시계열 한계 명확 → 다변량 ML 필요 근거
- 섹션 12: "향후 SHAP 계획" → **v2 SHAP 실분석 결과** 기반으로 전면 교체
  - delta_us10y_t1 현재값 = 압도적 1위 (0.4653bp, 2위 대비 8.6배)
  - v0→v2 비교로 누수 수정의 증거를 SHAP으로 입증
  - 변수 그룹별, Beeswarm 비선형 관계, 시계열 분석 시각화 6종 추가
- 시간 배분: 섹션 4 (1분→1분 15초), 섹션 12 (1분→1분 30초), 합계 약 17분

### 🛠️ 코드 수정 내역 (Code Changes)
- **Changed**: `docs/midterm_presentation_script.md` — 섹션 4, 12 전면 재작성 + 시간 배분 테이블 업데이트

## [2026-05-26 19:15] 🎯 15분 중간 발표 대본 작성

### 💬 진행 및 결정 사항 (Discussion)
- 사용자가 지정한 13개 섹션 구조(선행연구 → 변수선택 → 학습구조 → 베이스라인 → XGBoost → LSTM/누수발견 → 누수수정 → 파생변수 → 검증 → v2결과 → 변수정리 → SHAP → 마무리)에 맞춰 대본 작성
- 각 섹션에 🔑 핵심 내용(필수 포함 사항)과 🎤 대본을 분리하여 발표자가 핵심을 놓치지 않도록 구성
- 기존 midterm_presentation.md, midterm_presentation_v2.md, comparison_v0_v1_v2.md 등의 정확한 수치를 교차 검증하여 대본에 반영
- 시간 배분 총 ~16분 15초로, 3개 조절 포인트(선행연구 간략화, 검증 1줄, 변수정리 합치기)로 15분 맞춤 가능
- 예상 질문 & 답변 8건 포함

### 🛠️ 코드 수정 내역 (Code Changes)
- **Added**: `docs/midterm_presentation_script.md` — 13개 섹션 15분 발표 대본 + 핵심 내용 + PPT 구성 제안 + 예상 질문 8건

## [2026-05-26 17:00] 🎯 중간 발표용 시각화 6종 개선 생성

### 💬 진행 및 결정 사항 (Discussion)
- 기존 `reports/figures/` 하위의 발표용 차트가 시각적으로 부족하여, 프리미엄 다크모드 디자인으로 전면 재설계
- 이모지/특수문자(Δ, ŷ, ≈, ✅, ⚠️ 등)가 Malgun Gothic 폰트에서 렌더링되지 않는 문제 발견 → 일반 텍스트로 교체하여 해결
- 베이스라인, XGBoost v0, RMSE 4모델, 누수 수정(v0→v1), 파생변수 분석표, 13변수 검증 등 6종의 차트를 일괄 생성
- CSV 데이터 소스 확인: baseline_results_w3.csv, baseline_results_w5.csv, xgb_quantile_eval_w3.csv, lstm_quantile_eval_w4.csv, leakage_fix_comparison_test.csv

### 🛠️ 코드 수정 내역 (Code Changes)
- **Added**: `scripts/viz_presentation_improved.py` — 발표 시각화 생성 스크립트 (다크모드, 6종 차트)
- **Added**: `reports/figures/improved/` 디렉토리 및 하위 6개 PNG 파일:
  - `baseline_naive_arima_improved.png` — Naive vs ARIMA 3패널 비교
  - `baseline_with_xgb_v0_improved.png` — Naive/ARIMA/XGBoost v0 비교
  - `rmse_4models_v0_improved.png` — 4모델 RMSE/방향정확도/Coverage
  - `v0_v1_leakage_fix_improved.png` — 누수 수정 전/후 비교 (화살표 포함)
  - `derived_features_analysis.png` — 파생변수 5개 도메인 근거 테이블
  - `derived_features_validation_13vars.png` — 13변수 3단계 검증 (상관/VIF/Granger)
- **Fixed**: 이모지 글리프 미지원 문제 → ASCII 텍스트로 교체
- **Added**: `docs/history.md` — 작업 히스토리 파일 생성

# macro-bond-forecast

> **다변량 시계열 딥러닝과 SHAP을 활용한 한국 국고채 10년물 금리 예측**
> 2026학년도 머신러닝 팀 프로젝트 · 7주 · 2~3인

국내외 거시경제 지표(한미 금리, 인플레이션, VIX, KOSPI 등)를 입력으로 한국 국고채 10년물의 다음 영업일 금리를 예측한다. Naive·ARIMA·XGBoost 베이스라인과 LSTM 모델을 비교하고, SHAP으로 변수별 시차 효과까지 정량화하는 것이 목표.

자세한 설계는 `docs/project_plan.md` (계획서) 참고.

---

## 폴더 구조

```
macro-bond-forecast/
├── data/                        # ⚠️  git 제외 (용량/민감)
│   ├── raw/                     # API 원본 (수정 금지)
│   ├── interim/                 # 중간 변환 (wide 통합)
│   └── processed/               # 모델 학습용 최종 (8개 변수 확정 후)
├── notebooks/                   # 번호 prefix 로 실행 순서 명시
│   ├── 01_eda.ipynb
│   ├── 02_feature_selection.ipynb   # 22개 → 8개 축소 정당화
│   ├── 03_baselines.ipynb           # Naive / ARIMA / XGBoost
│   ├── 04_lstm.ipynb
│   └── 05_shap_analysis.ipynb
├── src/                         # 재사용 모듈 (notebook 에서 import)
│   ├── data/                    # 수집 함수
│   ├── features/                # lag / scaling / window
│   ├── models/                  # baseline, lstm
│   └── visualization/
├── scripts/                     # CLI 진입점
│   ├── 01_verify_ecos_codes.py
│   └── 02_collect_data.py
├── configs/                     # 하이퍼파라미터, 경로
├── models/                      # ⚠️  git 제외 (학습된 가중치)
├── reports/                     # 분석 리포트, 그림
├── app/                         # Streamlit 대시보드
├── tests/
├── docs/                        # 팀 역할, 데이터 사전, 계획서
├── prompts/                     # 핵심 AI 프롬프트 (선택)
├── AI_USAGE_LOG.md              # ⚠️  안내문 필수
├── VALIDATION_LOG.md            # ⚠️  안내문 필수
├── requirements.txt
├── .env.example                 # API 키 템플릿
└── .gitignore
```

---

## 빠른 시작

### 1. 클론 & 가상환경

```bash
git clone https://github.com/donguk77/macro-bond-forecast.git
cd macro-bond-forecast

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. API 키 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 본인 키로 교체.

- **ECOS**: https://ecos.bok.or.kr/api/ (즉시 발급)
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html (이메일 인증 후 발급, 32자리)

> ⚠️ `.env` 는 `.gitignore` 에 의해 추적되지 않는다. **절대 키를 코드에 직접 박지 말 것.**

### 3. 데이터 수집

```bash
# (선택) ECOS 항목코드 검증
python scripts/01_verify_ecos_codes.py

# 본격 수집 (ECOS + FRED + yfinance, 2~3분 소요)
python scripts/02_collect_data.py
```

산출물은 `data/raw/`, `data/interim/` 에 저장된다.

### 4. EDA

```bash
jupyter lab
# notebooks/01_eda.ipynb 부터 차례로 실행
```

---

## 협업 규칙

### 브랜치 전략

- `main`: 항상 동작 가능한 상태 유지. 직접 push 금지.
- `feature/{이름}-{작업}`: 작업 단위 브랜치
  - 예: `feature/dongs-eda`, `feature/teammate-baseline`
- 작업 후 PR(Pull Request) → 1명 이상 리뷰 → main 머지

### 커밋 메시지 컨벤션

```
[type] 짧은 요약 (한국어 OK)

본문 (선택)
```

- `type`: `feat` (새 기능), `fix` (버그), `docs` (문서), `refactor`, `data` (데이터/수집), `eda`, `model`, `chore`

예: `[data] FRED 결측치 처리 로직 추가`

### 절대 커밋하지 말 것

- `.env` (API 키)
- `data/` 하위 데이터 파일 (용량)
- `models/` 학습된 가중치
- 노트북 출력 셀이 무거운 경우 → `Cell > All Output > Clear` 후 커밋

### AI 사용 기록

본 프로젝트는 안내문 핵심 원칙에 따라 AI 도구 사용을 적극 활용한다. 단, **모든 AI 활용은 `AI_USAGE_LOG.md` 에, 검증·수정 이력은 `VALIDATION_LOG.md` 에 기록**한다.

---

## 7주 일정

| 주차 | 작업 | 산출물 |
|------|------|--------|
| 1 | 데이터 수집, EDA | `data/raw`, `data/interim`, `notebooks/01_eda.ipynb` |
| 2 | 전처리, Lag, Naive/ARIMA | `src/features/`, `notebooks/03_baselines.ipynb` |
| 3 | XGBoost 베이스라인 | `notebooks/03_baselines.ipynb` |
| 4 | LSTM 구현·학습 | `src/models/lstm.py`, `notebooks/04_lstm.ipynb` |
| 5 | Optuna 튜닝 (TCN 비교, 선택) | `configs/`, 최종 모델 |
| 6 | SHAP 분석, 위기구간 분석 | `notebooks/05_shap_analysis.ipynb`, `reports/figures/` |
| 7 | Streamlit 대시보드, 발표자료 | `app/streamlit_app.py`, `reports/final_report.md` |

---

## 변수 목록 (광역 22개 → EDA 후 8개로 축소)

| 분류 | 핵심 8개 (계획서 확정) | 출처 |
|------|----------------------|------|
| 타겟 | `kr_treasury_10y` 국고채 10년물 | ECOS |
| 국내 정책 | `kr_base_rate` 한국은행 기준금리 | ECOS |
| 국내 인플레 | `kr_cpi` CPI | ECOS |
| 국내 단기 | `kr_treasury_3y` 국고채 3년물 | ECOS |
| 미국 장기 | `us_treasury_10y` 미국 10년물 | FRED |
| 미국 정책 | `us_fed_funds` 연방기금금리 | FRED |
| 글로벌 위험 | `vix` VIX 지수 | FRED |
| 국내 위험자산 | `kospi` KOSPI | yfinance |

> 환율(`krw_usd`)은 정부 개입 구조 때문에 의도적으로 제외 (계획서 §3.3 참고). 검증용으로만 수집.

---

## 라이선스

학생 프로젝트 용도. 외부 데이터(ECOS / FRED / yfinance)는 각 출처의 이용 약관을 따른다.

# macro-bond-forecast

> **다변량 시계열 LSTM 분위수 회귀로 한국 국고채 10년물 일별 변화량(Δy) 예측**
> 2026학년도 머신러닝 팀 프로젝트 · 7주 · 2~3인

거시경제 지표 8개를 입력으로 한국 국고채 10년물 **일별 변화량 `Δy_t = (y_t − y_{t-1}) × 100` (bp)** 을 분위수 회귀(q=[0.05, 0.5, 0.95])로 예측한다. 점추정 외에 90% 예측 구간을 함께 제공하고, SHAP 으로 거시 변수의 시차 효과를 정량화한다.

자세한 설계는 `docs/project_plan.md` (계획서 v5.1) 참고.

---

## 핵심 결과 (test 구간 mean ± std)

| 지표 | A0 LSTM (Δfeature[t-1]) | Naive (Δŷ=0) | 평가 |
|------|--------------------------|---------------|------|
| **RMSE q50 (bp)** | **4.195 ± 0.030** | 4.647 | **-9.7%** ✅ |
| **Coverage 90%** | **0.902 ± 0.014** | — | 목표 정확 ✅ |
| **Dir_Acc q50** | **0.638 ± 0.011** | — | 목표 0.55 +15.4%p ✅ |

### DM test (HAC + HLN + Bonferroni α*=0.0167) — **3/3 모두 유의**

| 비교 | DM_HLN | p-value | Bonf 유의 |
|------|--------|---------|-----------|
| A0 vs Naive | -6.805 | <0.0001 | ✅ |
| A0 vs XGBoost | -6.963 | <0.0001 | ✅ |
| A0 vs LSTM raw | -6.714 | <0.0001 | ✅ |

### 차별화 포인트

1. **거시경제 도메인** — 학생 프로젝트에서 드문 채권/금리 (자산운용·증권·중앙은행 어필)
2. **부진 → 진단 → 회복 서사** (W4 raw 4.535 → A0 4.195) — 안내문 §9 오류 분석 모범 사례
3. **DM test 3/3 p<0.0001** — 학부 7주 압도적 통계 입증
4. **메타-검증 5회 누적** (#30 → #36 → #37 → #40 → #43) — audit 도구 자체의 자가 검증
5. **VALIDATION_LOG 43건** = 안내문 최소 3건의 **14.3배**

---

## 폴더 구조

```
macro-bond-forecast/
├── data/                              # ⚠️  git 제외 (용량/민감)
│   ├── raw/                           # API 원본
│   ├── interim/                       # wide 통합
│   └── processed/                     # 모델 학습용 + W4-W6 예측 (DM test 입력)
├── notebooks/                         # 번호 prefix 로 실행 순서 명시
│   ├── 01_eda.ipynb                   # W1 — 광역 22개 EDA + freeze 9 결정
│   ├── 02_feature_selection.ipynb     # W2(a) — 상관·VIF·Granger 산출물 3건
│   ├── 02b_preprocess_baseline.ipynb  # W2(b) — 전처리·Lag/Rolling·Naive·ARIMA·누수 리뷰
│   ├── 03_freeze_xgboost.ipynb        # W3 — 변수 freeze 8개 + XGBoost 분위수 회귀
│   ├── 04_lstm_quantile.ipynb         # W4 — LSTM 분위수 회귀 (raw, reference baseline)
│   ├── 05_tuning_ablation.ipynb       # W5 — A0(Δfeat[t-1]) + grid 5×5 + A1' ablation
│   ├── 06_shap_error_analysis.ipynb   # W6 — 분위수별 SHAP + DM test + 오류 분석 4축
│   └── 07_final_demo.ipynb            # W7 — 발표용 종합 데모
├── scripts/                           # CLI 진입점 + audit
│   ├── 01_verify_ecos_codes.py        # ECOS 항목코드 검증
│   ├── 02_collect_data.py             # 데이터 수집 (ECOS + FRED + yfinance)
│   ├── 03_eda_check.py                # W1 EDA 자동 검증
│   ├── 04_leakage_audit.py            # CL-01~07 + CL-05b/c 누수 감사
│   ├── 05_lstm_diff_ablation.py       # A0 사전 검증 (LOG #35)
│   ├── 06_w5_meta_verify.py           # W5 메타-검증
│   ├── 07_w6_meta_verify.py           # W6 메타-검증
│   └── 08_full_audit_w1w6.py          # W1-W6 전체 정합성 점검
├── reports/                           # 분석 산출물 + figure
│   ├── baseline_results_w{2,3,4,5}.csv
│   ├── lstm_a0_*_w5.csv               # multi-seed, grid, final eval
│   ├── ablation_a1_w5.csv             # 환율 ablation
│   ├── lstm_a0_shap_w6.npz            # 분위수별 SHAP 텐서
│   ├── dm_test_w6.csv                 # DM test 3개 비교
│   ├── error_analysis_w6.csv          # 4축 (a)~(d)
│   ├── crisis_labels_w6.csv           # 위기구간 정량 라벨
│   ├── channel_validation_w6.csv      # 거시경제 채널 부합 (V6 영역 분리)
│   ├── leakage_audit_w2.csv           # audit 결과 (7 ✅ + 2 ❌ CL-05b/c)
│   ├── summary_w6.md                  # 통합 요약
│   └── figures/                       # 25+ figures
├── app/
│   └── streamlit_app.py               # 7주차 §8 B1 1페이지 미니 데모 (필수)
├── docs/                              # 계획서 + 변수 검증 + freeze + ablation 문서
├── models/                            # ⚠️ git 제외 (.pt, .pkl, .json)
├── configs/config.yaml
├── prompts/                           # AI 프롬프트 (선택)
├── tests/
├── AI_USAGE_LOG.md                    # ⚠️ 안내문 §7 필수
├── VALIDATION_LOG.md                  # ⚠️ 안내문 §7 필수 (현재 43건)
├── requirements.txt
├── .env.example                       # API 키 템플릿 (.env 는 .gitignore)
└── .gitignore
```

---

## 빠른 시작

```bash
git clone https://github.com/donguk77/macro-bond-forecast.git
cd macro-bond-forecast

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

cp .env.example .env             # API 키 입력 (ECOS, FRED)
python scripts/02_collect_data.py
```

### Streamlit 데모 실행

```bash
streamlit run app/streamlit_app.py
```

### 노트북 실행 순서

```bash
jupyter lab
# 01 → 02 → 02b → 03 → 04 → 05 → 06 → 07 순서대로
```

---

## 7주 일정 (실제 수행)

| 주차 | 작업 | 핵심 결과 |
|------|------|-----------|
| **1** | 광역 22개 수집, EDA, XGBoost+TreeExplainer Hello World | freeze 9 변수 결정, σ(Δy)=4.0 bp 확인 |
| **2** | 상관·VIF·Granger 분석, 전처리, Naive/ARIMA, 누수 audit | 7/7 ✅ (false positive 4건 fix, LOG #30) |
| **3** | 변수 freeze 8개 (kospi 제외), XGBoost 분위수 회귀 | test RMSE 4.644, monotonicity 0건 crossing |
| **4** | LSTM 분위수 회귀 (raw) + LSTM-SHAP DeepExplainer | RMSE 4.535 (정체), 부진 진단 → LOG #35 |
| **5** | A0 (Δfeat[t-1]) 회복 + grid 5×5 + A1' ablation + CQR | **RMSE 4.195 (-7.5%), Coverage 0.902, Dir 0.638** |
| **6** | 분위수별 SHAP + DM test + 오류 분석 4축 + 위기구간 | **DM 3/3 p<0.0001**, 채널 strong 1/1 (kr_base_rate) |
| **7** | Streamlit 1페이지 + Jupyter 종합 데모 + 발표 자료 | `app/streamlit_app.py`, `notebooks/07_final_demo.ipynb` |

---

## 변수 freeze 8개 (W3 §3.2 freeze_final_w3.md)

| 분류 | 변수 | 출처 | 거시 채널 |
|------|------|------|-----------|
| 타겟 | `kr_treasury_10y` | ECOS | (예측 대상) |
| 국내 단기 | `kr_treasury_3y` | ECOS | 장단기 스프레드 |
| 국내 정책 | `kr_base_rate` | ECOS | 통화정책 전이 |
| 미국 장기 | `us_treasury_10y` | FRED | **한미 동조성** (top-1 SHAP) |
| 미국 정책 | `us_fed_funds` | FRED | 미국 통화정책 |
| 미국 인플레 | `us_breakeven_10y` | FRED | 기대인플레 (BEI) |
| 글로벌 위험 | `vix` | FRED | 안전자산 선호 |
| 위험 자산 | `sp500` | yfinance | 위험자산 흐름 |
| 글로벌 달러 | `dxy` | FRED | EM 자본유출 |

> 환율(`krw_usd`)은 정부 개입 구조로 의도적 제외 (§3.4). A1' 환율 ablation 결과 Δ≈0 으로 **정량 입증** (LOG #39).

---

## 메타-검증 5회 누적 (안내문 §7 직격)

| 라운드 | 발견 | 사용자 직감 |
|--------|------|-------------|
| **#30** (W2) | audit false positive 4건 | 자체 발견 |
| **#36** (W4 코드) | C1~C4 (CL-05b 정책변수 lag) | "한번더 검증을 해줘" |
| **#37** (#36 메타) | V1~V7 (CL-05c 미국 마감 5변수) | "한번더 검증이 필요할꺼 같은데?" |
| **#40** (W5) | V2 val-overfit + V9 HP 일관성 | "5주차도 검증을 먼저 해야 할 거 같아" |
| **#43** (W6) | V3 lookahead + V6 noise 영역 + V7 CL-03 fp | "검증을 먼저 해줘" |

**현재 audit 상태**: ✅ 7건 (CL-01~04, CL-05, CL-06, CL-07) + ❌ 2건 (CL-05b/c, W4 raw 잔존 결함, A0 자동 해결).

---

## 협업 규칙

- 브랜치: `main` 직접 push 가능 (소규모 팀, 솔로 작업)
- 커밋: `[type] 짧은 요약` — type: `feat`/`fix`/`docs`/`data`/`eda`/`model`
- **절대 커밋 금지**: `.env`, `data/`, `models/*.pt|*.pkl`, `.venv/`
- AI 활용은 `AI_USAGE_LOG.md` 에, 검증 이력은 `VALIDATION_LOG.md` 에 기록

---

## 라이선스

학생 프로젝트 용도. 외부 데이터(ECOS / FRED / yfinance)는 각 출처의 이용 약관을 따른다.

**GitHub**: https://github.com/donguk77/macro-bond-forecast

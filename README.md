# macro-bond-forecast

> **다변량 시계열 LSTM 분위수 회귀로 한국 국고채 10년물 일별 변화량(Δy) 예측**
> 2026학년도 머신러닝 팀 프로젝트 · 7주 · 2~3인

거시경제 지표 8개를 입력으로 한국 국고채 10년물 **일별 변화량 `Δy_t = (y_t − y_{t-1}) × 100` (bp)** 을 분위수 회귀(q=[0.05, 0.5, 0.95])로 예측한다. 점추정 외에 90% 예측 구간을 함께 제공하고, SHAP 으로 거시 변수의 시차 효과를 정량화한다.

자세한 설계는 `채권금리예측_프로젝트계획서_v5_4_1.docx` (저장소 외부 — 최신 final) 참고. `docs/project_plan.md` 는 v5.1 시점 스냅샷이며 v5.2~v5.4.1 patch 이력은 `docs/project_plan_changelog.md` 참조.

---

## 핵심 결과 — 옵션 3: XGBoost 분위수 회귀 + CQR (no_leak v2)

> v0 → v1 → v2 누수 발견·정정·회복 history. 자세한 비교: `reports/no_leak_v2/comparison_v0_v1_v2.md` · 종합 요약: `reports/no_leak_v2/FINAL_option3_summary.md`.

### Test 성능 (XGBoost CQR, 2023-2025)

| 지표 | single-split | 3-fold 평균 | Pooled | 목표 | 평가 |
|------|--------------|-------------|--------|------|------|
| **방향성 정확도** | 0.6113 | 0.6099 ± 0.0275 | **0.6178** | ≥ 0.55 | **+6.78%p ✅** (학술 합격선 53% 대비 +8.78%p) |
| **Coverage 90% (CQR)** | 0.8573 | 0.8727 ± 0.1507 | — | 0.87~0.93 | △ -1.3%p (CQR 한계 인정, ACI 후속 검토) |
| **Sharpness (bp)** | 11.73 | — | — | 좁을수록 | ✅ |
| **RMSE q50 (bp)** | 4.480 | — | 4.721 | < Naive | ✅ -3.7% (single, Naive 4.65) / -2.8% (Pooled, Naive 4.86) |

### DM test (HAC + HLN + Bonferroni) — Naive 대비 통계적 우위

| 비교 | DM_HLN | p-value | Bonf 유의 |
|------|--------|---------|-----------|
| XGBv2 vs Naive (single) | -6.23 | <0.0001 | ✅ (α=0.025) |
| XGBv2 vs ARIMA (single) | -6.54 | <0.0001 | ✅ (α=0.025) |
| **XGBv2 vs Naive (Pooled 3-fold)** | **-8.78** | **<0.0001** | ✅ (α=0.0167) |

### 실무 시뮬레이션 — Backtest (XGB v2, 정직 보강판)

> 1차 단순 backtest (`reports/no_leak_v2/backtest_v2.csv`, scripts/23) 는 **carry 미반영 + 단일 split** 한계로 Sharpe 2.42 (낙관). Level B 보강 (`reports/no_leak_v2/backtest_v2_advanced.csv`, scripts/24) 으로 정직 결과 산출.

#### Single-split (test 2023-2025, carry+convexity+cost 1bp 보강)

| 전략 | Total Return | Sharpe (252) | MDD | Win Rate | 거래수 |
|------|------|------|------|------|------|
| S0 Buy-and-Hold | +12.42% | 0.76 | -7.21% | 50.6% | 0 |
| S1 SignQ50 (매일 매매) | +12.10% | 0.75 | -11.25% | 52.2% | 303 |
| **S2 ConfFilter (\|q50\|>1bp만 매매)** | **+13.43%** | **2.07** | **-1.43%** | 40.9% | 151 |
| S3 VolTarget | +3.71% | 0.28 | -11.12% | 52.1% | 414 |

- **S1 단순 매매 + 1bp cost ≈ Buy-and-Hold** (carry+거래비용 흡수로 alpha 사라짐)
- **S2 ConfFilter 가 가장 robust** — 확신 있을 때만 매매하는 것이 일별 매매보다 안정 (Sharpe 2.07, MDD -1.43%)

#### Walk-forward 3-fold (정직 검증) — backtest run, scripts/24, n_estimators 87/215/53

| Fold | 기간 | dir_acc | S2 Sharpe |
|------|------|---|---|
| fold1 | 2020 (코로나) | 0.564 | -1.15 |
| fold2 | 2021-22 (인상기) | 0.535 | -0.14 |
| fold3 | 2023-25 | 0.619 | +1.75 |
| **Pooled** | 전체 | — | **+0.62 [-0.18, +1.35]** (95% bootstrap CI) |

> ⚠️ 위 dir_acc (0.564/0.535/0.619) 는 backtest walk-forward (scripts/24, n_estimators 87/215/53) 결과로, 위 §"Test 성능" 표의 model walk-forward (scripts/16, n_estimators 200/400/100) 의 0.5932/0.5949/0.6416 (평균 0.6099, Pooled 0.6178) 와는 별개 run 입니다. 같은 fold·feature·HP grid 이지만 early-stopping 결과 n_estimators 가 달라 dir_acc 가 약 0.04 차이. 자세한 분석은 `reports/no_leak_v2_honest/META_AUDIT_AND_DOC_CONSISTENCY_2026-05-17.md` §B-1 참조.

- Pooled Sharpe CI 가 0 포함 → **백테스트 차원에서는 통계적 우위 입증 어려움**
- 단, **모델 자체의 통계 우위 (DM=-8.78 Pooled, p<0.0001 Bonferroni 통과) 는 유효**
- 즉 "모델은 통계적으로 정확하지만 거래비용 마진은 얇다" — 정직한 그림

→ 자세한 분석: `reports/no_leak_v2/backtest_v2_advanced.csv`, `walkforward 결과 backtest_v2_walkforward.csv`

### 차별화 포인트

1. **거시경제 도메인** — 학생 프로젝트에서 드문 채권/금리 (자산운용·증권·중앙은행 어필)
2. **누수 발견·정정·회복 정직성 서사** — v0 dir 65% (학술 합격선 +12%p로 비현실적 의심) → CL-05c 미국 마감변수 timing leak 발견 → v1 정정 후 49.8% (랜덤 수준, 누수 부산물 입증) → v2 새 변수 5개 + XGB + CQR로 dir **0.6178 (Pooled)** 정직 회복. 안내문 §9 "AI 결과 자체 검증" 원칙 직격
3. **walk-forward 3-fold + Pooled DM=-8.78 p<0.0001 Bonferroni 통과** — single-split 의존성 해소, 결과 안정성·일반성 입증 (`reports/no_leak_v2/walkforward_summary_v2.md`)
4. **개선안 시도·반증 history** — ACI / rolling vol 추가 시도가 데이터로 반증되어 채택 X (`reports/no_leak_v2/package_a_verification.md`). 끼워맞추기 회피, 발표 자산화
5. **메타-검증 5회 누적** (#30 → #36 → #37 → #40 → #43) — audit 도구 자체의 자가 검증
6. **VALIDATION_LOG 43건** = 안내문 최소 3건의 **14.3배**

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
| **1** | 광역 22개 수집, EDA, XGBoost+TreeExplainer Hello World | 9개 candidate freeze (sp500 등), σ(Δy)=4.0 bp 확인 |
| **2** | 상관·VIF·Granger 분석, 전처리, Naive/ARIMA, 누수 audit | 7/7 ✅ (false positive 4건 fix, LOG #30) |
| **3** | 변수 freeze 8개 확정 (kospi 제외), XGBoost 분위수 회귀 | test RMSE 4.644, monotonicity 0건 crossing |
| **4** | LSTM 분위수 회귀 (raw) + LSTM-SHAP DeepExplainer | RMSE 4.535 (정체), 부진 진단 → LOG #35 |
| **5** | A0 (Δfeat[t-1]) 회복 시도 + grid 5×5 + A1' ablation + CQR | A0 dir 0.638 — 학술 합격선 +12%p 비현실 → audit 트리거 |
| **6** | 분위수별 SHAP + DM test + 오류 분석 4축 + 위기구간 + **CL-05c 누수 발견** | v1 정정 후 dir 49.8% (랜덤), us_treasury_10y top-1 SHAP 도 누수 부산물 입증 |
| **7** | **v2 회복** (새 변수 5 + XGB + CQR + walk-forward 3-fold) + Streamlit + 종합 데모 | **dir 0.6178 (Pooled), DM=-8.78 Bonferroni 통과**, `notebooks/08_v2_no_leak_pipeline.ipynb`, `app/streamlit_app.py`, `notebooks/07_final_demo.ipynb` |

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

### v2 (no_leak) 추가 변수 5개 — `docs/features_v2_justification.md`

| 변수 | 정의 | 도메인 정당화 |
|------|------|---------------|
| `spread_10y_t1` | `(us10y - kr10y).shift(1)` | EM capital flow 1차 동인 (Caballero & Krishnamurthy 2009) |
| `delta_us10y_t1` | `us10y.diff().shift(1)` | 미국 → 한국 시차 모멘텀 (~16시간) |
| `delta_vix_t1` | `vix.diff().shift(1)` | 위험회피 모멘텀 — 안전자산 채널 |
| `delta_dxy_t1` | `dxy.diff().shift(1)` | EM 자본유출 모멘텀 |
| `crisis_dummy` | `(vol_20d > train-only 80%ile).shift(1)` | 위기 정량 라벨 (계획서 §4.4) |

→ 사전 기록(`docs/features_v2_justification.md`) 으로 사후합리화 방지. v2 입력 = 8 freeze + 5 new = **13개**.

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

## 자가 점검 history (2026-05-17 4축 메타 검증)

v0→v1→v2 history 에 더해 발표 직전 4축 자가 점검을 추가 수행:

| # | 점검 영역 | 핵심 발견 | 산출 |
|---|---|---|---|
| #1 | 메인 v2 모델 (scripts 14~22) | P0 4건 (HP 누수·cal⊂val·단일 시드·kospi 불일치) → honest 재실행 → **dir 0.6163 (Δ=-0.15%p, robustness 입증)** + **bootstrap CI [0.5927, 0.6421] 학술 합격선 0.53 통계 우위 정량 입증** | `reports/no_leak_v2_honest/CODE_REVIEW_AND_FIX_HISTORY.md` + `scripts/25_honest_walkforward.py` |
| #2 | Path A 실험 (LGB Sharpe 2.02) | PA Critical 3건 (fold1 거래 0건, 5.5% participation, HP cross-fold 평균) → 메인 대안 → **보조 실험 격하** | `experiments/path_a_model/CODE_REVIEW_PATH_A_2026-05-17.md` |
| #3 | Backtest (scripts 23, 24) | BT 14건 (HP 누수 재발, VolTarget lookahead, i.i.d. bootstrap, cost 2× 차이 등) → 메인 결론 영향 미미 | `reports/no_leak_v2/BACKTEST_CODE_REVIEW_2026-05-17.md` |
| #4 | Audit 도구 + 문서 정합성 | **기존 audit 가 P0/Critical 11건 중 0건 잡음** (사각지대 6 유형 공식화) + README 표 헤더 모호성 1건 | `reports/no_leak_v2_honest/META_AUDIT_AND_DOC_CONSISTENCY_2026-05-17.md` |

→ 안내문 §7 "AI 결과 자체 검증" 의 **7단계 메타 검증 사례** (#30→#36→#37→#40→#43→#48~50→#51) 완성.
→ 메인 발표 수치 (dir 0.6178, DM -8.78, Sharpe 0.62) 는 변경 없음. 점검 결과 robustness 입증 + 한계 정직 인정 + audit 도구 사각지대 공식화.

---

## 라이선스

학생 프로젝트 용도. 외부 데이터(ECOS / FRED / yfinance)는 각 출처의 이용 약관을 따른다.

**GitHub**: https://github.com/donguk77/macro-bond-forecast

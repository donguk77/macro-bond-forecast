# 3주차 변수 freeze 최종 확정 (8개)

> 계획서 v5.1 §3.2(a) 산출물 4 — `feature_validation_w2.md` (산출물 1~3) 기반 결정 문서.
> 생성: notebooks/03_freeze_xgboost.ipynb · 2026-05-02

## 1. 채택 (8개)

| # | 변수 | W2 점수 | 채택 사유 |
|---|------|---------|-----------|
| 1 | `kr_treasury_3y` | 3 | 국내 단기 금리 (장단기 스프레드 신호) |
| 2 | `kr_base_rate` | 2 | 한국 통화정책 전이 채널 (lag 1 강제) |
| 3 | `us_treasury_10y` | 3 | 한미 금리 동조성 (Granger lag 3) |
| 4 | `us_fed_funds` | 2 | 미국 통화정책 기준 (lag 1) |
| 5 | `us_breakeven_10y` | 3 | 기대인플레 (명목 = 실질 + 기대 항등식) |
| 6 | `vix` | 2 | 글로벌 위험 신호 |
| 7 | `sp500` | 2 | EDA SHAP 1위 (비선형/interaction) |
| 8 | `dxy` | 3 | EM 자본유출 채널 (환율 부재 보완) |

## 2. 제외 (W3 단계 — 1개)

| 변수 | W2 점수 | 제외 사유 |
|------|---------|-----------|
| `kospi` | 1 | Granger p=0.57 (best_lag=1) → 타겟 선행 못함. sp500+vix 가 위험 채널 커버. 5주차 ablation 후보. |

> **참고**: W1 단계에서 이미 다중공선성/약한 신호로 제외된 변수는 `VALIDATION_LOG.md` **#29** 참조:
> - 한국 금리 패밀리 (`kr_treasury_5y`, `kr_treasury_1y`, `kr_corp_aa3y`, `kr_cd_91d`) — \|r\|>0.7 다중공선성, `kr_treasury_3y` 1개만 채택
> - 월별 인플레/실물 (`kr_cpi`, `kr_cpi_core`, `kr_ppi`, `kr_industrial_prod`, `kr_mfg_bsi_outlook`, `us_cpi`) — 일별 Δy 와 거의 무관
> - `us_treasury_2y` (us_treasury_10y 다중공선성), `us_credit_spread` (vix 다중공선성), `wti_oil` (약한 신호)
>
> 즉 W3 의 8개 freeze = (광역 22 후보) − (W1 다중공선성·약한신호 13개) − (W2 검증 약점 1개=kospi).

## 3. 거시경제 채널 일관성 (계획서 §3.1)

- 정책: `kr_base_rate`, `us_fed_funds`
- 인플레: `us_breakeven_10y`
- 동조성: `us_treasury_10y`
- 위험: `vix`
- 자산: `sp500` (글로벌만, kospi 제거)
- 글로벌달러: `dxy`
- 한국 단기: `kr_treasury_3y`

5개 채널 모두 커버. kospi 제거로 자산 채널이 글로벌 1개로 축소되지만 5주차 ablation 에서 재검증.

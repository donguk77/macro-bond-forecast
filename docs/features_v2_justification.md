# v2 새 변수 도메인 정당화

> features_v2_no_leak.csv 에 추가된 변수 5개의 도메인·학술 근거.
> 작성: scripts/14_features_v2.py 자동 생성 (사후합리화 방지 — 결과 보기 전 사전 기록).

| 변수 | 정의 | 도메인 정당화 |
|---|---|---|
| `spread_10y_t1` | (us10y - kr10y).shift(1) | 한미 금리차는 외국인 채권 자금 흐름의 1차 동인. Caballero & Krishnamurthy (2009) 등 EM capital flow 학술 표준. 환율 부재로 흡수 못한 EM 충격 채널. |
| `delta_us10y_t1` | us10y.diff().shift(1) | 어제 미국 10년 변화량 — t-1 → t 시차 모멘텀 신호. 미국 종가 형성(KST 새벽) → 한국 시초가 갭 → 한국 종가까지 영향. |
| `delta_vix_t1` | vix.diff().shift(1) | 어제 VIX 변화량 — 위험회피 모멘텀. VIX 급등은 안전자산 선호로 한국 채권 매수 → 금리 ↓ 시차 효과. |
| `delta_dxy_t1` | dxy.diff().shift(1) | 어제 달러 인덱스 변화량 — EM 자본 유출입 모멘텀. DXY ↑ → EM 자본 유출 → 한국 채권 매도 → 금리 ↑. |
| `crisis_dummy` | (vol_20d > train-only 80%ile).shift(1) | 계획서 §4.4 위기 정량 정의. Train-only quantile로 누수 차단. 위기 구간 Coverage 회복용. |

## 누수 차단 점검 (사전)
- 모든 변수 `.shift(1)` 적용 → CL-05/05c 준수
- 위기 더미 임계값은 train 구간 통계로만 결정 → CL-08 준수
- rolling vol 자체도 `.shift(1)` → CL-03 준수

## 통계
- 위기 더미 train-only 임계값 = 4.108 bp
- 입력 변수 (raw + 새변수) = 14개
- 최종 feature 수 (lag/roll 포함, 라벨 제외) = 162
- 데이터 기간 = 2010-02-25 ~ 2025-12-30

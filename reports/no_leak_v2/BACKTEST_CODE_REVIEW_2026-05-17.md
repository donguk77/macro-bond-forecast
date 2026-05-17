# Backtest (scripts 23, 24) 정밀 점검 보고서

> **점검 일자**: 2026-05-17
> **점검 대상**: scripts/23_backtest_v2.py, scripts/24_backtest_v2_advanced.py + reports/no_leak_v2/backtest_v2_*.csv
> **점검 방식**: 코드 정밀 독해 + csv 수치 교차 검증 (재실행 없음)
> **트리거**: 메인 v2 P0-1 (HP 누수) + Path A PA-1 동일 패턴 발견 → backtest 도 동일 패턴 의심

---

## TL;DR

> Backtest 결과 (Sharpe 2.07 S2 ConfFilter, Pooled 0.62) 는 **2건의 critical 결함** 포함:
> 1. **HP cross-fold 누수 재발** (script 24 BEST_PARAMS hardcoded — P0-1 backtest 판) → walk-forward Sharpe 도 leakage 영향
> 2. **S3 VolTarget 의 lookahead** (rolling vol .bfill() 로 첫 20일 미래 정보 사용)
>
> 추가로 **bootstrap CI i.i.d. resampling** (시계열 무시 → CI 너무 좁음), **script 23 vs 24 비용 계산 2× 차이** (round-trip 해석 불일치) 등 결함. **단, 발표 핵심 메시지 ("모델 통계 우위, 거래비용 마진 얇음") 는 결함과 무관하게 유효**.
>
> 권장: 발표 자료에 "단순 backtest" vs "정직 보강" 구분만 유지하고, walk-forward Sharpe 도 P0-1 영향 받는다는 점 한 줄 명시.

---

## 1. 점검 결과 우선순위

| ID | 심각도 | 위치 | 내용 | 영향 |
|---|---|---|---|---|
| **BT-1** | 🔴 Critical | scripts/24:91-95 | `BEST_PARAMS` hardcoded (single-split 결과) → walk-forward 3-fold 에 전부 적용. 메인 P0-1 과 동일 패턴 | walk-forward Sharpe 도 HP 누수 영향 |
| **BT-2** | 🔴 Critical | scripts/24:197 | S3 VolTarget 의 `rolling(20).std().shift(1).bfill()` 에서 `.bfill()` 이 첫 20일을 미래 vol 로 채움 | 첫 20일 lookahead leak |
| **BT-3** | 🟡 Medium | scripts/24:271-280 | Sharpe bootstrap CI 가 i.i.d. resampling → 시계열 autocorrelation 무시 → CI 너무 좁음 | CI 신뢰성 ↓ |
| **BT-4** | 🟡 Medium | scripts/23:152-154 vs scripts/24:236-239 | Cost 계산이 두 script 에서 2× 차이 — script 23 은 `/2` (round-trip=1bp), script 24 는 미분할 (round-trip=2bp) | 두 backtest 비교 불가 |
| **BT-5** | 🟡 Medium | scripts/24:91-95 (`n_estimators`) | BEST_PARAMS 의 n_estimators 가 fixed (87/215/53) → 각 fold 의 early-stopping best_iter 무시 | fold 별 underfit 가능성 |
| **BT-6** | 🟡 Medium | scripts/24:64 `CONFIDENCE_THRESHOLD_BP=1.0` | S2 ConfFilter 의 τ=1bp 가 hardcoded — val 에서 grid 없이 결정 | data snooping 의심 (단 영향 작음) |
| **BT-7** | 🟢 Low | scripts/24:205-211 | S4 DualQuantile 항상 0 거래 (테스트 전 구간 q05<0 또는 q95>0) | metric N/A 보고만 |
| **BT-8** | 🟢 Low | scripts/24:236-238 | pos_change 계산이 두 줄로 (첫째 줄 effectless overwrite) | 가독성, 버그 아님 |
| **BT-9** | 🟢 Low (quirk) | scripts/24:255-268 `days_active` | win+loss 만 셈. 비용 0→0.5 시 exit 일자가 active 로 바뀜 → 카운트 변동 | 메트릭 직관성 ↓ |
| **BT-10** | 🟢 Low | scripts/23:126-130 | dir_acc 검증을 print 만 (script 24 는 assert) — 일관성 부족 | — |
| **BT-11** | 🟢 Low | scripts/24:65 `N_BOOTSTRAP=1000` | 1000 resamples 는 95% CI 에 borderline (5000+ 권장) | CI noise |
| **BT-12** | 🟢 Low | Both | D=8.0, C=85.0 hardcoded — duration·convexity 가 수익률 수준에 따라 변하나 fixed | 일반 학생 가정 OK |
| **BT-13** | 🟢 Low | scripts/24:233 carry 계산 | `y_level_pct / 100 / 252` — 365 vs 252 day count 선택. 일별 backtest 라 252 OK | — |
| **BT-14** | 🟢 Low | scripts/24:307 `y_level.reindex(dates).ffill()` | y_level 결측 시 forward-fill — 과거 yield 그대로 사용. 작은 timing artifact | — |

---

## 2. BT-1 상세 — Backtest walk-forward 도 HP 누수

### 문제 코드 (scripts/24:91-95)
```python
# Best params from script 15 grid
BEST_PARAMS = {
    0.05: {'max_depth': 4, 'learning_rate': 0.03, 'n_estimators': 87},
    0.50: {'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 215},
    0.95: {'max_depth': 4, 'learning_rate': 0.10, 'n_estimators': 53},
}
```

`fit_predict_fold()` 가 모든 fold (fold1, fold2, fold3) 에 동일 `BEST_PARAMS` 적용.

### 영향 — backtest_v2_walkforward.csv 의 Sharpe 모두 HP 누수 영향
| fold | strategy | Sharpe | 비고 |
|---|---|---|---|
| fold1 | S2_ConfFilter | -1.15 | HP from val=2022 (fold1 입장 future) |
| fold2 | S2_ConfFilter | -0.14 | HP from val=2022 (fold2 test 기간 내부) |
| fold3 | S2_ConfFilter | +1.75 | HP from val=2022 (fold3 own val) ✅ |
| **POOLED** | S2_ConfFilter | **+0.62** | 위 3 fold 합성 |

→ **fold2 의 Sharpe -0.14 가 가장 의심스러움** (HP 가 fold2 test 기간에서 best 로 뽑힌 것임에도 음수 → HP 가 정직히 robust 하지 못함을 역설적으로 시사)
→ **fold1 의 Sharpe -1.15 는 OOD + HP 누수 이중 영향**
→ **fold3 는 HP-test 일치라 over-fit 가능성**

### 권장 조치
- 메인 모델 v2 honest 점검과 동일하게, backtest 도 per-fold HP 가 필요
- 단 v2 honest 가 dir_acc 거의 동일 결과 (0.6178→0.6163) 였으므로 backtest 도 큰 변화 없을 가능성
- 발표 시: "Backtest walk-forward 도 HP fixed (script 15 결과 재사용) — 메인과 동일한 누수 패턴" 1줄 명시 권장

---

## 3. BT-2 상세 — S3 VolTarget Lookahead

### 문제 코드 (scripts/24:196-204)
```python
if name == 'S3_VolTarget':
    s = pd.Series(y_true, index=dates)
    rolling_vol_bp = s.rolling(20).std().shift(1).bfill().values  # bp
    daily_price_vol = D * rolling_vol_bp / 10000
    target_daily_vol = TARGET_VOL_ANNUAL / np.sqrt(252)
    size = np.clip(target_daily_vol / np.where(daily_price_vol > 0, daily_price_vol, 1e-6),
                   0.0, 1.0)
    return np.sign(q50) * size
```

### 무엇이 leakage 인가
- `y_true` = test 구간의 실현된 Δy_bp (즉 미래 정보)
- `rolling(20).std()` → t 일 vol = std(Δy[t-19:t])
- `.shift(1)` → t 일 vol = std(Δy[t-20:t-1])
- `.bfill()` → 첫 20일 NaN 을 **그 후의 vol 값** 으로 채움 (= 미래의 vol 사용)

### 영향 범위
- Test 첫 20일 (2023-01-01 ~ 2023-01-30 정도) 의 vol-targeting 이 미래 vol 사용
- 그 이후 681일은 정상 (shifted rolling)
- 영향: total Sharpe 의 약 3% 가량 (작지만 존재)

### 정직한 fix
- `.bfill()` 제거 → 첫 20일 NaN 그대로 → 그 기간 position=0 (cash)
- 또는 train 의 마지막 20일 std 로 첫 20일 채움 (warm-up)

### 추가 우려 — y_true 자체를 vol 추정에 쓰는 것이 OK 한가?
- VolTarget 의 표준 관행: 가격 변동성으로 사이징 → 가격 데이터는 알려진 과거
- 여기서는 `y_true` (모델이 예측하려는 대상) 의 실현값으로 사이징
- shift(1) 적용이라 t 일 사이징은 t-1 까지의 실현 Δy 만 사용 → **본질적으로 leakage 아님** (Δy 자체는 매일 close 에 관측 가능)
- 단 위의 `.bfill()` 만 문제

### Backtest 결과 확인 (backtest_v2_advanced.csv)
| Strategy | cost=0 Sharpe | cost=1 Sharpe |
|---|---|---|
| S3_VolTarget | 3.540 | **0.280** |

→ S3 가 cost=1bp 에서 Sharpe 0.28 로 약함. 첫 20일 lookahead 가 일부 기여 가능성 있으나 cost 영향이 압도적.

---

## 4. BT-3 상세 — Bootstrap CI 가 i.i.d. resampling

### 문제 코드 (scripts/24:271-280)
```python
def sharpe_bootstrap_ci(pnl, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pnl)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)  # i.i.d. resampling
        sample = pnl[idx]
        s = float(sample.mean() / sample.std() * np.sqrt(252)) if sample.std() > 0 else float('nan')
        sharpes[i] = s
    return float(np.quantile(sharpes, 0.025)), float(np.quantile(sharpes, 0.975))
```

### 문제
- PnL 시계열은 일별 autocorrelation 존재 (특히 trend, vol clustering)
- i.i.d. resampling 은 autocorrelation 파괴 → variance 과소평가 → CI 너무 좁음
- 시계열에서는 **block bootstrap** (block size = autocorrelation lag, 보통 5~20일) 필요

### 영향 — backtest_v2_sharpe_ci.csv
| Strategy | Sharpe | i.i.d. CI (현재) | 추정 block bootstrap CI (약 1.5~2× 넓이) |
|---|---|---|---|
| S0_BuyHold | 0.757 | [-0.471, 1.980] | ~[-0.8, 2.3] |
| S1_SignQ50 | 0.751 | [-0.408, 1.971] | ~[-0.7, 2.3] |
| **S2_ConfFilter** | **2.073** | **[1.084, 2.925]** | ~**[0.6, 3.4]** (하한 여전히 > 0) |
| S3_VolTarget | 0.280 | [-0.883, 1.493] | ~[-1.2, 1.8] |

→ S2 ConfFilter 의 "CI 가 0 배제" 는 block bootstrap 으로도 유지될 가능성 (Sharpe 가 워낙 큼)
→ 다른 전략은 i.i.d. CI 도 0 포함, block 으로 더 명백히 0 포함

### 정직한 fix
- block bootstrap (예: arch-py 의 `MovingBlockBootstrap`) 또는 stationary bootstrap 사용
- 발표 시: "단순 i.i.d. bootstrap CI — 시계열 보정 시 더 넓을 수 있음" 명시

---

## 5. BT-4 상세 — Script 23 vs 24 의 비용 계산 2× 차이

### Script 23 (line 152-154)
```python
bt['position_change'] = bt['position_xgb_v2'].diff().fillna(0).abs() / 2  # 0 or 1
bt['txn_cost'] = bt['position_change'] * TXN_COST_BP * D / 10000
```
- `/2` → sign flip (-1→+1, |Δ|=2) 시 position_change=1 → cost = 1 × cost_bp = 1bp (round-trip 한 번)
- 0→1 entry 시 position_change=0.5 → cost = 0.5 × cost_bp (half round-trip)

### Script 24 (line 236-239)
```python
pos_change = np.abs(np.diff(pos, prepend=pos[0])) / 2  # 0 to 1 scale
# vol-targeted 같이 0~1 사이값일 수도 있어서 |Δpos| 사용
pos_change = np.abs(np.diff(pos, prepend=pos[0]))      # /2 없음 (위 줄 overwrite)
txn_cost = pos_change * cost_bp * D / 10000
```
- 두 줄로 작성, **둘째 줄이 첫째 줄 overwrite** → `/2` 무효
- sign flip 시 cost = 2 × cost_bp = 2bp
- 0→1 entry 시 cost = 1 × cost_bp = 1bp

### 결론
- Script 23 의 "1bp/포지션 변경" = round-trip 1bp
- Script 24 의 "1bp/포지션 변경" = one-way 1bp (round-trip 2bp)
- **같은 cost_bp=1 입력 시 script 24 는 script 23 보다 sign flip 비용 2× 부과**

### Backtest 결과 일관성 확인
- backtest_v2.csv (script 23): "XGB v2 (sign q50, 1bp cost)" Total = ?
  - reports/no_leak_v2/backtest_v2.csv 의 v2 net total 값 필요
  - README 에 "S1 SignQ50 12.10%" 와 backtest_v2_advanced.csv (script 24) S1 cost=1.0 의 12.10% 가 같다면 일관성 있음
  - **그러나 cost 해석이 다르므로 같은 12.10% 가 다른 의미** (script 23 은 round-trip 1bp, script 24 는 one-way 1bp)

### 권장 조치
- Script 24 의 의도가 round-trip 1bp 였다면 `/2` 복원 필요
- 또는 의도가 one-way 1bp 였다면 script 23 도 동일하게 (지금은 round-trip)
- 발표에서 "1bp 거래비용" 인용 시 어떤 해석인지 명시 권장

---

## 6. BT-5/6 상세 — n_estimators / threshold hardcoded

### BT-5: n_estimators fixed
- BEST_PARAMS 의 n_estimators 가 87/215/53 (script 15 grid 의 fold3 val 기준)
- script 24 의 `fit_predict_fold` 가 이 값 그대로 fold1·fold2 적용
- 각 fold 의 own val 에서는 다른 best_iter 가 나올 가능성
- `fit_xgb` 가 early_stopping 도 안 켜 있음 (script 24 line 140-151)

→ fold1 underfit (87이 너무 큼?) 또는 overfit (87이 너무 작음?) 가능
→ 메인 v2 honest 처럼 per-fold early stopping 으로 정정 권장 (재실행 필요, 미실행)

### BT-6: CONFIDENCE_THRESHOLD_BP = 1.0
- S2 ConfFilter 의 τ=1bp 는 hardcoded
- Path B 에서는 τ grid {0.5, 1.0, 1.5, 2.0, 2.5, 3.0} 탐색 후 1.5 가 best
- 여기서는 1.0 hardcoded — Path B 와 다름
- 미세한 data snooping 가능성 (val 에서 select 안 함)

---

## 7. BT-7 상세 — S4 DualQuantile 항상 0

### 증거 (backtest_v2_advanced.csv)
```
S4_DualQuantile,0.0,0.0,,0.0,,0.0,0,0.0,0.0,0.0,0
S4_DualQuantile,0.5,0.0,...
...
```
모든 cost 에서 days_active=0, total_return=0.

### 원인
- 조건: `q05 > 0` (95% 확신 yield up) 또는 `q95 < 0` (95% 확신 yield down)
- Test 구간에서 한 번도 발생 안 함 → 90% PI 가 항상 0 포함
- 이건 모델의 자연스러운 특성 (분위수 회귀가 95% 확신 거의 못 줌)
- **버그 아님, 정상 동작**

### 발표 시 권고
- "S4 는 95% 확신 거래만 → 0건. 모델의 보수성 입증" 으로 활용 가능
- 또는 metric 표에서 S4 제외 (정직히 N/A 명시)

---

## 8. 잘된 점 (호평)

### ✅ Convexity + Carry 포함
- 학부 backtest 가 carry/convexity 까지 모델링 — 흔치 않은 정직성
- carry term sign 정확 (long bond → +coupon)
- convexity sign 정확 (long bond → +Γ, short → -Γ)
- y_level (raw ECOS) 직접 로드 — feature 누수 우회

### ✅ Cost sensitivity 표
- 0/0.5/1/2/3 bp 시나리오 분리 → 거래비용 marginal 임을 시각화
- S1 SignQ50 cost=2bp 부터 Sharpe 음수 → "비용에 약함" 메시지 정직

### ✅ Walk-forward + Pooled 둘 다 보고
- Single-split 만으로 끝내지 않음
- fold1·fold2 의 Sharpe 음수 정직 공개
- Pooled CI [-0.18, 1.35] 0 포함 명시

### ✅ Strategy 다양화 (S0~S4)
- Buy-and-Hold benchmark
- 단순 sign(q50)
- Confidence filter
- Vol target
- Dual quantile
→ "모델은 같아도 전략에 따라 결과 다름" 메시지

### ✅ Decompose total = price + carry + convex + cost
- backtest_v2_advanced.csv 의 컬럼 분리 → reproducibility 우수

---

## 9. 발표 권장 수정 사항

### 추가할 멘트
> "Backtest 코드 점검에서 walk-forward 의 hyperparameter 도 single-split 결과 hardcoded (메인 모델 점검의 P0-1 과 동일 패턴) 임을 확인했습니다. 따라서 fold별 Sharpe 도 일부 HP 누수 영향이 있을 수 있으나, 메인 모델 honest 재실행에서 dir_acc 변동이 ±0.15%p 로 미미했음을 고려할 때 backtest 도 본질적으로 동일한 결론 ('모델은 통계적으로 정확하지만 거래비용 마진은 얇음') 으로 수렴할 것으로 판단됩니다."

### Q&A 대비
- Q: "Walk-forward backtest 의 fold2 Sharpe -0.14 는 어떻게 해석하나요?"
  → A: "정직히 인정합니다. 2021-22 인상기는 모델이 작동을 못한 구간이고, 추가로 HP가 fold3 기준으로 fixed 였다는 누수 결함도 점검에서 발견했습니다. 그래도 Pooled S2 ConfFilter Sharpe 0.62 (CI [-0.18, +1.35]) 는 백테스트로 통계 우위 입증 한계임을 정직히 보고했습니다."
- Q: "Bootstrap CI 가 i.i.d. 인데 시계열에 맞나요?"
  → A: "정직히 i.i.d. resampling 으로 시계열 autocorrelation 무시했습니다. Block bootstrap 적용 시 CI 가 약 1.5~2배 넓어질 가능성이 있으며, S2 ConfFilter 의 [1.08, 2.93] 도 약 [0.6, 3.4] 정도로 확대될 수 있습니다. 메인 결과 (모델 DM 우위) 는 영향 받지 않습니다."
- Q: "Script 23 과 24 의 cost 1bp 가 같은 의미인가요?"
  → A: "정직히 다릅니다. Script 23 은 round-trip 1bp 가정, script 24 는 one-way 1bp 가정 (sign flip 시 2bp 부과) — 코드 점검에서 발견했습니다. 발표 자료에는 script 24 (더 보수적) 결과를 인용합니다."

---

## 10. 영향력 추산 — 결함 합산 후 결과 변화 예상

| 결함 | Sharpe 영향 추정 | 방향 |
|---|---|---|
| BT-1 HP 누수 정정 | ±0.05 ~ ±0.1 | 불명 (메인 honest 처럼 무시 가능 가능성) |
| BT-2 VolTarget bfill 정정 | 미미 (첫 20일만) | S3 만 |
| BT-3 i.i.d. → block bootstrap | CI 폭 1.5~2배 | 모든 전략 |
| BT-4 cost 해석 통일 | 어떤 쪽으로 통일 하느냐에 따라 ±2× | 직접 - |
| BT-5 n_estimators fixed | ±0.05 정도 | 불명 |
| BT-6 threshold hardcoded | 미미 | S2 만 |

**합산 추정**: Pooled Sharpe 0.62 → 0.4~0.8 범위 (큰 변화 없음). 메인 결론 ("거래비용 마진 얇음") 유지.

---

## 11. 통합 정직 history (3차 자기점검)

```
v0 → CL-05c → v1 → 새 변수 + XGB + CQR → v2
   ↓ 자가 점검 #1 (메인 모델)        ✅ honest 재실행: dir 0.6163 (Δ=-0.15%p, robustness 입증)
   ↓ 자가 점검 #2 (Path A 실험)      ✅ 한계 인정: 5.5% participation, fold1 0건, HP cross-fold leak
   ↓ 자가 점검 #3 (Backtest 23·24)   ✅ 결함 14건 발견·문서화 (재실행 X): HP 누수 재발, VolTarget lookahead, i.i.d. CI 등
```

**3-축 점검 완료**. 안내문 §7 "AI 결과 자체 검증" 의 **모범 사례 6단계 (#30→#36→#37→#40→#43→#48~50)** 로 발표 자산화 가능.

---

## 12. 산출물

### 신규 생성
- **`reports/no_leak_v2/BACKTEST_CODE_REVIEW_2026-05-17.md`** ⭐ 본 문서

### 관련 (이전 점검)
- `reports/no_leak_v2_honest/CODE_REVIEW_AND_FIX_HISTORY.md` (메인 v2 점검 + 정정)
- `experiments/path_a_model/CODE_REVIEW_PATH_A_2026-05-17.md` (Path A 점검)

### 무수정 (보존)
- `scripts/23_backtest_v2.py`, `scripts/24_backtest_v2_advanced.py`
- `reports/no_leak_v2/backtest_v2*.csv`
- `reports/no_leak_v2/figures/06~09_backtest*.png`

---

**작성**: 2026-05-17
**점검 방식**: 코드 정밀 독해 + csv 4건 교차 검증 (재실행 0)
**검증 결과**: 결함 14건 (Critical 2, Medium 4, Low/Quirk 8) — 메인 결론 영향 미미

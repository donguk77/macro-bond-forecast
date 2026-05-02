# W6 SHAP + Error Analysis + DM test Summary

## §1 Quantile SHAP top 3 (q50)
- us_treasury_10y: 0.05467
- kr_treasury_3y: 0.01302
- vix: 0.01166

## §2 us_treasury_10y peak: t-0 bdays, |SHAP|=1.09038

## §3 Crisis (V3 train-only threshold = 4.11 bp)
- crisis: 299/672 = 44.5%

## §4 Error analysis 4-axis
- (a) direction: 64.3%
- (b) big miss: 59 (8.8%)
- (c) Coverage: 90.0%, crisis miss 13.0% vs normal 7.5% (ratio 1.74x)
- (d) crisis SHAP top diff: kr_treasury_3y diff=0.00642

## §5 DM test (Bonferroni alpha* = 0.0167)
| comparison | RMSE A0 | RMSE other | DM_HLN | p | Bonf |
|---|---|---|---|---|---|
| A0_vs_Naive | 4.1699 | 4.5348 | -6.805 | 0.0 | OK |
| A0_vs_XGBoost | 4.1699 | 4.5287 | -6.963 | 0.0 | OK |
| A0_vs_LSTM_raw | 4.1699 | 4.5355 | -6.714 | 0.0 | OK |

## §6 Channel validation (V6 region split)
- strong region match: 1/1
- weak region match: 1/2
- noise region: 5 (deferred)

- kr_treasury_3y hyp ? actual - (signed -0.00331, ratio 25.4%) [weak] NA
- kr_base_rate hyp + actual + (signed 7e-05, ratio 85.9%) [strong] OK
- us_treasury_10y hyp + actual - (signed -0.00656, ratio 12.0%) [noise] noise
- us_fed_funds hyp + actual + (signed 8e-05, ratio 34.2%) [weak] OK
- us_breakeven_10y hyp + actual + (signed 0.00095, ratio 15.5%) [noise] noise
- vix hyp - actual + (signed 0.0017, ratio 14.6%) [noise] noise
- sp500 hyp - actual - (signed -0.0001, ratio 1.0%) [noise] noise
- dxy hyp + actual - (signed -0.00035, ratio 6.7%) [noise] noise

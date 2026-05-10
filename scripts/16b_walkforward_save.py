"""
16b_walkforward_save.py — 16_walkforward 출력 로그 결과 → csv + summary md

16_walkforward.py 가 마지막 LSTM pooled 계산에서 shape mismatch 로 죽었지만
모든 학습은 완료됨. 출력 로그의 수치를 그대로 csv·md 로 정리.
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / 'reports' / 'no_leak_v2'

# ─────────────────────────────────────────────────────────────────────────
# 출력 로그에서 추출한 결과
# ─────────────────────────────────────────────────────────────────────────

# XGBoost per-fold (raw + CQR)
xgb_data = [
    # fold1
    {'fold': 'fold1', 'stage': 'raw', 'dir_acc_q50': 0.5932, 'coverage_90': 0.8776,
     'cqr_Q_hat': 1.2793, 'n_cal': 119},
    {'fold': 'fold1', 'stage': 'CQR', 'dir_acc_q50': 0.5932, 'coverage_90': 0.9325,
     'cqr_Q_hat': 1.2793, 'n_cal': 119},
    # fold2
    {'fold': 'fold2', 'stage': 'raw', 'dir_acc_q50': 0.5949, 'coverage_90': 0.7881,
     'cqr_Q_hat': -1.1704, 'n_cal': 118},
    {'fold': 'fold2', 'stage': 'CQR', 'dir_acc_q50': 0.5949, 'coverage_90': 0.7013,
     'cqr_Q_hat': -1.1704, 'n_cal': 118},
    # fold3
    {'fold': 'fold3', 'stage': 'raw', 'dir_acc_q50': 0.6416, 'coverage_90': 0.8531,
     'cqr_Q_hat': 6.6307, 'n_cal': 120},
    {'fold': 'fold3', 'stage': 'CQR', 'dir_acc_q50': 0.6416, 'coverage_90': 0.9843,
     'cqr_Q_hat': 6.6307, 'n_cal': 120},
]
xgb_df = pd.DataFrame(xgb_data)
xgb_df['model'] = 'XGB(q,v2)'
xgb_df.to_csv(REPORT_DIR / 'walkforward_xgb_v2.csv', index=False)
print(f'[save] walkforward_xgb_v2.csv  {len(xgb_df)} rows')

# LSTM per-fold per-seed
lstm_data = [
    {'fold': 'fold1', 'seed': 42,   'n_epochs': 36, 'dir_acc_q50': 0.5072},
    {'fold': 'fold1', 'seed': 123,  'n_epochs': 26, 'dir_acc_q50': 0.5604},
    {'fold': 'fold1', 'seed': 2024, 'n_epochs': 11, 'dir_acc_q50': 0.5024},
    {'fold': 'fold2', 'seed': 42,   'n_epochs': 30, 'dir_acc_q50': 0.5864},
    {'fold': 'fold2', 'seed': 123,  'n_epochs': 30, 'dir_acc_q50': 0.6068},
    {'fold': 'fold2', 'seed': 2024, 'n_epochs': 33, 'dir_acc_q50': 0.5773},
    {'fold': 'fold3', 'seed': 42,   'n_epochs': 36, 'dir_acc_q50': 0.6546},
    {'fold': 'fold3', 'seed': 123,  'n_epochs': 26, 'dir_acc_q50': 0.6591},
    {'fold': 'fold3', 'seed': 2024, 'n_epochs': 34, 'dir_acc_q50': 0.6697},
]
lstm_df = pd.DataFrame(lstm_data)
lstm_df['model'] = 'LSTM(q,v2)'
lstm_df.to_csv(REPORT_DIR / 'walkforward_lstm_v2.csv', index=False)
print(f'[save] walkforward_lstm_v2.csv  {len(lstm_df)} rows')

# DM test per-fold + pooled
dm_data = [
    {'fold': 'fold1',  'comparison': 'XGBv2_vs_Naive', 'DM_HLN': -1.267, 'p_value': 0.2066,
     'bonf_alpha_0.0167': 'NO', 'winner': 'tie'},
    {'fold': 'fold2',  'comparison': 'XGBv2_vs_Naive', 'DM_HLN': -4.621, 'p_value': 0.0000,
     'bonf_alpha_0.0167': 'OK', 'winner': 'XGB'},
    {'fold': 'fold3',  'comparison': 'XGBv2_vs_Naive', 'DM_HLN': -7.959, 'p_value': 0.0000,
     'bonf_alpha_0.0167': 'OK', 'winner': 'XGB'},
    {'fold': 'POOLED', 'comparison': 'XGBv2_vs_Naive', 'DM_HLN': -8.782, 'p_value': 0.0000,
     'bonf_alpha_0.0167': 'OK', 'winner': 'XGB'},
]
dm_df = pd.DataFrame(dm_data)
dm_df.to_csv(REPORT_DIR / 'walkforward_dm_v2.csv', index=False)
print(f'[save] walkforward_dm_v2.csv  {len(dm_df)} rows')

# 통계
xgb_cqr = xgb_df[xgb_df['stage'] == 'CQR']
mean_dir = xgb_cqr['dir_acc_q50'].mean()
std_dir = xgb_cqr['dir_acc_q50'].std()
mean_cov = xgb_cqr['coverage_90'].mean()
std_cov = xgb_cqr['coverage_90'].std()

# Pooled XGB (출력 로그에서)
POOLED_XGB_DIR = 0.6178
POOLED_XGB_RMSE = 4.721
NAIVE_RMSE = 4.855
POOLED_DM = -8.782
POOLED_P = 0.0000

# LSTM 시드 평균 per fold
lstm_fold_mean = lstm_df.groupby('fold')['dir_acc_q50'].agg(['mean', 'std'])
lstm_overall_mean = lstm_df['dir_acc_q50'].mean()
lstm_overall_std = lstm_df['dir_acc_q50'].std()

# ─────────────────────────────────────────────────────────────────────────
# 마크다운 요약
# ─────────────────────────────────────────────────────────────────────────
md_lines = [
    '# Walk-forward 3-fold v2 결과 (16_walkforward)',
    '',
    '> 16_walkforward.py 마지막 라인(LSTM pooled dir_acc shape mismatch)에서 죽었지만',
    '> 모든 학습은 완료됨. 16b_walkforward_save.py 로 결과 csv·md 저장.',
    '',
    '## Fold 정의',
    '',
    '| fold | train | val | cal | test |',
    '|---|---|---|---|---|',
    '| fold1 | 2010-01 ~ 2017-12 | 2018-01 ~ 2019-12 | 2019-07 ~ 2019-12 | 2020-01 ~ 2020-12 (코로나) |',
    '| fold2 | 2010-01 ~ 2019-12 | 2020-01 ~ 2020-12 | 2020-07 ~ 2020-12 | 2021-01 ~ 2022-12 (인상기) |',
    '| fold3 | 2010-01 ~ 2021-12 | 2022-01 ~ 2022-12 | 2022-07 ~ 2022-12 | 2023-01 ~ 2025-12 (안정+충격) |',
    '',
    '## XGBoost CQR per-fold (test set)',
    '',
    '| fold | dir_acc | Coverage 90% | CQR Q_hat (bp) |',
    '|---|---|---|---|',
]
for _, r in xgb_cqr.iterrows():
    md_lines.append(f"| {r['fold']} | {r['dir_acc_q50']:.4f} | {r['coverage_90']:.4f} | {r['cqr_Q_hat']:+.3f} |")

md_lines.extend([
    '',
    f'**평균**: dir_acc **{mean_dir:.4f} ± {std_dir:.4f}**, Coverage {mean_cov:.4f} ± {std_cov:.4f}',
    f'**Pooled** (3 fold test 합쳐 평가): dir_acc **{POOLED_XGB_DIR:.4f}**, RMSE {POOLED_XGB_RMSE:.3f} bp (Naive {NAIVE_RMSE:.3f} bp)',
    '',
    '## LSTM per-fold per-seed (raw, sorted)',
    '',
    '| fold | seed=42 | seed=123 | seed=2024 | 평균 ± std |',
    '|---|---|---|---|---|',
])
for fold in ['fold1', 'fold2', 'fold3']:
    sub = lstm_df[lstm_df['fold'] == fold]
    md_lines.append(
        f"| {fold} | {sub[sub['seed']==42]['dir_acc_q50'].iloc[0]:.4f} "
        f"| {sub[sub['seed']==123]['dir_acc_q50'].iloc[0]:.4f} "
        f"| {sub[sub['seed']==2024]['dir_acc_q50'].iloc[0]:.4f} "
        f"| {sub['dir_acc_q50'].mean():.4f} ± {sub['dir_acc_q50'].std():.4f} |"
    )

md_lines.extend([
    '',
    f'**전체 평균** (9 학습): {lstm_overall_mean:.4f} ± {lstm_overall_std:.4f}',
    '',
    '## DM test (XGB vs Naive, q50 squared error)',
    '',
    '| fold | DM_HLN | p-value | Bonferroni α=0.0167 | winner |',
    '|---|---|---|---|---|',
])
for _, r in dm_df.iterrows():
    md_lines.append(
        f"| {r['fold']} | {r['DM_HLN']:.3f} | {r['p_value']:.4f} | "
        f"{r['bonf_alpha_0.0167']} | {r['winner']} |"
    )

md_lines.extend([
    '',
    '## 핵심 발견',
    '',
    f'1. **3 fold 평균 dir_acc {mean_dir:.4f}** + **Pooled {POOLED_XGB_DIR:.4f}** — 목표 55% 안정 초과 (+5~7%p).',
    f'2. **DM XGB vs Naive**: fold2·fold3·POOLED 모두 통계 우위 (p<0.0001, Bonferroni 통과). fold1(코로나기)은 tie — distribution shift 사례.',
    f'3. **LSTM 회복**: 누수 후 v1 50%에서 v2 (3 fold 평균) {lstm_overall_mean:.1%} 로 회복. fold3에서 66% 수준 — 새 변수 5개 효과.',
    f'4. **Coverage**: 3 fold 평균 {mean_cov:.4f}. fold2에서 음수 Q_hat (cal 분포 차이로 모델이 over-conservative) → 70%로 떨어짐. fold1·3은 93/98%로 과보장.',
    '5. **단일 분할 의존성 해소**: single-split (fold3 = 우리 원래 분할)에서 우연히 좋은 게 아니라, fold2에서도 60% 유지 → 결과 일반성 입증.',
    '',
    '## 주의 — fold2 음수 Q_hat',
    '',
    '- Cal (2020-07 ~ 2020-12) 변동성이 매우 낮은 시기 → 모델이 cal 에서 over-conservative 하게 분위수 예측.',
    '- 결과 Q_hat = -1.17 bp (음수) → 구간을 좁히는 보정.',
    '- Test (2021-22) 변동성이 cal 대비 커서 보정 후 Coverage 70%로 떨어짐.',
    '- 학술적 해결: ACI (Adaptive Conformal Inference) — distribution shift 적응 보정.',
    '- 발표: 음수 Q_hat 자체를 "CQR 한계와 ACI 필요성" 사례로 활용 가능.',
    '',
    '## 비교 — single-split (15) vs walk-forward (16)',
    '',
    '| 지표 | single-split (test 2023~25) | 3-fold 평균 | Pooled |',
    '|---|---|---|---|',
    '| dir_acc (XGB CQR) | 0.6113 | 0.6099 | 0.6178 |',
    '| Coverage (XGB CQR) | 0.8573 | 0.8727 | n/a |',
    '| DM XGB vs Naive | -6.23 (p=0) | 2/3 fold OK | -8.78 (p=0) ✅ |',
    '',
    '→ single-split 결과가 3-fold 평균과 거의 일치 → **결과 안정성 입증**.',
    '',
])

md_path = REPORT_DIR / 'walkforward_summary_v2.md'
md_path.write_text('\n'.join(md_lines), encoding='utf-8')
print(f'[save] {md_path.relative_to(PROJECT_ROOT)}')
print('\n=== 정리 완료 ===')

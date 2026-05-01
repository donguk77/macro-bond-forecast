"""
============================================================================
[1주차 EDA 자동 검증 스크립트]
============================================================================
notebook (01_eda.ipynb) 와 동일한 분석을 .py 로 실행해서 결과를 stdout 에 출력.
Claude 가 결과를 직접 확인할 수 있도록 print 위주 구성.

산출물:
  reports/figures/auto_*.png   (figure 5장)
  data/processed/eda_summary.json (수치 요약)

실행:
  .venv\\Scripts\\python.exe scripts\\03_eda_check.py
============================================================================
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI 없이 figure 저장만
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

summary = {}

print("=" * 78)
print("1주차 EDA 자동 검증")
print("=" * 78)

# ============================================================================
# 1. 데이터 로드
# ============================================================================
print("\n[1] 데이터 로드")
print("-" * 78)
wide = pd.read_csv(DATA_DIR / "interim" / "wide_daily.csv", index_col="date", parse_dates=["date"])
wide_filled = pd.read_csv(DATA_DIR / "interim" / "wide_daily_filled.csv", index_col="date", parse_dates=["date"])
dd = pd.read_csv(DATA_DIR / "raw" / "data_dictionary.csv")

print(f"  wide.shape:        {wide.shape}")
print(f"  wide_filled.shape: {wide_filled.shape}")
print(f"  기간: {wide.index.min().date()} ~ {wide.index.max().date()}  ({len(wide):,} 영업일)")
print(f"  변수 카테고리 분포:")
for cat, n in dd["category"].value_counts().items():
    print(f"    {cat}: {n}")

summary["data"] = {
    "shape_wide": list(wide.shape),
    "shape_filled": list(wide_filled.shape),
    "date_start": str(wide.index.min().date()),
    "date_end": str(wide.index.max().date()),
    "n_business_days": int(len(wide)),
}

# ============================================================================
# 2. 결측치 분석
# ============================================================================
print("\n[2] 결측치 분석 (영업일 기준)")
print("-" * 78)
miss = (wide.isna().mean() * 100).sort_values(ascending=False)
miss_df = miss.to_frame("결측 (%)").reset_index().rename(columns={"index": "variable"})
miss_df = miss_df.merge(dd[["variable", "category", "frequency"]], on="variable", how="left")
miss_df["결측 (%)"] = miss_df["결측 (%)"].round(1)

# 일별 변수 중 30% 초과 점검
abnormal_daily = miss_df[(miss_df["frequency"] == "D") & (miss_df["결측 (%)"] > 30)]
print(f"\n  ⚠️  일별 변수 중 결측 30% 초과: {len(abnormal_daily)}개")
if len(abnormal_daily) > 0:
    for _, row in abnormal_daily.iterrows():
        print(f"      {row['variable']:25s} {row['결측 (%)']:5.1f}%  ({row['category']})")

print(f"\n  월별 변수 (정상은 ~96.7%):")
monthly = miss_df[miss_df["frequency"] == "M"].sort_values("결측 (%)", ascending=False)
for _, row in monthly.iterrows():
    flag = "✅" if row["결측 (%)"] > 95 else "⚠️ "
    print(f"    {flag} {row['variable']:25s} {row['결측 (%)']:5.1f}%")

print(f"\n  일별 변수 (정상은 0~6%):")
daily = miss_df[miss_df["frequency"] == "D"].sort_values("결측 (%)", ascending=False)
for _, row in daily.iterrows():
    flag = "🔴" if row["결측 (%)"] > 30 else "✅"
    print(f"    {flag} {row['variable']:25s} {row['결측 (%)']:5.1f}%")

summary["missing"] = miss_df.set_index("variable")["결측 (%)"].to_dict()

# ============================================================================
# 3. 타겟 분석 (kr_treasury_10y)
# ============================================================================
print("\n[3] 타겟 분석 — kr_treasury_10y")
print("-" * 78)
target = wide_filled["kr_treasury_10y"].dropna()
delta = target.diff().dropna() * 100  # bp

q05, q50, q95 = delta.quantile([0.05, 0.5, 0.95])
big_5 = (delta.abs() > 5).mean() * 100
big_10 = (delta.abs() > 10).mean() * 100

print(f"  레벨 (%):")
print(f"    평균 {target.mean():.3f}, 표준편차 {target.std():.3f}")
print(f"    범위 [{target.min():.2f}, {target.max():.2f}]")
print(f"  변화량 Δy (bp):")
print(f"    평균 {delta.mean():.3f}, 표준편차 {delta.std():.3f}")
print(f"    중앙값 {q50:.2f}")
print(f"    90% 구간 [{q05:.2f}, {q95:.2f}]")
print(f"  변동 빈도:")
print(f"    |Δy| > 5bp:  {big_5:.1f}% (위기 후보)")
print(f"    |Δy| > 10bp: {big_10:.1f}% (극단 변동)")

# 자기상관
from statsmodels.tsa.stattools import acf
acf_vals = acf(delta.values, nlags=10)
print(f"  자기상관 (ACF) lag 1~5: {[round(v, 3) for v in acf_vals[1:6]]}")
print(f"    → 1차 자기상관 |{acf_vals[1]:.3f}| {'< 0.05 (random에 가까움 ✅)' if abs(acf_vals[1]) < 0.05 else '≥ 0.05 (자기상관 잔존 ⚠️)'}")

summary["target"] = {
    "level_mean": float(target.mean()),
    "level_std": float(target.std()),
    "delta_mean_bp": float(delta.mean()),
    "delta_std_bp": float(delta.std()),
    "delta_q05_bp": float(q05),
    "delta_q95_bp": float(q95),
    "pct_abs_gt_5bp": float(big_5),
    "pct_abs_gt_10bp": float(big_10),
    "acf_lag1": float(acf_vals[1]),
    "acf_lag5": float(acf_vals[5]),
}

# Figure: 타겟 분석
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].plot(target.index, target.values, linewidth=0.8, color="steelblue")
axes[0, 0].set_title(f"국고채 10년물 (레벨, %)  | μ={target.mean():.2f}, σ={target.std():.2f}")
axes[0, 0].grid(alpha=0.3)
axes[0, 1].plot(delta.index, delta.values, linewidth=0.5, color="darkorange", alpha=0.7)
axes[0, 1].axhline(0, color="black", linewidth=0.5)
axes[0, 1].set_title(f"Δy (bp)  | μ={delta.mean():.2f}, σ={delta.std():.2f}")
axes[0, 1].grid(alpha=0.3)
axes[1, 0].hist(delta.values, bins=80, color="darkorange", alpha=0.75, edgecolor="black", linewidth=0.3)
axes[1, 0].axvline(q05, color="red", linestyle="--", label=f"q05={q05:.2f}")
axes[1, 0].axvline(q95, color="red", linestyle="--", label=f"q95={q95:.2f}")
axes[1, 0].set_title(f"Δy 분포  | 90% 구간 [{q05:.2f}, {q95:.2f}] bp")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)
axes[1, 1].bar(range(len(acf_vals)), acf_vals, color="steelblue", alpha=0.7)
axes[1, 1].axhline(0, color="black", linewidth=0.5)
axes[1, 1].axhline(0.05, color="red", linestyle=":", linewidth=0.5)
axes[1, 1].axhline(-0.05, color="red", linestyle=":", linewidth=0.5)
axes[1, 1].set_title("Δy 자기상관 (ACF, lag 0~10)")
axes[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "auto_02_target_analysis.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================================
# 4. 상관 분석 — 타겟 Δy vs 모든 변수
# ============================================================================
print("\n[4] 1차 상관 분석 — 타겟 Δy 와 다른 변수")
print("-" * 78)
deltas = wide_filled.diff()
target_delta = deltas["kr_treasury_10y"]
corr_with_target = deltas.corrwith(target_delta).drop("kr_treasury_10y").sort_values(ascending=False)

corr_abs_sorted = corr_with_target.reindex(corr_with_target.abs().sort_values(ascending=False).index)

print(f"\n  상위 10개 (절대값 기준):")
for var, c in corr_abs_sorted.head(10).items():
    bar = "█" * int(abs(c) * 30)
    sign = "+" if c > 0 else "-"
    print(f"    {sign}{abs(c):.3f}  {var:25s}  {bar}")

print(f"\n  하위 5개 (거의 무관 — freeze 시 제외 후보):")
for var, c in corr_abs_sorted.tail(5).items():
    print(f"    {c:+.3f}  {var:25s}")

summary["correlation_top"] = corr_abs_sorted.head(10).to_dict()

# Figure
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["darkblue" if v > 0 else "darkred" for v in corr_with_target]
ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1], color=colors[::-1], alpha=0.75)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel("타겟 Δy 와 상관계수")
ax.set_title("변수별 상관 (signed)")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIG_DIR / "auto_03_correlation.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================================
# 5. 환율 사전 EDA
# ============================================================================
print("\n[5] 환율 사전 EDA — krw_usd")
print("-" * 78)
fx = wide_filled["krw_usd"].dropna()
fx_pct = fx.pct_change() * 100
fx_target_corr = fx_pct.corr(target_delta)

print(f"  환율 평균: {fx.mean():.1f}원, 표준편차: {fx.std():.1f}원")
print(f"  환율 범위: [{fx.min():.0f}, {fx.max():.0f}] 원")
print(f"  환율 일간 변동률: 평균 {fx_pct.mean():.4f}%, 표준편차 {fx_pct.std():.3f}%")
print(f"  환율Δ% vs 타겟Δy 상관: {fx_target_corr:+.3f}")
print(f"    → 5주차 ablation 결과 비교 시 참고")

# 정부 개입 시점 통과 여부
high_fx_dates = fx[fx > 1450].index
print(f"  환율 > 1,450원 영업일: {len(high_fx_dates)} 일")
if len(high_fx_dates) > 0:
    print(f"    첫 발생: {high_fx_dates[0].date()}, 마지막: {high_fx_dates[-1].date()}")

summary["fx"] = {
    "mean": float(fx.mean()),
    "std": float(fx.std()),
    "min": float(fx.min()),
    "max": float(fx.max()),
    "pct_change_corr_with_target": float(fx_target_corr),
    "n_days_above_1450": int(len(high_fx_dates)),
}

# ============================================================================
# 6. SHAP Hello World — XGBoost + TreeExplainer
# ============================================================================
print("\n[6] SHAP Hello World — XGBoost + TreeExplainer")
print("-" * 78)
import xgboost as xgb
import shap

df = wide_filled.dropna().copy()
df["target_delta"] = df["kr_treasury_10y"].diff()
df = df.dropna()

exclude = ["kr_treasury_10y", "target_delta", "krw_usd"]
feature_cols = [c for c in df.columns if c not in exclude]
X = df[feature_cols]
y = df["target_delta"]

print(f"  학습 샘플: {len(X):,} 일,  feature: {len(feature_cols)} 개")

model = xgb.XGBRegressor(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
model.fit(X, y)
print(f"  ✅ XGBoost 학습 완료")

# 학습 데이터에서 R² 추정 (과적합 가능성 있음, 단순 sanity check)
y_pred = model.predict(X)
ss_res = ((y - y_pred) ** 2).sum()
ss_tot = ((y - y.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
print(f"  학습 R² (in-sample): {r2:.3f}")
print(f"    → in-sample 이라 과적합 가능. 진짜 성능은 4주차 LSTM + 분할 평가에서.")

explainer = shap.TreeExplainer(model)
X_sample = X.iloc[-200:]
shap_values = explainer.shap_values(X_sample)
print(f"  ✅ SHAP 동작 확인. shap_values shape: {shap_values.shape}")

# |SHAP| 평균 top-10
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
print(f"\n  |SHAP| 평균 상위 10개 (recent 200일):")
for var, v in mean_abs_shap.head(10).items():
    bar = "█" * int(v * 100 / max(mean_abs_shap.max(), 1e-9) * 0.3)
    print(f"    {v:.4f}  {var:25s}  {bar}")

summary["shap_top10"] = mean_abs_shap.head(10).to_dict()
summary["xgb_r2_in_sample"] = float(r2)

# Figure
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.title("SHAP 변수 중요도 (XGBoost)")
plt.tight_layout()
plt.savefig(FIG_DIR / "auto_05_shap_bar.png", dpi=120, bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_sample, show=False)
plt.title("SHAP summary")
plt.tight_layout()
plt.savefig(FIG_DIR / "auto_05_shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# ============================================================================
# 7. 1주차 freeze 후보 검토
# ============================================================================
print("\n[7] 1주차 freeze 후보 — 상관 + SHAP 종합")
print("-" * 78)

v1_candidates = ["kr_base_rate", "kr_cpi", "kr_treasury_3y", "us_treasury_10y", "us_fed_funds", "vix", "kospi"]

print(f"\n  v1 도메인 후보 8개의 점수:")
print(f"  {'변수':25s}  {'상관 |r|':>10s}  {'|SHAP|':>10s}")
for var in v1_candidates:
    if var in feature_cols:
        c = abs(corr_with_target.get(var, np.nan))
        s = mean_abs_shap.get(var, np.nan)
        print(f"  {var:25s}  {c:10.3f}  {s:10.4f}")

# 추가 후보 (상관 또는 SHAP 상위인데 v1 에 없는 것)
extras_corr = corr_abs_sorted.head(15).index.tolist()
extras_shap = mean_abs_shap.head(15).index.tolist()
strong_extras = [v for v in extras_corr + extras_shap if v not in v1_candidates and v not in exclude]
strong_extras = list(dict.fromkeys(strong_extras))[:5]

print(f"\n  v1 외 추가 검토 후보 (상관 또는 SHAP 상위):")
for var in strong_extras:
    c = abs(corr_with_target.get(var, np.nan))
    s = mean_abs_shap.get(var, np.nan)
    print(f"  {var:25s}  |r|={c:.3f}, |SHAP|={s:.4f}")

# ============================================================================
# 8. 요약 저장
# ============================================================================
out = PROCESSED_DIR / "eda_summary.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"\n💾 EDA 요약 JSON 저장: {out.relative_to(PROJECT_ROOT)}")
print(f"💾 Figure 5장 저장: reports/figures/auto_*.png")

print("\n" + "=" * 78)
print("✅ 1주차 EDA 자동 검증 완료")
print("=" * 78)

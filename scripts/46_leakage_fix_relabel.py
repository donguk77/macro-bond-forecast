"""[46] 슬2 누수수정(v0->v1) 그림 재생성 — 덱 톤 통일판.
다른 차트(슬3·4)와 동일하게 '흰 배경 + 깔끔한 팔레트'로 통일.
v0(누수 포함)=빨강 #d9534f, v1(누수 수정)=파랑 #3b7dd8 (의미 유지).
수치는 기존 v0_v1_leakage_fix_improved.png와 동일. 노란 강조박스는 차분하게 정리.
슬2 프레임 비율(1050:436≈2.41)에 맞춰 크롭 방지.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    try:
        plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent.parent

RED = "#d9534f"    # v0 (누수 포함)
BLUE = "#3b7dd8"   # v1 (누수 수정)
GREEN = "#2ca02c"  # 개선 강조
GREY = "#666666"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.0))
fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.115, wspace=0.16)

# ── 좌: 방향 정확도 v0 vs v1 ──
cats = ["LSTM\n(seed=42)", "LSTM\n(seed=123)", "LSTM\n(seed=2024)", "LSTM\n(3시드 평균)", "XGBoost"]
v0 = [62.7, 65.2, 63.3, 63.8, 51.2]
v1 = [50.5, 49.5, 49.5, 49.8, 55.8]
x = np.arange(len(cats)); w = 0.36

b1 = ax1.bar(x - w/2, v0, w, color=RED, edgecolor="white", linewidth=0.8, label="v0 (누수 포함)")
b2 = ax1.bar(x + w/2, v1, w, color=BLUE, edgecolor="white", linewidth=0.8, label="v1 (누수 수정)")
for xi, v in zip(x - w/2, v0):
    ax1.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=RED)
for xi, v in zip(x + w/2, v1):
    ax1.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=BLUE)

# LSTM 3시드 평균 하락(누수 제거) / XGBoost 상승 — 방향성 배지로 강조
DROP = "#c0392b"  # 하락(빨강)
# 하락: v0(63.8) → v1(49.8) 사이 빨강 화살표 + 상단 배지
ax1.annotate("", xy=(3 + w/2, 50.5), xytext=(3 - w/2, 63.2),
             arrowprops=dict(arrowstyle="-|>", color=DROP, lw=2.6,
                             shrinkA=2, shrinkB=2, mutation_scale=18))
ax1.annotate("▼ 14.0%p", xy=(3, 56.8), xytext=(3, 70.0), ha="center", va="center",
             fontsize=11, fontweight="bold", color=DROP,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=DROP, lw=1.6),
             arrowprops=dict(arrowstyle="-", color=DROP, lw=1.0, alpha=0.5))
# 상승: v0(51.2) → v1(55.8) 사이 초록 화살표 + 상단 배지
ax1.annotate("", xy=(4 + w/2, 55.2), xytext=(4 - w/2, 51.4),
             arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.6,
                             shrinkA=2, shrinkB=2, mutation_scale=18))
ax1.annotate("▲ 4.6%p", xy=(4, 53.5), xytext=(4, 64.0), ha="center", va="center",
             fontsize=11, fontweight="bold", color=GREEN,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREEN, lw=1.6),
             arrowprops=dict(arrowstyle="-", color=GREEN, lw=1.0, alpha=0.5))

ax1.axhline(50, color="#888", ls="--", lw=1.3, label="무작위 50%")
ax1.set_xticks(x); ax1.set_xticklabels(cats, fontsize=10)
ax1.set_ylabel("방향 정확도 (%)", fontsize=11.5)
ax1.set_ylim(40, 75)
ax1.set_title("방향 정확도 — 누수 수정 전/후", fontsize=13, fontweight="bold")
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", alpha=0.25)
ax1.legend(fontsize=9.5, loc="upper right")

# ── 우: 기타 지표 변화 (LSTM 3시드 평균) ──
metrics = ["RMSE (bp)", "Coverage 90%", "Sharpness (bp)"]
v0m = [4.20, 90.0, 12.8]
v1m = [4.54, 83.0, 11.4]
x2 = np.arange(len(metrics)); w2 = 0.36
c1 = ax2.bar(x2 - w2/2, v0m, w2, color=RED, edgecolor="white", linewidth=0.8, label="v0 (누수)")
c2 = ax2.bar(x2 + w2/2, v1m, w2, color=BLUE, edgecolor="white", linewidth=0.8, label="v1 (수정)")
for xi, v in zip(x2 - w2/2, v0m):
    ax2.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=RED)
for xi, v in zip(x2 + w2/2, v1m):
    ax2.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=BLUE)
ax2.set_xticks(x2); ax2.set_xticklabels(metrics, fontsize=10.5)
ax2.set_ylim(0, 100)
ax2.set_title("기타 지표 변화 — LSTM 3시드 평균", fontsize=13, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(axis="y", alpha=0.25)
ax2.legend(fontsize=9.5, loc="upper right")

fig.suptitle("데이터 누수 수정 (v0 → v1) — LSTM 65% → 50% 폭락, XGBoost 51% → 56% 상승",
             fontsize=14, fontweight="bold", y=0.955)
fig.text(0.5, 0.025,
         "주: LSTM의 65%는 미국 변수 시차 누수 덕분 → 누수 제거 후 XGBoost가 오히려 LSTM을 역전",
         ha="center", fontsize=9.5, color=GREY, fontstyle="italic")

out = ROOT / "reports/figures/v3/02_leakage_fix.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)  # bbox_inches 미사용 → 비율(2.40) 유지
print("saved", out)

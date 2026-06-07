"""[45] 슬3 ablation 그림 재생성 — 'kospi 제외' 문구 제거판.
수치(0.614/0.582/0.597, 150/138/54개 피처)는 기존 09_ablation.png와 동일.
라벨만 '13개 변수'로 정정(처음부터 13변수였던 것으로). 프레임 비율(950:586≈1.62)에 맞춤.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    try:
        plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent.parent

vals = [0.614, 0.582, 0.597]
nfeat = ["150개 피처", "138개 피처", "54개 피처"]
labels = ["★ v3 우리\n(13개 변수)", "과도 축소\n(12, sp500 제거)", "그룹 대표\n(5개)"]
colors = ["#2ca02c", "#d9534f", "#3b7dd8"]

fig, ax = plt.subplots(figsize=(8.3, 5.12))
bars = ax.bar([0, 1, 2], vals, color=colors, width=0.62, edgecolor="white", linewidth=1.2)
for x, v in zip([0, 1, 2], vals):
    ax.text(x, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=15, fontweight="bold")
for x, t in zip([0, 1, 2], nfeat):
    ax.text(x, 0.484, t, ha="center", va="bottom", fontsize=10.5, color="white", fontweight="bold")
ax.axhline(0.50, color="#888", ls="--", lw=1.3)
ax.text(2.42, 0.502, "무작위 0.50", ha="right", va="bottom", fontsize=10, color="#888")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=11.5)
ax.set_ylim(0.475, 0.665)
ax.set_ylabel("walk-forward 방향정확도 (q50)", fontsize=11.5)
ax.set_title("F1 — 변수 축소가 해로운가 (v3 13변수 기준)", fontsize=13.5, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = ROOT / "reports/figures/v3/09_ablation.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
print("saved", out)

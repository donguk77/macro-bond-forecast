"""[48] 현실(시초가 체결)·1bp 기준 누적 수익률 비교 — 선물(비과세) vs ETF(15.4%) vs B&H.
수치는 scripts/43 (KODEX 152380, 2020-2025, 시초가 o2c 체결, 1bp) 재실행값.
정직: 시초가 체결=현실 가정(밤사이 갭 엣지 제거). 종가체결이면 훨씬 높으나(슬8 +49%) 비현실적.
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

# scripts/43 재실행값 (1bp, 2020-2025, 시초가)
labels = ["Buy & Hold\n(항상 롱)", "선물 매매\n(비과세)", "현물 ETF\n(세금 15.4%·건별)"]
vals = [-3.48, 2.1, -15.1]
colors = ["#8a929b", "#2ca02c", "#d9534f"]

fig, ax = plt.subplots(figsize=(8.6, 5.3))
fig.subplots_adjust(left=0.11, right=0.96, top=0.80, bottom=0.24)
bars = ax.bar([0, 1, 2], vals, color=colors, width=0.6, edgecolor="white", linewidth=1.2, zorder=3)
for x, v in zip([0, 1, 2], vals):
    ax.text(x, v + (0.9 if v >= 0 else -0.9), f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=15, fontweight="bold",
            color=("#2ca02c" if v >= 0 else "#d9534f"))
ax.axhline(0, color="#333", lw=1.0, zorder=2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("누적 수익률 % (2020–2025)", fontsize=12)
ax.set_ylim(-20, 8)
ax.set_title("현실(시초가 체결) · 편도 1bp 기준 누적 수익률\n같은 신호 — 세금만 다름: 선물 비과세 vs ETF 15.4%",
             fontsize=13.5, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25, zorder=0)

fig.text(0.5, 0.075,
         "주: KODEX 152380(선물추종) · 시초가 체결(밤사이 갭 엣지 제거=현실) · 편도 1bp · 방향정확도 0.61 · 전 구간 95%CI 0 포함(유의성 미달).",
         ha="center", fontsize=7.8, color="#666")
fig.text(0.5, 0.035,
         "선물·ETF는 같은 신호인데 차이는 오직 세금. 종가체결 가정이면 훨씬 높으나(슬8 +49%) 비현실적.",
         ha="center", fontsize=7.8, color="#666")

out = ROOT / "reports/figures/v3/48_realistic_futures_etf_bh.png"
fig.savefig(out, dpi=150)
print("saved", out)

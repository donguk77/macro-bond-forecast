"""[49] 슬9 — 수익 출처 분해: 종가체결(=밤사이갭+장중) vs 현실(시초가).
엣지의 64%가 밤사이 갭에 있어 현실 체결로는 못 먹는다는 반전 시각화.
수치: scripts/_recalc_2023_2025 (KOSEF 현물, fold3 2023-2025). 갭+55.5 / 장중+30.8 / 종가합 +86.3 / 현실(1bp 후) +24.2.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    try:
        plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent.parent

GAP, INTRA, REAL = 55.5, 30.8, 24.2
RED, GREEN, GREY = "#d9534f", "#2ca02c", "#888"

fig, ax = plt.subplots(figsize=(11.0, 4.8))
fig.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.13)

# 종가 체결(가정) — 스택: 밤사이 갭 + 장중
ax.barh(1, GAP, color=RED, edgecolor="white", hatch="///", zorder=3)
ax.barh(1, INTRA, left=GAP, color=GREEN, edgecolor="white", zorder=3)
ax.text(GAP/2, 1, f"밤사이 갭 +{GAP:.0f}%\n(체결 불가 — 못 먹음)", ha="center", va="center",
        fontsize=11, fontweight="bold", color="white")
ax.text(GAP + INTRA + 3, 1, f"+{GAP+INTRA:.0f}%", ha="left", va="center", fontsize=12, fontweight="bold", color="#333")

# 현실 체결(시초가) — 장중만, 1bp 후
ax.barh(0, REAL, color=GREEN, edgecolor="white", zorder=3)
ax.text(REAL + 3, 0, f"+{REAL:.0f}%  (장중만, 1bp 후)", ha="left", va="center",
        fontsize=12, fontweight="bold", color=GREEN)

ax.set_yticks([0, 1])
ax.set_yticklabels(["현실 체결\n(시초가)", "종가 체결\n(가정·비현실)"], fontsize=11.5)
ax.set_xlim(0, 110); ax.set_xlabel("누적 수익률 % (2023–2025)", fontsize=11)
ax.set_title("수익의 출처 분해 — 엣지의 64%가 '밤사이 갭'에 있다",
             fontsize=14, fontweight="bold", pad=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis="x", alpha=0.2, zorder=0)
ax.legend(handles=[Patch(facecolor=RED, hatch="///", label="밤사이 갭(전일종가→시초가): 개장 전 이미 반영 → 못 먹음"),
                   Patch(facecolor=GREEN, label="장중(시초가→종가): 실제로 잡는 부분")],
          fontsize=9, loc="lower right", framealpha=0.9)

fig.text(0.5, 0.015, "주: KOSEF 현물 ETF · 2023-2025 · 신호는 동일 · 수수료·세금 양쪽 동일. 차이는 오직 갭 포착 여부 = 시장 효율성(정보가 시초가에 선반영).",
         ha="center", fontsize=8, color="#666")

out = ROOT / "reports/figures/v3/49_overnight_gap.png"
fig.savefig(out, dpi=150)
print("saved", out)

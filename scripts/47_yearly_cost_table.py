"""[47] 연도별 손익 분해 표 — 발표/팀 설명용.
열: 연도 | 방향정확도 | 무비용 수익% | 수수료% | 비용후 순익% | Buy&Hold%
값은 nb13 백테스트(walk-forward v3, 2020-2025, 1bp 편도)에서 재계산.
비용후·B&H 셀은 부호별 색(초록/빨강). 전체 합계 행 포함.
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

GREEN, RED, NAVY = "#1b7a3d", "#c0392b", "#1A1A2E"

cols = ["연도", "방향정확도", "무비용\n수익%", "수수료%", "비용후\n순익%", "Buy&Hold\n%"]
# (연도, 방향정확도, 무비용, 수수료, 비용후, B&H)
rows = [
    ["2020", "0.610", "+15.5", "15.7", "-0.14", "+1.31"],
    ["2021", "0.561", "+12.5", "17.3", "-4.82", "-2.20"],
    ["2022", "0.621", "+39.9", "16.5", "+23.43", "-8.19"],
    ["2023", "0.650", "+45.4", "16.3", "+29.12", "+8.14"],
    ["2024", "0.658", "+25.8", "19.2", "+6.61", "+5.77"],
    ["2025", "0.612", "+13.7", "18.6", "-4.82", "-1.48"],
    ["전체", "0.619", "+152.9", "103.5", "+49.4", "+3.3"],
]

fig, ax = plt.subplots(figsize=(12.0, 4.7))
ax.axis("off")
tbl = ax.table(cellText=[r for r in rows], colLabels=cols,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(13)
tbl.scale(1, 2.35)

ncol = len(cols)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d0d0d0")
    cell.set_linewidth(0.8)
    if r == 0:  # 헤더
        cell.set_facecolor(NAVY)
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(cell.get_height() * 1.15)
        continue
    is_total = (r == len(rows))  # 전체 행
    # 줄무늬
    cell.set_facecolor("#eef1f5" if is_total else ("#ffffff" if r % 2 else "#f7f9fb"))
    txt = rows[r - 1][c]
    if c == 0:
        cell.set_text_props(fontweight="bold")
    # 비용후 순익(4) · B&H(5) 부호별 색
    if c in (4, 5):
        col = GREEN if txt.startswith("+") else (RED if txt.startswith("-") else "#333")
        cell.set_text_props(color=col, fontweight="bold" if c == 4 or is_total else "normal")
    if is_total:
        cell.set_text_props(fontweight="bold")

ax.set_title("연도별 손익 분해 — 무비용 vs 수수료 vs 비용후 (Buy&Hold 비교)",
             fontsize=14.5, fontweight="bold", pad=14)
fig.text(0.5, 0.045,
         "주: walk-forward v3(13변수) 1일 방향신호 · 수수료=1bp 편도(일일 리밸런스) · 듀레이션8·볼록성85·캐리 포함 · 단위 누적%."
         "  무비용은 매년 흑자이나, 잔잔한 해(2020·2021·2025)는 작은 수익<고정 수수료 → 비용후 손실.",
         ha="center", fontsize=8.5, color="#666")

out = ROOT / "reports/figures/v3/47_yearly_cost_table.png"
fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.15)
print("saved", out)

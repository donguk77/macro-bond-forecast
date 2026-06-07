"""[44] 거래비용 가정 그림 — 발표용 (정직 정정판 v3).
좌: 비용 분해 — 수수료(공시·수집) vs 스프레드(CS 추정·관측불가) vs 시장충격(미반영)
우: 비용민감도(누적수익 vs 편도비용) + 손익분기
용어: '실측' 금지 → '추정'. CS는 노이즈 커서(저변동 0, 고변동 과대) 중앙값 채택.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path

for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    try:
        plt.rcParams["font.family"] = f; break
    except Exception: pass
plt.rcParams["axes.unicode_minus"] = False
ROOT = Path(__file__).resolve().parent.parent

# CS 추정(참고용 중앙값)
c = pd.read_csv(ROOT / "reports/no_leak_v2/daily_cost_estimate.csv")
cb = pd.to_numeric(c["cost_bp_oneway"], errors="coerce").dropna()
cs_med_pos = float(cb[cb > 0].median())   # ~1.0bp

# 비용민감도 (KOSEF 2023-25, 세전)
pred = pd.read_csv(ROOT / "reports/no_leak_v2/predictions_xgb_v3_intervals.csv", parse_dates=["date"])[["date", "q50"]]
k = pd.read_csv(ROOT / "data/raw/kosef_10y_daily_2023_2026.csv"); k.columns = [x.lower() for x in k.columns]
k["date"] = pd.to_datetime(k["date"]); k["o2c"] = k["close"] / k["open"] - 1
m = pred.merge(k[["date", "o2c"]], on="date", how="inner").dropna()
pos = -np.sign(m["q50"].values); g = pos * m["o2c"].values; dp = np.abs(np.diff(pos, prepend=pos[0]))
xs = np.linspace(0, 6, 121); ys = [(g - dp * x / 10000).sum() * 100 for x in xs]
lo, hi = 0, 10
for _ in range(50):
    mid = (lo + hi) / 2
    if (g - dp * mid / 10000).sum() > 0: lo = mid
    else: hi = mid
be = lo
USED = 1.0
COMM = 0.18           # 온라인 수수료 (공시·수집)
SPREAD_LO, SPREAD_HI, SPREAD_MID = 0.5, 1.0, 0.75   # 스프레드(추정)

# 슬10 프레임 비율(1406.8:578.9 ≈ 2.43)에 맞춰 크롭 방지 + 양옆/상하 여백 확보.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.35))
fig.subplots_adjust(left=0.065, right=0.935, top=0.80, bottom=0.205, wspace=0.24)

# 좌: 비용 분해 (수평 누적막대) — 수집 가능(공시) vs 수집 불가(호가부재→추정) 명시
ax1.barh(0, COMM, color="#2e7d32", edgecolor="white",
         label=f"수수료(공시값 — 수집 가능) {COMM:.2f}bp")
ax1.barh(0, SPREAD_MID, left=COMM, color="#90caf9", edgecolor="white", hatch="///",
         label=f"스프레드(호가 부재 — 수집 불가, CS 추정) ~{SPREAD_LO:.1f}–{SPREAD_HI:.1f}bp")
ax1.errorbar(COMM + SPREAD_MID, 0, xerr=[[SPREAD_MID - SPREAD_LO], [SPREAD_HI - SPREAD_MID]],
             fmt="none", ecolor="#1565c0", capsize=5, lw=1.5)
ax1.axvline(USED, color="#d62728", ls="--", lw=2.5, label="백테스트 사용값 1.0bp")
ax1.set_xlim(0, 2.0); ax1.set_ylim(-1, 1.4); ax1.set_yticks([])
ax1.set_xlabel("편도 거래비용 (bp)")
ax1.set_title("① 비용 분해 — 수집 가능(수수료) vs 수집 불가(스프레드)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5, loc="upper right")
ax1.text(0.02, 0.05, "시장충격: 수집 불가 — 미반영  ·  CS 추정 중앙값 ~%.1fbp" % cs_med_pos,
         transform=ax1.transAxes, fontsize=8.5, color="#555")

# 우: 비용민감도
ax2.plot(xs, ys, color="#1f77b4", lw=2.2)
ax2.axhline(0, color="k", lw=0.8)
ax2.axvline(USED, color="#d62728", ls="--", lw=2)
ax2.axvline(be, color="#888", ls="-.", lw=1.5)
ax2.scatter([USED], [(g - dp * USED / 10000).sum() * 100], color="#d62728", zorder=5, s=45)
ax2.set_title("② 비용민감도 — 누적수익 vs 편도비용 (KOSEF 2023–25, 세전)", fontsize=11, fontweight="bold")
ax2.set_xlabel("편도 거래비용 (bp)"); ax2.set_ylabel("누적 수익률 (%)")
ax2.annotate("사용 1.0bp", (USED, (g - dp * USED / 10000).sum() * 100),
             textcoords="offset points", xytext=(6, -16), fontsize=9, color="#d62728")
ax2.annotate(f"손익분기 ~{be:.1f}bp", (be, 0), textcoords="offset points", xytext=(4, 10), fontsize=9, color="#555")

fig.suptitle("거래비용 가정: 수수료는 공시값으로 수집, 스프레드는 추정 · 사용 1bp · 손익분기 ~%.1fbp" % be,
             fontsize=12.5, fontweight="bold", y=0.955)
# 출처 — 프레임 안쪽(하단 여백)에 배치해 크롭 방지
fig.text(0.5, 0.055,
         "주: 수수료=증권사 공시(실제값) · 스프레드=장중 호가데이터 부재로 직접 관측 불가 → 고가-저가 Corwin–Schultz 추정"
         "(노이즈 큼: 저변동일 0·고변동일 과대 → 중앙값 채택) · 시장충격 미반영 · 세금 별도.",
         ha="center", fontsize=7.5, color="#666")
fig.text(0.5, 0.022,
         "출처: 채권형 ETF 매매차익 15.4%(Kodex 세금가이드) · 국채선물 양도세 11%·손익통산(삼일PwC) · Samsung Futures 계약스펙 · Corwin & Schultz(2012, J. Finance)",
         ha="center", fontsize=7.5, color="#666")
out = ROOT / "reports/figures/cost_conservatism.png"
fig.savefig(out, dpi=150)  # bbox_inches='tight' 미사용 → figsize 비율(2.43) 그대로 유지
print("saved", out, "| comm", COMM, "spread", f"{SPREAD_LO}-{SPREAD_HI}", "cs_med", round(cs_med_pos, 2), "breakeven", round(be, 2))

# -*- coding: utf-8 -*-
r"""
42 - 2026 라이브 OOS 시각화 (발표 블록6, HANDOFF §3.2 / final_presentation_plan §10.2)

동결 v3 모델(train<=2024)로 2026 93영업일 순수 포워드. 정직 결과:
방향 0.576(CI 0.5 포함=유의X), 진폭 DM p=0.003(naive 능가), 백테 1bp +5.2%(CI 0 포함)·2bp면 음수.

입력: reports/no_leak_v2/live_oos_2026_xgb.csv (date,y_true,q50,y10,gross,net,on_lo,on_hi)
출력: reports/figures/live_oos_2026.png
실행: .venv\Scripts\python.exe scripts/42_viz_live_oos_2026.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib, matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "no_leak_v2"
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

lv = pd.read_csv(REP / "live_oos_2026_xgb.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
d = lv["date"]; yt = lv["y_true"].values; q = lv["q50"].values

# 방향 정확도 (y_true!=0만)
nz = yt != 0
hit = (np.sign(q) == np.sign(yt))
acc = hit[nz].mean()
n = int(nz.sum())
# 부트스트랩 CI
rng = np.random.default_rng(0); bs = [hit[nz][rng.integers(0, n, n)].mean() for _ in range(5000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
# 롤링 정확도(20d)
roll = pd.Series(hit, index=d).rolling(20).mean()

# 누적 손익(csv net = 1bp 반영 백테스트)
cum_net = np.cumprod(1 + lv["net"].values) - 1
cum_gross = np.cumprod(1 + lv["gross"].values) - 1

# 온라인 conformal 구간 — 워밍업/sentinel 필터(lo<hi & |bound|<50bp)
lob, hib = lv["on_lo"].values, lv["on_hi"].values
valid = (lob < hib) & (np.abs(lob) < 50) & (np.abs(hib) < 50)
cov = ((yt >= lob) & (yt <= hib))[valid].mean() if valid.sum() else float("nan")

fig, ax = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle(f"2026 라이브 OOS — 동결 v3 모델 순수 포워드 (n={n}일, 2026-01~05)", fontsize=14, fontweight="bold")

# A: 롤링 방향 정확도
a = ax[0, 0]
a.plot(d, roll, color="#1f77b4", lw=1.8, label="20일 롤링 방향정확도")
a.axhline(acc, color="crimson", ls="-", lw=2, label=f"전체 {acc:.3f}")
a.axhline(0.5, color="k", ls="--", lw=1, label="무작위 0.5")
a.axhspan(lo, hi, color="crimson", alpha=0.10, label=f"95% CI [{lo:.3f},{hi:.3f}]")
a.set_ylim(0, 1); a.set_title("① 방향 정확도 (CI가 0.5 포함 → 유의성 미확보)"); a.legend(fontsize=8.5, loc="lower left"); a.grid(alpha=.3)

# B: 예측 vs 실현 (bp), 적중/실패 색
b = ax[0, 1]
b.axhline(0, color="k", lw=.7)
b.scatter(d[nz & hit], yt[nz & hit], s=22, c="#2ca02c", label="방향 적중", zorder=3)
b.scatter(d[nz & ~hit], yt[nz & ~hit], s=22, c="#d62728", label="방향 실패", zorder=3)
b.plot(d, q, color="#1f77b4", lw=1.2, alpha=.8, label="예측 q50")
b.set_ylabel("일별 Δy (bp)"); b.set_title("② 예측(q50) vs 실현(Δy) — 적중/실패"); b.legend(fontsize=8.5); b.grid(alpha=.3)

# C: 온라인 conformal 구간 커버리지
c = ax[1, 0]
c.axhline(0, color="k", lw=.7)
dv = d[valid]
c.fill_between(dv, lob[valid], hib[valid], color="#1f77b4", alpha=.18, label="온라인 90% 구간")
inb = valid & (yt >= lob) & (yt <= hib)
c.scatter(d[inb], yt[inb], s=14, c="#2ca02c", label="구간 내", zorder=3)
c.scatter(d[valid & ~((yt >= lob) & (yt <= hib))], yt[valid & ~((yt >= lob) & (yt <= hib))], s=18, c="#d62728", label="구간 밖", zorder=3)
c.set_ylabel("Δy (bp)"); c.set_title(f"③ 온라인 conformal 구간 (coverage {cov:.2f}, 목표 0.90)"); c.legend(fontsize=8.5); c.grid(alpha=.3)

# D: 누적 손익
e = ax[1, 1]
e.axhline(0, color="k", lw=.7)
e.plot(d, cum_gross * 100, color="#7fb3d5", ls="--", lw=1.6, label=f"비용 0 (gross) {cum_gross[-1]*100:+.1f}%")
e.plot(d, cum_net * 100, color="#1f77b4", lw=2.2, label=f"1bp 반영 (net) {cum_net[-1]*100:+.1f}%")
e.set_ylabel("누적 수익 (%)"); e.set_title("④ 비용반영 누적손익 (2bp면 음수 — 비용 민감)"); e.legend(fontsize=8.5); e.grid(alpha=.3)

fig.tight_layout(rect=[0, 0, 1, 0.97])
out = FIG / "live_oos_2026.png"
fig.savefig(out, dpi=130)
print(f"dir_acc={acc:.4f} n={n} CI=[{lo:.3f},{hi:.3f}] coverage={cov:.3f} valid_band={int(valid.sum())}")
print(f"cum_net={cum_net[-1]*100:+.2f}% cum_gross={cum_gross[-1]*100:+.2f}%")
print(f"[saved] {out}")

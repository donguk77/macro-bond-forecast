# -*- coding: utf-8 -*-
r"""
41 - 회전율 축소 테스트 (HANDOFF 2026-05-30 §3 최우선 과제)

신호가 강할 때만(|q50|>tau) 매매해 회전율을 줄이면, 현실 체결(시초가)+수수료(1bp)
+채권ETF 매도건별 세금(15.4%, 손실상계X) 후에도 B&H를 (유의하게) 이기는가?

중요: 현재 데이터의 q50 는 이미 basis point 단위(y_true 와 동일). 임계값은 |q50|>tau(bp) 직접 비교.
방법론은 demo_app/scripts40 과 동일. 유의성=블록부트스트랩(전략-B&H 최종수익차 95% CI).

실행: .venv\Scripts\python.exe scripts/41_turnover_reduction.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "no_leak_v2"
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

def tax_per_day(pos, o2c, rate=0.154):
    pos = np.asarray(pos, float); o2c = np.asarray(o2c, float)
    n = len(pos); tax = np.zeros(n); seg = 1.0
    for t in range(n):
        if pos[t] != 0: seg *= (1.0 + pos[t] * o2c[t])
        nxt = pos[t + 1] if t + 1 < n else 0.0
        if pos[t] != 0 and nxt != pos[t]:
            r = seg - 1.0
            if r > 0: tax[t] += r * rate
            seg = 1.0
    return tax

def load_merged():
    iv = pd.read_csv(REP / "predictions_xgb_v3_intervals.csv", parse_dates=["date"])
    f3 = iv[iv["fold"] == "fold3"][["date", "q50"]]
    sig = f3.drop_duplicates("date").sort_values("date")   # 2023-2025 (라이브 2026 제외 — 슬12에서만)
    etf = pd.read_csv(ROOT / "data/raw/kosef_10y_daily_2023_2026.csv", parse_dates=["date"]).sort_values("date")
    df = sig.merge(etf[["date", "open", "close"]], on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["o2c"] = df["close"] / df["open"] - 1.0
    return df

def net_series(df, tau, cost_bp=1.0, apply_tax=True):
    q = df["q50"].values
    pos = np.where(np.abs(q) > tau, -np.sign(q), 0.0)   # q50 는 이미 bp 단위
    ret = df["o2c"].values
    prev = np.concatenate([[0.0], pos[:-1]])
    trade = np.abs(pos - prev)
    tax = tax_per_day(pos, ret) if apply_tax else np.zeros(len(pos))
    net = pos * ret - trade * cost_bp / 1e4 - tax
    return net, int(np.sum(pos != prev)), int(np.sum(pos != 0))

def boot_diff(strat_daily, bh_daily, block=10, n=4000, seed=0):
    rng = np.random.default_rng(seed); T = len(strat_daily); nb = int(np.ceil(T / block))
    d = np.empty(n)
    for i in range(n):
        s = rng.integers(0, T, size=nb)
        idx = np.concatenate([np.arange(x, x + block) % T for x in s])[:T]
        d[i] = (np.prod(1 + strat_daily[idx]) - 1) - (np.prod(1 + bh_daily[idx]) - 1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())

def main():
    df = load_merged()
    bh_curve = df["close"] / df["close"].iloc[0] - 1.0
    bh = float(bh_curve.iloc[-1])
    bh_daily = df["close"].pct_change().fillna(0.0).values
    absq = np.abs(df["q50"].values)
    fin = lambda a: float(np.cumprod(1 + a)[-1] - 1)
    taus = [0.0] + [float(np.percentile(absq, p)) for p in [25, 50, 60, 70, 80, 90]]

    print(f"N={len(df)}  {df.date.min().date()}..{df.date.max().date()}  B&H={bh*100:+.2f}%")
    print("|q50| 분위수(bp):", {f"p{p}": round(float(np.percentile(absq, p)), 3) for p in [25,50,60,70,80,90]})
    print(f"\n{'tau(bp)':>8}{'trades':>8}{'days_in':>8}{'no-cost':>9}{'+1bp':>8}{'+1bp+tax':>10}{'beat':>6}")
    rows = []
    for t in taus:
        n0, _, _ = net_series(df, t, 0.0, False)
        n1, ntr, di = net_series(df, t, 1.0, False)
        n2, _, _ = net_series(df, t, 1.0, True)
        r = dict(tau_bp=t, n_trades=ntr, days_in_market=di, ret_nocost=fin(n0), ret_1bp=fin(n1), ret_1bp_tax=fin(n2))
        rows.append(r)
        print(f"{t:>8.3f}{ntr:>8d}{di:>8d}{r['ret_nocost']*100:>8.1f}%{r['ret_1bp']*100:>7.1f}%"
              f"{r['ret_1bp_tax']*100:>9.1f}%{('Y' if r['ret_1bp_tax']>bh else 'n'):>6}")
    out = pd.DataFrame(rows); out["buy_hold"] = bh
    out.to_csv(REP / "turnover_reduction_summary.csv", index=False)

    print("\n[블록부트스트랩 95% CI: 전략(+1bp+세금) - B&H 최종수익차]  CI 0 미포함=유의")
    boot = {}
    for t in taus:
        sd, _, _ = net_series(df, t, 1.0, True)
        lo, hi, p = boot_diff(sd, bh_daily)
        boot[f"{t:.3f}"] = dict(lo=lo, hi=hi, p=p, sig=("SIG" if (lo > 0 or hi < 0) else "ns"))
        print(f"  tau={t:>6.3f}: [{lo*100:+.1f}%, {hi*100:+.1f}%]  P(beat)={p:.2f}  {boot[f'{t:.3f}']['sig']}")
    (REP / "turnover_bootstrap.json").write_text(json.dumps(boot, indent=1), encoding="utf-8")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1b = ax1.twinx(); ta = [r["tau_bp"] for r in rows]
    ax1.plot(ta, [r["ret_1bp_tax"]*100 for r in rows], "o-", color="#1f77b4", label="전략 +1bp+세금(매도건별)")
    ax1.plot(ta, [r["ret_1bp"]*100 for r in rows], "s--", color="#7fb3d5", alpha=.8, label="전략 +1bp(세금前)")
    ax1.axhline(bh*100, color="k", ls="--", lw=2, label=f"Buy&Hold ({bh*100:+.1f}%)")
    ax1b.plot(ta, [r["days_in_market"] for r in rows], "^:", color="gray", alpha=.6, label="시장 보유일수")
    ax1.set_xlabel("신호 임계값 tau (|q50|, yield bp)"); ax1.set_ylabel("누적수익 (%)"); ax1b.set_ylabel("시장 보유일수")
    ax1.set_title("회전율 축소: tau별 현실 순수익 vs Buy&Hold")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper left", fontsize=9); ax1.grid(alpha=.3)
    for r in rows:
        if r["tau_bp"] in (taus[0], taus[2], taus[3], taus[5]):
            sd, _, _ = net_series(df, r["tau_bp"], 1.0, True)
            ax2.plot(df["date"], (np.cumprod(1+sd)-1)*100, label=f"tau={r['tau_bp']:.2f} ({r['days_in_market']}일보유)")
    ax2.plot(df["date"], bh_curve*100, "k--", lw=2, label="Buy & Hold")
    ax2.set_ylabel("누적수익 (%) [시초가+1bp+세금]"); ax2.set_title("현실 순수익 곡선")
    ax2.legend(fontsize=9); ax2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "turnover_reduction.png", dpi=130)
    print(f"\n[saved] {REP/'turnover_reduction_summary.csv'}\n[saved] {FIG/'turnover_reduction.png'}")

if __name__ == "__main__":
    main()

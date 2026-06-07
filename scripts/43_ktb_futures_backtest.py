"""
[43] KTB 10년 국채선물 백테스트 (KODEX 152380 프록시)
demo_app(KOSEF) 동일 방법론을 선물추종 ETF 152380에 적용.
- 신호: pos = -sign(q50)  (금리하락 예측=롱)  · 시초가 체결(o2c, 장중만, 누수안전)
- 비용: |Δpos| * cost_bp/10000  (편도 bp)
- 세금: 선물=양도세 11% 연간 손익통산 / 현물ETF=배당소득세 15.4% 매도건별
- 블록 부트스트랩 95% CI (block=10, n=4000)
정직 caveat: 152380은 선물 보유 ETF(운용보수 드래그), 세금 11%는 선물 직접거래 가정.
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "reports/no_leak_v2/predictions_xgb_v3_intervals.csv"
FUT  = ROOT / "data/raw/kodex_10y_ktb_futures_etf_152380.csv"
np.random.seed(42)

# --- 데이터 병합 ---
pred = pd.read_csv(PRED, parse_dates=["date"])[["date", "q50", "y_true"]]
fut  = pd.read_csv(FUT)
fut.columns = [str(c).lower() for c in fut.columns]
fut["date"] = pd.to_datetime(fut["date"])
fut["o2c"] = fut["close"] / fut["open"] - 1.0          # 시초가→종가 (현실, 장중)
fut["c2c"] = fut["close"] / fut["close"].shift(1) - 1.0 # 종가→종가 (B&H용)
df = pred.merge(fut[["date", "open", "close", "o2c", "c2c"]], on="date", how="inner").dropna().reset_index(drop=True)
print(f"병합: {len(df)}일  {df['date'].min().date()} ~ {df['date'].max().date()}")

pos = -np.sign(df["q50"].values)
ret_o2c = df["o2c"].values
bh_ret = df["c2c"].values

def strat_pnl(cost_bp):
    gross = pos * ret_o2c
    dpos = np.abs(np.diff(pos, prepend=pos[0]))
    return gross - dpos * cost_bp / 10000.0

def segments(p):
    p = np.asarray(p); seg = []; i = 0; n = len(p)
    while i < n:
        if p[i] == 0: i += 1; continue
        j = i
        while j + 1 < n and p[j+1] == p[i]: j += 1
        seg.append((i, j)); i = j + 1
    return seg

def tax_persale(p, pnl, rate):  # 매도 건별: 포지션 종료 구간별 양(+)이익에만 과세
    t = np.zeros(len(pnl))
    for s, e in segments(p):
        ssum = pnl[s:e+1].sum()
        if ssum > 0: t[e] = ssum * rate
    return t

def tax_annual(dates, pnl, rate):  # 연간 손익통산: 해 단위 순이익 양수면 과세
    t = np.zeros(len(pnl)); yrs = pd.to_datetime(dates).year
    for y in np.unique(yrs):
        m = yrs == y; ysum = pnl[m].sum()
        if ysum > 0: t[np.where(m)[0][-1]] = ysum * rate
    return t

def sharpe(p):
    return float(p.mean()/p.std()*np.sqrt(252)) if p.std() > 0 else float("nan")

def block_boot_ci(x, n=4000, block=10):
    x = np.asarray(x); N = len(x); nb = int(np.ceil(N/block)); tot = []
    for _ in range(n):
        idx = (np.random.randint(0, N-block+1, nb)[:, None] + np.arange(block)).ravel()[:N]
        tot.append(x[idx].sum())
    return float(np.quantile(tot, .025))*100, float(np.quantile(tot, .975))*100

def pbeat(strat_net, bh_net, n=4000, block=10):
    N = len(strat_net); nb = int(np.ceil(N/block)); c = 0
    for _ in range(n):
        idx = (np.random.randint(0, N-block+1, nb)[:, None] + np.arange(block)).ravel()[:N]
        if strat_net[idx].sum() > bh_net[idx].sum(): c += 1
    return c/n

dir_acc = float((np.sign(df["q50"]) == np.sign(df["y_true"])).mean())
days_in = int((pos != 0).sum()); turn = int(np.sum(np.abs(np.diff(pos, prepend=pos[0])) > 0))
bh_total = bh_ret.sum()*100
print(f"방향정확도(부호일치) {dir_acc:.3f} | 보유 {days_in}/{len(df)}일 | 포지션변경 {turn}회 | B&H(152380) {bh_total:+.2f}%\n")

print(f"{'시나리오':<34}{'순수익%':>9}{'Sharpe':>8}{'95%CI':>20}{'P(beat B&H)':>12}")
print("-"*85)
rows = []
for cost in [0.0, 0.5, 1.0, 2.0]:
    pnl = strat_pnl(cost)
    # (A) 세금 OFF
    net = pnl
    ci = block_boot_ci(net - bh_ret)
    rows.append((f"비용 {cost:.1f}bp · 세금OFF", net.sum()*100, sharpe(net), ci, pbeat(net, bh_ret)))
    # (B) 선물 세금 11% 연간통산
    tf = tax_annual(df["date"].values, pnl, 0.11); net_f = pnl - tf
    bh_tf = tax_annual(df["date"].values, bh_ret, 0.11); bh_nf = bh_ret - bh_tf
    ci = block_boot_ci(net_f - bh_nf)
    rows.append((f"비용 {cost:.1f}bp · 선물세금11%(연간통산)", net_f.sum()*100, sharpe(net_f), ci, pbeat(net_f, bh_nf)))
    # (C) 현물ETF 세금 15.4% 매도건별
    te = tax_persale(pos, pnl, 0.154); net_e = pnl - te
    bh_te = tax_persale(np.ones(len(bh_ret)), bh_ret, 0.154); bh_ne = bh_ret - bh_te
    ci = block_boot_ci(net_e - bh_ne)
    rows.append((f"비용 {cost:.1f}bp · 현물ETF세금15.4%(매도건별)", net_e.sum()*100, sharpe(net_e), ci, pbeat(net_e, bh_ne)))

for name, tot, shp, ci, pb in rows:
    print(f"{name:<34}{tot:>8.1f}%{shp:>8.2f}   [{ci[0]:>6.1f}, {ci[1]:>6.1f}]%{pb:>11.2f}")

print("\n주: 152380=KODEX 10년국채선물 ETF(선물 추종). 선물세금=직접거래 가정 프록시.")
print("    순수익은 명목(단리 합산) 기준. CI는 전략-B&H 초과수익 블록부트스트랩.")

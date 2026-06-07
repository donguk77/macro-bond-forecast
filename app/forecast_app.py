# -*- coding: utf-8 -*-
r"""
한국 국채 10Y — 예측·검증 통합 대시보드 (사용자용)
================================================================
탭① 오늘의 예측 — 방향 신호 + 불확실성 구간(online conformal) + 간밤 글로벌 동인.
탭② 비용·세금 검증 — 실거래 KOSEF ETF 백테스트로 체결현실·비용·세금까지 정직하게.
타깃: 채권 데스크 / 트레저리·리스크. 데이터: live_oos_2026 + predictions_xgb_v3_intervals + KOSEF ETF.
실행: .venv\Scripts\streamlit run app/forecast_app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "no_leak_v2"
LIVE = REP / "live_oos_2026_xgb.csv"
IV = REP / "predictions_xgb_v3_intervals.csv"
ETF = ROOT / "data" / "raw" / "kosef_10y_daily_2023_2026.csv"
MOD_DUR = 8.0          # 10Y 국고채 근사 수정듀레이션 (가격 VaR 환산)
TAX_RATE = 0.154       # 채권형 ETF 매매차익 배당소득세 15.4%
LIVE_START = pd.Timestamp("2026-01-01")

st.set_page_config(page_title="국채 10Y 예측·검증", page_icon="🇰🇷", layout="wide")

st.markdown('''<style>
.block-container{padding-top:1.2rem;max-width:1180px;}
[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e4e9f2;border-radius:12px;padding:10px 14px;}
[data-testid="stMetricValue"]{font-size:1.4rem;font-weight:700;color:#16243d;}
[data-testid="stMetricLabel"] p{color:#5a6a85;font-weight:600;}
section.main h3,section.main h4,section.main h5{color:#16243d;font-weight:800;}
[data-testid="stVerticalBlockBorderWrapper"]{box-shadow:0 2px 8px rgba(20,40,80,.06);border-radius:14px;}
button[data-baseweb="tab"]{font-size:1.05rem;font-weight:700;}
thead th{background:#16243d;color:#fff !important;}
</style>''', unsafe_allow_html=True)


# ============================ 데이터 ============================
@st.cache_data
def load_live():
    return pd.read_csv(LIVE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data
def load_combined():
    """2023–25 test(fold3) + 2026 live + 실제 KOSEF ETF OHLC 결합 (검증 탭용)."""
    iv = pd.read_csv(IV, parse_dates=["date"])
    f3 = iv[iv["fold"] == "fold3"][["date", "q50", "y_true"]].copy(); f3["period"] = "test"
    lv = pd.read_csv(LIVE, parse_dates=["date"])[["date", "q50", "y_true"]].copy(); lv["period"] = "live"
    df = pd.concat([f3, lv], ignore_index=True).sort_values("date").reset_index(drop=True)
    etf = pd.read_csv(ETF, parse_dates=["date"]).sort_values("date")
    etf["etf_c2c"] = etf["close"] / etf["close"].shift(1) - 1
    etf["etf_o2c"] = etf["close"] / etf["open"] - 1
    return df.merge(etf[["date", "etf_c2c", "etf_o2c"]], on="date", how="inner").reset_index(drop=True)


# ============================ 백테 (검증 탭) ============================
def backtest(df, exec_mode="open", cost_bp=1.0):
    pos = -np.sign(df["q50"].values)
    ret = df["etf_c2c"].values if exec_mode == "close" else df["etf_o2c"].values
    dpos = np.abs(np.diff(pos, prepend=pos[0]))
    return pos * ret - dpos * cost_bp / 10000


def buyhold(df):
    return df["etf_c2c"].values


def breakeven_bp(df):
    pos = -np.sign(df["q50"].values); g = pos * df["etf_o2c"].values
    dp = np.abs(np.diff(pos, prepend=pos[0])).sum()
    return float(g.sum() / dp * 10000) if dp > 0 else float("nan")


def _segments(pos):
    pos = np.asarray(pos); seg = []; i = 0; n = len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1; continue
        j = i
        while j + 1 < n and pos[j + 1] == pos[i]:
            j += 1
        seg.append((i, j)); i = j + 1
    return seg


def tax_per_day(pos, pnl, rate=TAX_RATE):
    t = np.zeros(len(pnl))
    for s, e in _segments(pos):
        r = float(np.sum(pnl[s:e + 1]))
        if r > 0:
            t[e] = r * rate
    return t


# ============================ 예측 탭 헬퍼 ============================
def gauge(lo, hi, mid, color):
    pad = max(1.5, (hi - lo) * 0.3)
    fig = go.Figure()
    fig.add_shape(type="line", x0=lo, x1=hi, y0=0, y1=0, line=dict(color=color, width=16), opacity=0.85)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="markers+text", marker=dict(size=9, color=color),
                             text=[f"{lo:+.1f}", f"{hi:+.1f}"], textposition="bottom center",
                             textfont=dict(size=12, color="#5a6a85"), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[mid], y=[0], mode="markers+text",
                             marker=dict(size=24, color="white", line=dict(color=color, width=4)),
                             text=[f"{mid:+.1f}"], textposition="top center",
                             textfont=dict(size=14, color=color, family="Arial Black"), showlegend=False))
    fig.add_vline(x=0, line=dict(color="#aaa", width=1.2, dash="dash"))
    fig.update_layout(height=140, margin=dict(t=26, b=24, l=14, r=14),
                      xaxis=dict(range=[min(lo, 0) - pad, max(hi, 0) + pad], zeroline=False, showgrid=False, tickfont=dict(size=11)),
                      yaxis=dict(visible=False, range=[-1, 1]), plot_bgcolor="white", paper_bgcolor="white")
    return fig


def driver_panel():
    DRIVERS = [
        ("미국 10Y (%)", "us_treasury_10y", "▲ → 한국 금리 ↑ 압력　★모델 최대 동인", True),
        ("미국 2Y (%)", "us_treasury_2y", "연준 정책금리 기대 반영", False),
        ("VIX 변동성", "vix", "▲ → 안전자산 선호 → 금리 ↓", False),
        ("달러 DXY", "dxy", "▲ → EM 자본유출 → 금리 ↑ 압력", False),
        ("S&P 500", "sp500", "▲ → 위험선호 → 금리 ↑ 경향", False),
        ("WTI 유가 ($)", "wti_oil", "▲ → 인플레 기대 → 금리 ↑", False),
    ]
    src = None
    for f in [ROOT / "data/interim/wide_daily_filled.csv", ROOT / "data/processed/features_v3_candidate.csv"]:
        if f.exists():
            src = pd.read_csv(f)
            if "date" in src.columns:
                src["date"] = pd.to_datetime(src["date"]); src = src.sort_values("date")
            break
    if src is None:
        st.caption("글로벌 동인 데이터 미연결."); return
    found = []
    for label, col, eff, star in DRIVERS:
        if col in src.columns:
            s = src[col].dropna()
            if len(s) >= 2:
                found.append((label, col, float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-2]), eff, star))
    if not found:
        st.caption("글로벌 동인 컬럼 없음."); return
    key = [r for r in found if r[1] in ("us_treasury_10y", "vix", "dxy")]
    if key:
        cols = st.columns(len(key))
        for c, (label, col, val, chg, eff, star) in zip(cols, key):
            c.metric(label, f"{val:,.2f}", f"{chg:+.2f}")
    tdf = pd.DataFrame([{"지표": ("★ " + label) if star else label, "최신값": f"{val:,.2f}",
                         "전일 Δ": f"{chg:+.2f}", "한국 금리 영향 (도메인)": eff}
                        for label, col, val, chg, eff, star in found])
    st.table(tdf)
    us = [r for r in found if r[1] == "us_treasury_10y"]
    if us:
        d = us[0][3]; msg = "상승 압력 ▲" if d > 0 else ("하락 압력 ▼" if d < 0 else "중립 —")
        st.caption(f"💡 간밤 **미국 10Y가 {d:+.2f}%p** 변화 — 우리 모델이 가장 크게 반영하는 동인으로, "
                   f"오늘 한국 금리엔 **{msg}**. 모델은 이 US→KR 오버나잇 스필오버를 학습했습니다.")


# ============================ 탭 ① 오늘의 예측 ============================
def tab_forecast():
    lv = load_live(); last = lv.iloc[-1]
    q50 = float(last["q50"]); lo = float(last["on_lo"]); hi = float(last["on_hi"])
    base_date = last["date"].date(); level = float(last.get("y10", np.nan))

    nz = lv["y_true"] != 0
    hit = (np.sign(lv["q50"]) == np.sign(lv["y_true"]))
    live_acc = hit[nz].mean()
    rng = np.random.default_rng(0); n = int(nz.sum())
    bs = [hit[nz].values[rng.integers(0, n, n)].mean() for _ in range(2000)]
    acc_lo, acc_hi = np.percentile(bs, [2.5, 97.5])
    valid = (lv["on_lo"] < lv["on_hi"]) & (lv["on_lo"].abs() < 50) & (lv["on_hi"].abs() < 50)
    live_cov = ((lv["y_true"] >= lv["on_lo"]) & (lv["y_true"] <= lv["on_hi"]))[valid].mean()
    cum_net = float(np.cumprod(1 + lv["net"].values)[-1] - 1)

    up = q50 > 0
    c_main = "#c0392b" if up else "#1f6fb2"; c_grad = "#e35d4c" if up else "#3b8fd0"
    arrow = "▲" if up else "▼"; dir_text = "금리 상승" if up else "금리 하락"
    lev_txt = f"　·　현재 10Y <b>{level:.3f}%</b>" if not np.isnan(level) else ""

    st.markdown(f'''
    <div style="background:linear-gradient(135deg,{c_main},{c_grad});border-radius:18px;
                padding:24px 32px;color:#fff;box-shadow:0 4px 16px rgba(20,40,80,.20);">
      <div style="font-size:.9rem;opacity:.9;">기준일 {base_date} · 모델 v3 <b>동결</b>(학습 ≤2025){lev_txt}</div>
      <div style="font-size:2.5rem;font-weight:800;line-height:1.15;margin:.2rem 0;">{arrow} 내일 {dir_text} 예상</div>
      <div style="font-size:1.12rem;opacity:.96;">예측 Δy <b>{q50:+.2f} bp</b>　·　90% 구간 [{lo:+.1f}, {hi:+.1f}] bp　·　폭 {hi - lo:.1f} bp</div>
    </div>''', unsafe_allow_html=True)
    st.write("")

    unit = st.radio("구간 단위", ["금리 bp", "가격 % (듀레이션 환산)"], horizontal=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### ① 방향 · 거래 신호")
            st.markdown(f'<div style="font-size:2rem;font-weight:800;color:{c_main};">{arrow} {dir_text}</div>'
                        f'<div style="color:#5a6a85;">예측 Δy {q50:+.2f} bp · 신뢰도 '
                        f'{"강" if abs(q50) > 1 else ("중" if abs(q50) > 0.4 else "약")}</div>', unsafe_allow_html=True)
            pos = "숏 · 듀레이션 축소" if up else "롱 · 듀레이션 확대"
            st.markdown(f'**포지션 함의:** 채권 가격 {"하락" if up else "상승"} → **{pos}**')
            st.markdown(f'전략 라이브 누적손익(1bp): **{cum_net * 100:+.1f}%**')
            st.line_chart(pd.DataFrame({"누적손익(net, %)": (np.cumprod(1 + lv["net"].values) - 1) * 100},
                                       index=lv["date"]), height=200)
            st.warning("⚠️ 추세장 한정 · 라이브 단독 방향 유의성 없음(CI 0.5 포함) · 비용 2bp면 초과수익 상쇄")
    with c2:
        with st.container(border=True):
            st.markdown("#### ② 불확실성 구간 (90%)")
            if unit.startswith("금리"):
                st.plotly_chart(gauge(lo, hi, q50, c_main), width="stretch")
                st.markdown(f'내일 Δy **[{lo:+.1f}, {hi:+.1f}] bp**　·　중앙값 {q50:+.2f}　·　폭 {hi - lo:.1f} bp')
            else:
                p_lo = -hi * MOD_DUR / 100; p_hi = -lo * MOD_DUR / 100
                st.plotly_chart(gauge(p_lo, p_hi, -q50 * MOD_DUR / 100, c_main), width="stretch")
                st.markdown(f'가격변동 **[{p_lo:+.2f}, {p_hi:+.2f}] %** (D≈{MOD_DUR:.0f})　·　**VaR(90% 하단) {min(p_lo, p_hi):+.2f}%**')
            st.warning("⚠️ XGBoost 외삽 한계 → 위기 첫날 구간 과소 가능 · 커버리지 라이브 0.84(목표 0.90)")

    st.write("")
    with st.container(border=True):
        st.markdown("#### ③ 성능 기록 · 트랙 레코드")
        perf = pd.DataFrame({
            "지표": ["방향 정확도", "거래 1bp Sharpe", "구간 커버리지"],
            "In-sample (2023–25 test)": ["0.62 (DM p≈0, 유의)", "1.46 (CI [0.56, 2.40], 유의)", "0.905"],
            "라이브 OOS (2026)": [f"{live_acc:.3f} (CI [{acc_lo:.3f}, {acc_hi:.3f}], 유의X)",
                                  "1.96 (CI [-1.33, 5.90], 0 포함)", f"{live_cov:.3f} (위기 미달)"],
        })
        st.table(perf)
        st.caption("본 도구는 **금리 방향 전망·리스크 범위 추정용**이며 매매 수익을 보장하지 않습니다. "
                   "거래비용·세금 영향은 [비용·세금 검증] 탭에서 직접 확인하세요.")

    st.write("")
    with st.container(border=True):
        st.markdown("#### ④ 간밤 글로벌 동인 (overnight drivers)")
        driver_panel()


# ============================ 탭 ② 비용·세금 검증 ============================
def tab_validation():
    df = load_combined()
    st.markdown("실제 **KOSEF 국고채10년 ETF** 체결가로 백테스트 — 체결 시점·거래비용·세금까지 넣어 '예측이 실제 수익이 되는지' 검증합니다.")
    cc1, cc2, cc3 = st.columns([1.5, 1, 1])
    period = cc1.radio("기간", ["2023–25 test (발표 기준)", "전체 (2023–26)", "2026 라이브만"])
    execm = cc2.radio("체결 시점", ["시초가 (현실)", "종가 (이론)"],
                      help="신호는 종가에 산출 → 현실은 다음날 시초가 체결. 종가체결은 못 잡는 밤사이 갭까지 포함하는 낙관 가정.")
    cost = cc3.slider("거래비용 bp(편도)", 0.0, 10.0, 1.0, 0.5)
    tax = st.checkbox("세금 15.4% 반영 (채권ETF 매도 건별·손실 상계X)", value=False)

    if period.startswith("2023"):
        win = df[df["date"] <= pd.Timestamp("2025-12-31")]
    elif period.startswith("2026"):
        win = df[df["date"] >= LIVE_START]
    else:
        win = df
    win = win.reset_index(drop=True)
    if len(win) < 10:
        st.warning("선택 구간 데이터가 부족합니다."); return

    em = "open" if execm.startswith("시초가") else "close"
    pos = -np.sign(win["q50"].values)
    pnl = backtest(win, em, cost); bh = buyhold(win)
    tax_s = tax_per_day(pos, pnl) if tax else np.zeros(len(pnl))
    tax_b = tax_per_day(np.ones(len(bh)), bh) if tax else np.zeros(len(bh))
    cum = np.cumsum(pnl - tax_s) * 100; cum_bh = np.cumsum(bh - tax_b) * 100
    be = breakeven_bp(win)
    valid = win["y_true"] != 0
    dacc = float((np.sign(win["q50"][valid]) == np.sign(win["y_true"][valid])).mean()) if valid.any() else float("nan")

    st.markdown(f"##### 성과 — {win['date'].min().date()} ~ {win['date'].max().date()} · {len(win)}영업일 · **{execm.split()[0]}체결**")
    k = st.columns(4)
    k[0].metric("방향 정확도", f"{dacc:.3f}", f"{(dacc - 0.5) * 100:+.1f}%p vs 0.5")
    k[1].metric(f"전략 누적 ({cost:.1f}bp{'+세금' if tax else ''})", f"{cum[-1]:+.1f}%", f"{cum[-1] - cum_bh[-1]:+.1f}%p vs B&H")
    k[2].metric("Buy & Hold", f"{cum_bh[-1]:+.1f}%", "벤치마크", delta_color="off")
    k[3].metric("손익분기 (편도)", f"{be:.1f} bp", "이 위면 손실", delta_color="off")

    c1, c2 = st.columns([1.4, 1], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("###### 누적 수익률 — 전략 vs Buy & Hold")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=win["date"], y=cum, name="모델 전략", line=dict(color="crimson", width=2.3)))
            fig.add_trace(go.Scatter(x=win["date"], y=cum_bh, name="Buy & Hold", line=dict(color="black", width=1.2, dash="dash")))
            fig.add_hline(y=0, line=dict(color="black", width=0.5))
            fig.update_layout(height=300, margin=dict(t=20, b=20, l=40, r=10), yaxis_title="누적 %",
                              legend=dict(orientation="h", y=1.15, x=0.3))
            st.plotly_chart(fig, width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("###### 체결 시점 분해 (비용 0)")
            e_c2c = float(np.sum(backtest(win, "close", 0.0))) * 100
            e_o2c = float(np.sum(backtest(win, "open", 0.0))) * 100
            fe = go.Figure(go.Bar(x=["종가체결<br>(이론)", "시초가체결<br>(현실)", "밤사이 갭<br>(못잡음)"],
                                  y=[e_c2c, e_o2c, e_c2c - e_o2c], marker_color=["#b0b0b0", "#c0392b", "#e08a3c"],
                                  text=[f"{e_c2c:+.0f}%", f"{e_o2c:+.0f}%", f"{e_c2c - e_o2c:+.0f}%"], textposition="outside"))
            fe.add_hline(y=0, line=dict(color="black", width=0.5))
            fe.update_layout(height=300, margin=dict(t=20, b=10, l=40, r=10), yaxis_title="누적 %")
            st.plotly_chart(fe, width="stretch")

    gap_pct = (e_c2c - e_o2c) / e_c2c * 100 if e_c2c != 0 else float("nan")
    st.info(f"💡 **정직한 결론:** 종가체결(이론) {e_c2c:+.0f}% 중 **{gap_pct:.0f}%가 밤사이 갭**(개장 전 반영 → 현실 못 먹음). "
            f"현실 시초가 체결은 {e_o2c:+.0f}%이고 **손익분기 편도 {be:.1f}bp**로 마진이 얇습니다. "
            f"여기에 **채권ETF 매도건별 세금 15.4%**까지 넣으면 회전 많은 전략은 Buy&Hold 우위를 잃습니다 "
            f"(위 세금 토글로 직접 확인). → 저비용·비과세 수단(KTB 선물·직접매매)이나 회전율 축소가 향후 과제.")


# ============================ main ============================
st.title("🇰🇷 한국 국채 10Y — 예측 · 검증 대시보드")
st.caption("동결 v3 XGBoost 분위수 회귀 · 거시변수 13개 · 익일 Δy(bp) 예측 → 실거래 KOSEF ETF 검증")
tab1, tab2 = st.tabs(["📈 오늘의 예측", "🔍 비용·세금 검증"])
with tab1:
    tab_forecast()
with tab2:
    tab_validation()

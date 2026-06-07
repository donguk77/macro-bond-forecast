# -*- coding: utf-8 -*-
"""
국고채 10Y 금리 예측 — 통합 데모 앱 (기말 최종, 실거래 ETF 기반)
================================================================
2023–2025 test + 2026 라이브 OOS 를 하나의 연속 시계열로 합치고,
**실제 KOSEF 국고채10년 ETF 가격**으로 백테스트(yield→duration 근사 폐기).

정직성 핵심:
- 백테스트를 **실제 ETF 체결가**로 수행 → carry·convexity·운용보수는 ETF NAV 에 이미 반영(중복 제거).
- **체결 시점 토글**: 종가체결(이론, 신호 난 종가에 바로 체결 가정 — 비현실) vs 시초가체결(현실 next-open).
  엣지 대부분이 **overnight 갭**에 있어 종가체결은 과대평가 → 기본값은 시초가(현실).
- **2023–25 = test**(통계 증명, DM 유의) / **2026 = 진짜 라이브 OOS**(실용 확인) 를 화면에서 구분.
- 거래비용(슬라이더, 기본 1bp 틱수준)·세금 15.4%(매도 건별/연간 순이익) 모두 노출. 게이팅/앙상블은 v3 에 불필요(scripts/39).

실행: streamlit run app/demo_app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / 'reports' / 'no_leak_v2'
LIVE = REP / 'live_oos_2026_xgb.csv'
IV = REP / 'predictions_xgb_v3_intervals.csv'
ECOS = ROOT / 'data' / 'raw' / 'raw_ecos.csv'
ETF = ROOT / 'data' / 'raw' / 'kosef_10y_daily_2023_2026.csv'   # 실거래 KOSEF 국고채10년 ETF
COST = REP / 'daily_cost_estimate.csv'                          # Corwin-Schultz 날짜별 스프레드

LIVE_START = pd.Timestamp('2026-01-01')
TAX_RATE = 0.154   # 채권형 ETF 매매차익 배당소득세 15.4%

st.set_page_config(page_title='국고채 10Y 금리 예측 데모', page_icon='📈', layout='wide')

st.markdown('''<style>
[data-testid="stMetric"]{background:#f7f9fc;border:1px solid #e4e9f2;border-radius:12px;padding:12px 16px;box-shadow:0 1px 3px rgba(20,40,80,.05);}
[data-testid="stMetricValue"]{font-size:1.5rem;font-weight:700;color:#16243d;}
[data-testid="stMetricLabel"] p{color:#5a6a85;font-weight:600;}
section.main h1{color:#16243d;font-weight:800;letter-spacing:-.5px;}
section.main h2{color:#1f3a56;border-left:5px solid #c0392b;padding-left:12px;margin-top:.2rem;}
section.main h3{color:#1f3a56;}
[data-testid="stSidebar"]{background:#f4f6fa;}
hr{margin:.8rem 0;}
</style>''', unsafe_allow_html=True)


@st.cache_data
def load_combined():
    """2023–25 test(fold3) + 2026 live + 실제 ETF OHLC 를 연속 시계열로 결합."""
    iv = pd.read_csv(IV, parse_dates=['date'])
    f3 = iv[iv['fold'] == 'fold3'][['date', 'q50', 'y_true', 'online_lo', 'online_hi']].rename(
        columns={'online_lo': 'lo', 'online_hi': 'hi'})
    f3['period'] = 'test'
    lv = pd.read_csv(LIVE, parse_dates=['date'])[['date', 'q50', 'y_true', 'on_lo', 'on_hi']].rename(
        columns={'on_lo': 'lo', 'on_hi': 'hi'})
    lv['period'] = 'live'
    df = pd.concat([f3, lv], ignore_index=True).sort_values('date').reset_index(drop=True)

    # 금리 레벨(시그널 표시용)
    ec = pd.read_csv(ECOS, parse_dates=['date'])
    ylev = ec[ec['variable'] == 'kr_treasury_10y'].set_index('date')['value'].sort_index()
    df['y_lev'] = ylev.reindex(df['date']).ffill().values

    # 실제 ETF OHLC → 일별 수익 분해 (종가체결/시초가체결)
    etf = pd.read_csv(ETF, parse_dates=['date']).sort_values('date')
    etf['etf_c2c'] = etf['close'] / etf['close'].shift(1) - 1   # 종가→종가 (갭+장중)
    etf['etf_o2c'] = etf['close'] / etf['open'] - 1             # 시초가→종가 (장중만)
    df = df.merge(etf[['date', 'etf_c2c', 'etf_o2c']], on='date', how='inner')

    # 날짜별 추정 스프레드 (Corwin-Schultz, price %)
    try:
        cdf = pd.read_csv(COST, parse_dates=['date'])[['date', 'cs_spread_pct']]
        df = df.merge(cdf, on='date', how='left')
    except FileNotFoundError:
        df['cs_spread_pct'] = np.nan
    fill = df['cs_spread_pct'].median() if df['cs_spread_pct'].notna().any() else 0.0
    df['cs_spread_pct'] = df['cs_spread_pct'].fillna(fill)
    return df.reset_index(drop=True)


def backtest(df, exec_mode='open', cost_bp=5.0, use_estimated=False):
    """실제 ETF 체결 백테스트. pos=-sign(q50)(금리↓예측=ETF롱).
    exec_mode='close' 종가체결(이론, 갭 포함) / 'open' 시초가체결(현실, 장중만).
    비용은 price 기준(1bp=0.01%). ETF NAV 에 운용보수·carry 이미 반영."""
    pos = -np.sign(df['q50'].values)
    ret = df['etf_c2c'].values if exec_mode == 'close' else df['etf_o2c'].values
    gross = pos * ret
    dpos = np.abs(np.diff(pos, prepend=pos[0]))
    if use_estimated:
        cost = dpos * (df['cs_spread_pct'].values / 2) / 100   # CS 반스프레드(price 분수)
    else:
        cost = dpos * cost_bp / 10000                          # price bp
    return gross - cost


def buyhold(df):
    return df['etf_c2c'].values   # 항상 롱 ETF (총수익=가격+이자, NAV 반영)


def breakeven_bp(df):
    """선택 구간 시초가체결 기준 편도 손익분기(price-bp) = 무비용 누적 / 총 포지션변경수."""
    pos = -np.sign(df['q50'].values)
    g = pos * df['etf_o2c'].values
    dpos = np.abs(np.diff(pos, prepend=pos[0])).sum()
    return float(g.sum() / dpos * 10000) if dpos > 0 else float('nan')


def sharpe(p):
    p = np.asarray(p, float)
    return float(p.mean() / p.std() * np.sqrt(252)) if p.std() > 0 else float('nan')


def sharpe_ci(p, nb=1000, seed=42):
    p = np.asarray(p, float); rng = np.random.default_rng(seed); n = len(p); s = []
    for _ in range(nb):
        x = p[rng.integers(0, n, n)]
        if x.std() > 0:
            s.append(x.mean() / x.std() * np.sqrt(252))
    return (float(np.quantile(s, .025)), float(np.quantile(s, .975))) if s else (np.nan, np.nan)


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
    """매도 건별 과세(배당소득·실제): 각 트레이드 종료일에 양(+)이익 × rate. 손실 상계 없음."""
    t = np.zeros(len(pnl))
    for s, e in _segments(pos):
        realized = float(np.sum(pnl[s:e + 1]))
        if realized > 0:
            t[e] = realized * rate
    return t


def tax_annual(dates, pnl, rate=TAX_RATE):
    """연간 순이익 과세(비교·양도소득세식): 한 해 손익 통산 후 양수해에만 연말 과세(손실 통산O)."""
    t = np.zeros(len(pnl))
    yrs = pd.DatetimeIndex(dates).year
    for y in np.unique(yrs):
        idx = np.where(yrs == y)[0]
        net = float(np.sum(pnl[idx]))
        if net > 0:
            t[idx[-1]] = net * rate
    return t


df = load_combined()
last = df.iloc[-1]

# ============================== Sidebar ==============================
st.sidebar.header('⚙️ 설정')
period_opt = st.sidebar.radio(
    '기간',
    ['전체 (2023–2026)', '최근 3개월', '최근 6개월', '최근 1년', '2026 라이브만', '2023–25 test만', '사용자 지정'],
    index=5,   # 기본 = 2023–25 test (발표 기준, 라이브 제외)
)
exec_label = st.sidebar.radio('체결 시점', ['시초가 (현실)', '종가 (이론)'],
                              help='신호는 T 종가에 산출 → 현실은 T+1 시초가 체결. '
                                   '종가체결은 못 잡는 overnight 갭까지 포함하는 낙관 가정.')
exec_mode = 'open' if exec_label.startswith('시초가') else 'close'
exec_tag = '시초가' if exec_mode == 'open' else '종가'
show_theory = st.sidebar.checkbox('종가체결(이론) 참고선 표시', value=False,
                                  help='켜면 누적수익 차트에 종가체결(이론·무비용) 점선을 겹쳐 표시.')

cost_bp = st.sidebar.slider('거래비용 (price bp, 편도)', 0.0, 10.0, 1.0, 0.5,
                            help='1 price-bp = 0.01%(편도). 유동성 좋은 KOSEF 틱수준 ~0.25~0.5bp.')
st.sidebar.caption('💡 현실 틱수준 편도 ~0.5~1bp. 손익분기는 선택 구간 기준으로 아래 ④에 표시됩니다.')
cost_label = f'{cost_bp:.1f}bp'
use_est = False   # 추정 실비용 모드 제거(CS 추정 부적합) — 슬라이더 단일

apply_tax = st.sidebar.checkbox('세금 15.4% 반영', value=False,
                                help='국내 채권형 ETF 매매차익 = 배당소득세 15.4%.')
if apply_tax:
    tax_method = st.sidebar.radio('세금 방식', ['매도 건별 (배당소득·실제)', '연간 순이익 (비교)'],
                                  help='국내 채권형 ETF 는 **매도 건별 원천징수·손실 상계 X**(실제, 보수적). '
                                       '"연간 순이익"은 한 해 손익 통산 후 양수해에만 과세(더 관대) — 양도소득세식 비교용.')
else:
    tax_method = '매도 건별 (배당소득·실제)'
capital_eok = st.sidebar.number_input('자본 기준 (억원)', min_value=1, max_value=10000, value=100, step=10,
                                      help='원 단위 손익 표시용. 세후 수익률(%)은 자본 크기와 무관(비례).')
CAPITAL_EOK = capital_eok

end = df['date'].max()
if period_opt == '최근 3개월':
    start = end - pd.DateOffset(months=3)
elif period_opt == '최근 6개월':
    start = end - pd.DateOffset(months=6)
elif period_opt == '최근 1년':
    start = end - pd.DateOffset(years=1)
elif period_opt == '2026 라이브만':
    start = LIVE_START
elif period_opt == '2023–25 test만':
    start, end = df['date'].min(), pd.Timestamp('2025-12-31')
elif period_opt == '사용자 지정':
    rng = st.sidebar.date_input('날짜 범위', value=(df['date'].min().date(), end.date()),
                                min_value=df['date'].min().date(), max_value=end.date())
    start = pd.Timestamp(rng[0]); end = pd.Timestamp(rng[1]) if len(rng) > 1 else end
else:
    start = df['date'].min()

win = df[(df['date'] >= start) & (df['date'] <= end)].reset_index(drop=True)
be_win = breakeven_bp(win)                                  # 선택 구간 손익분기(편도 price-bp)
e_o2c_nc = float(np.sum(backtest(win, 'open', 0.0))) * 100  # 시초가·무비용
e_o2c_1bp = float(np.sum(backtest(win, 'open', 1.0))) * 100 # 시초가·1bp
e_c2c_nc = float(np.sum(backtest(win, 'close', 0.0))) * 100 # 종가·무비용(이론)
bh_nc = float(np.sum(buyhold(win))) * 100                   # Buy&Hold

# ============================== Header + 오늘의 시그널 ==============================
st.markdown('# 📈 한국 국고채 10년물 — 일별 금리 방향 예측 데모 (실거래 ETF)')
st.caption('동결 v3 XGBoost 분위수 회귀 · 거시변수 13개 · 익일 Δy(bp) 예측 → 실제 KOSEF 국고채10년 ETF 체결 백테스트 '
           '(분배금 조정가 · 기본 비용 1bp 틱수준)')

sig_dir = '🔴 숏 (금리 상승 예측)' if last['q50'] > 0 else '🟢 롱 (금리 하락 예측)'
conf = '강' if abs(last['q50']) > 1.0 else ('중' if abs(last['q50']) > 0.4 else '약')
hc1, hc2, hc3, hc4 = st.columns([1.4, 1, 1, 1.2])
hc1.markdown(f"### 🔔 최신 시그널 — {last['date'].date()}\n**{sig_dir}**  · 신뢰도 **{conf}**")
hc2.metric('예측 Δy (q50)', f"{last['q50']:+.2f} bp")
hc3.metric('현재 10Y 금리', f"{last['y_lev']:.2f} %")
hc4.metric('90% 예측구간', f"[{last['lo']:+.1f}, {last['hi']:+.1f}] bp")
st.caption('※ 모델은 *금리(yield)* 방향 예측 → 채권 가격은 역방향(금리↑=가격↓=숏). 신뢰도=|q50| 크기.')
st.divider()

# ============================== 윈도우 지표 ==============================
valid = win['y_true'] != 0
dir_acc = float((np.sign(win['q50'][valid]) == np.sign(win['y_true'][valid])).mean()) if valid.any() else float('nan')
pos = -np.sign(win['q50'].values)
pnl = backtest(win, exec_mode=exec_mode, cost_bp=cost_bp, use_estimated=use_est); bh = buyhold(win)
sh = sharpe(pnl); lo, hi = sharpe_ci(pnl)   # Sharpe 는 세전(신호 품질) 기준

if not apply_tax:
    tax_s = np.zeros(len(pnl)); tax_b = np.zeros(len(bh))
elif tax_method.startswith('연간'):
    tax_s = tax_annual(win['date'].values, pnl); tax_b = tax_annual(win['date'].values, bh)
else:
    tax_s = tax_per_day(pos, pnl); tax_b = tax_per_day(np.ones(len(bh)), bh)
pnl_at = pnl - tax_s; bh_at = bh - tax_b
cum = np.cumsum(pnl_at) * 100; cum_bh = np.cumsum(bh_at) * 100
gross_s = float(np.sum(pnl)); net_s = float(np.sum(pnl_at)); tax_tot_s = float(np.sum(tax_s))
net_b = float(np.sum(bh_at)); tax_tot_b = float(np.sum(tax_b))
is_live = (win['period'] == 'live').all()
label = '라이브 OOS' if is_live else ('test' if (win['period'] == 'test').all() else 'test+라이브')
disp_label = f'{cost_label}·{exec_tag}' + (' +세금' if apply_tax else '')

st.markdown(f"## ① 성과 요약 — {start.date()} ~ {end.date()}  ·  {len(win)}영업일  ·  *{label}*  ·  **{exec_tag}체결**")
k1, k2, k3, k4 = st.columns(4)
k1.metric('방향 정확도', f'{dir_acc:.3f}', f'{(dir_acc-0.5)*100:+.1f}%p vs 0.5', help=f'n={int(valid.sum())}')
k2.metric(f'누적 수익 ({disp_label})', f'{cum[-1]:+.2f}%', f'{cum[-1]-cum_bh[-1]:+.2f}%p vs B&H')
k3.metric('Sharpe (세전)', f'{sh:+.2f}', f'CI [{lo:+.2f}, {hi:+.2f}]', delta_color='off',
          help='신호 품질 지표라 세전 기준. CI 가 0 포함이면 통계 유의성 미확보(소표본/구간 따라).')
k4.metric('Buy & Hold', f'{cum_bh[-1]:+.2f}%', '벤치마크', delta_color='off')

if exec_mode == 'close':
    st.warning('⚠️ **종가체결(이론)** — 신호 난 종가에 즉시 체결 가정. 못 잡는 overnight 갭까지 포함해 '
               '**과대평가**. 현실 비교는 사이드바에서 *시초가(현실)* 선택.')

# 원 단위 (자본 기준) — 세금 비대칭 노출
st.markdown(f'**💰 {CAPITAL_EOK}억원 기준 손익**' + ('  ·  세금 ON' if apply_tax else '  ·  세금 OFF'))
w1, w2, w3, w4 = st.columns(4)
w1.metric('전략 세전 수익', f'{gross_s*CAPITAL_EOK:+.2f}억원')
w2.metric('납부 세금', f'-{tax_tot_s*CAPITAL_EOK:.2f}억원', help='이익 실현분 15.4% (전략은 자주 실현)')
w3.metric('전략 세후 수익', f'{net_s*CAPITAL_EOK:+.2f}억원', f'{(net_s-gross_s)*CAPITAL_EOK:+.2f}억원 세금')
w4.metric('B&H 세후 수익', f'{net_b*CAPITAL_EOK:+.2f}억원', f'-{tax_tot_b*CAPITAL_EOK:.2f}억원 세금',
          help='B&H 도 분배금/매도차익에 과세. 단 회전이 적어 세금 부담 작음.')
if apply_tax:
    st.caption(f'⚖️ 세금({tax_method}) — 전략 −{tax_tot_s*CAPITAL_EOK:.2f}억 / B&H −{tax_tot_b*CAPITAL_EOK:.2f}억. '
               f'**매도 건별** = 손실 매매 상계 안 됨(채권ETF 실제·보수적), **연간 순이익** = 손익 통산(더 관대). '
               f'전략은 회전이 많아 매도 건별에서 더 불리.')
st.caption(f'※ 수익률(%)은 자본 크기와 무관(비례). {CAPITAL_EOK}억원은 원 단위 체감용 — '
           f'단 KOSEF 일평균 거래대금 ~18억원이라 {CAPITAL_EOK}억 체결은 시장충격 발생(미반영, 정직 한계). '
           f'손익은 고정 명목(단리 합산) 기준.')

if len(win) < 60:
    st.warning(f'⚠️ 선택 구간 {len(win)}일 — 소표본이라 Sharpe·CI 해석 주의 (라이브 sanity-check 수준).')
st.divider()

# ============================== 누적 수익률 ==============================
st.markdown(f'## ② 누적 수익률 — 전략({exec_tag}체결{"·세금" if apply_tax else ""}) vs Buy & Hold')
fig = go.Figure()
if show_theory:
    cum_theory = np.cumsum(backtest(win, exec_mode='close', cost_bp=0.0)) * 100
    fig.add_trace(go.Scatter(x=win['date'], y=cum_theory, name='종가체결(이론·무비용)',
                             line=dict(color='lightgray', width=1.2, dash='dot')))
fig.add_trace(go.Scatter(x=win['date'], y=cum, name=f'모델 전략 ({disp_label})',
                         line=dict(color='crimson', width=2.3)))
fig.add_trace(go.Scatter(x=win['date'], y=cum_bh, name='Buy & Hold',
                         line=dict(color='black', width=1.2, dash='dash')))
fig.add_hline(y=0, line=dict(color='black', width=0.5))
fig.update_layout(height=380, margin=dict(t=30, b=30, l=40, r=10),
                  yaxis_title='누적 수익률 (%)', legend=dict(orientation='h', y=1.12, x=0.35))
st.plotly_chart(fig, width='stretch')
st.caption(f'🔴 **전략 최종 {cum[-1]:+.1f}%** = {exec_tag}체결·{cost_label}'
           f'{("·세금 " + tax_method.split(" ")[0]) if apply_tax else "·세금 OFF"}. '
           f'참고로 시초가·1bp·세금 OFF 면 **{e_o2c_1bp:+.0f}%** (B&H {bh_nc:+.0f}%), 세금 ON 이면 내려갑니다(① 원 단위 참조). '
           f'종가체결(이론) 참고선은 사이드바 체크박스로 on/off.')
st.divider()

# ============================== 예측 + 방향 + 구간 ==============================
st.markdown('## ③ 일별 예측 Δy + 방향 적중 + 90% 예측구간')
f2 = go.Figure()
f2.add_trace(go.Scatter(x=win['date'], y=win['lo'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
f2.add_trace(go.Scatter(x=win['date'], y=win['hi'], line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(70,130,180,0.18)', name='90% 예측구간'))
f2.add_trace(go.Scatter(x=win['date'], y=win['q50'], line=dict(color='steelblue', width=1.2), name='예측 q50'))
ok = np.sign(win['q50']) == np.sign(win['y_true'])
f2.add_trace(go.Scatter(x=win['date'][ok], y=win['y_true'][ok], mode='markers',
                        marker=dict(size=4.5, color='green'), name='실제 (방향 ✅)'))
f2.add_trace(go.Scatter(x=win['date'][~ok], y=win['y_true'][~ok], mode='markers',
                        marker=dict(size=4.5, color='red', symbol='x'), name='실제 (방향 ❌)'))
f2.add_hline(y=0, line=dict(color='gray', width=0.5))
f2.update_layout(height=400, margin=dict(t=30, b=30, l=40, r=10),
                 yaxis_title='Δy (bp)', legend=dict(orientation='h', y=1.1, x=0.3))
st.plotly_chart(f2, width='stretch')
st.caption('🟢 방향 적중 · 🔴 오답 · 구간은 0을 거의 항상 포함(0배제율≈0) → 구간=불확실성, 방향=q50 (nb12 정직 결론).')
st.divider()

# ============================== 체결 시점 비교 + 월별 ==============================
cc1, cc2 = st.columns(2)
with cc1:
    st.markdown('### ④ 체결 시점별 비교 (선택 구간, 비용 0)')
    e_c2c = float(np.sum(backtest(win, exec_mode='close', cost_bp=0.0))) * 100
    e_o2c = float(np.sum(backtest(win, exec_mode='open', cost_bp=0.0))) * 100
    # 갭 기여 = 종가 - 시초가
    fe = go.Figure(go.Bar(x=['종가체결\n(이론)', '시초가체결\n(현실)', '갭 기여\n(못잡음)'],
                          y=[e_c2c, e_o2c, e_c2c - e_o2c],
                          marker_color=['lightgray', 'crimson', 'orange']))
    fe.add_hline(y=0, line=dict(color='black', width=0.5))
    fe.update_layout(height=300, margin=dict(t=10, b=40, l=40, r=10), yaxis_title='누적 수익(%)')
    st.plotly_chart(fe, width='stretch')
    st.caption(f'엣지 상당부분이 **overnight 갭**(종가−시초가 = {e_c2c-e_o2c:+.1f}%p)에 있어 종가체결은 비현실적. '
               f'위는 비용 0 기준 — 현실 틱수준(편도 ~0.5~1bp)이면 시초가도 양수(**손익분기 편도 {be_win:.1f}bp**).')

with cc2:
    st.markdown('### ⑤ 월별 손익 (레짐 집중 노출)')
    mm = win.copy(); mm['pnl'] = backtest(mm, exec_mode=exec_mode, cost_bp=cost_bp, use_estimated=use_est) * 100
    mm['month'] = mm['date'].dt.to_period('M').astype(str)
    g = mm.groupby('month')['pnl'].sum().reset_index()
    fm = go.Figure(go.Bar(x=g['month'], y=g['pnl'],
                          marker_color=['green' if v > 0 else 'crimson' for v in g['pnl']]))
    fm.add_hline(y=0, line=dict(color='black', width=0.5))
    fm.update_layout(height=300, margin=dict(t=10, b=60, l=40, r=10), yaxis_title='월 손익(%)', xaxis_title='')
    fm.update_xaxes(tickangle=-60)
    st.plotly_chart(fm, width='stretch')
    st.caption('수익이 일부 월/추세 국면에 집중 — 엣지의 레짐 의존성을 덮지 않고 노출.')

st.divider()
st.markdown('### 📌 정직한 결론')
st.markdown(f'''
- **실거래 기반**: 실제 KOSEF 국고채10년 ETF **분배금 조정** 체결가로 백테스트(yield 근사 폐기). carry·분배금·운용보수 NAV 반영.
- **체결 현실성**: 엣지 상당부분이 **overnight 갭**에 있어 종가체결(이론) **{e_c2c_nc:+.0f}%**는 비현실. 장중만 잡는 시초가체결은 **{e_o2c_nc:+.0f}%**(무비용).
- **현실 비용 후에도 양수(얇음)**: 현실 틱수준 편도 ~0.5~1bp 면 시초가 **{e_o2c_1bp:+.0f}%**, **B&H({bh_nc:+.0f}%) 능가**. 단 **손익분기 편도 {be_win:.1f}bp** — 마진 얇고 비용 민감.
- **세금은 별도 역풍**: 국내 채권형 ETF = 배당소득세 15.4%, **매도 건별·손실 상계 X**(실제) → 회전 많은 전략에 불리. B&H 도 과세되나 회전 적어 부담 작음. (연간 순이익 방식 비교 토글 제공.) ④ 그림은 *수수료만*.
- **통계 증명 vs 라이브**: 2023–25 test 방향 DM 유의(p<0.0001)·in-sample 0.62 유효 — 단 *예측 정확도*이지 *거래 수익성*과 별개.
- **갈 길**: 저비용·비과세 수단(KTB 선물·장내 채권 직접매매 = 매매차익 비과세) 또는 **회전율 축소**로 비용·세금 절감 시 현실 alpha 여지.
''')
st.caption('데이터: predictions_xgb_v3_intervals.csv + live_oos_2026_xgb.csv + kosef_10y_daily_2023_2026.csv · 기말 통합 데모')

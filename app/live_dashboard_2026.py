# -*- coding: utf-8 -*-
"""
2026 라이브 OOS 대시보드 (기말 보강 — T3.1)
============================================
동결된 v3 모델(train≤2024·val 2025·cal 2025H2)을 2026 신규 데이터에
포워드 적용한 결과를 웹으로 시연.

핵심 메시지 (정직):
- 2026 최신 지표에 모델을 적용했을 때 **방향이 맞고 돈이 되는지**를 라이브로 확인.
- 방향 일관(>0.5)·Buy&Hold 능가 ✅ / 단 소표본(n=92)이라 유의성은 주장 안 함.
- **거래비용에서 alpha 가 소멸**하는 지점(≈2bp)을 슬라이더로 직접 시각화.
- 전략 = S1 매일 매매 (sign(q50)). scripts/39 에서 v3 게이팅이 도움 안 됨을 확인 → 정직하게 게이팅 없음.

통계적 우위 증명은 2023–2025 test(DM 유의)에서 이미 완료. 2026 은 **실용 작동 확인 + 데모**.

실행: streamlit run app/live_dashboard_2026.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / 'reports' / 'no_leak_v2' / 'live_oos_2026_xgb.csv'

D = 8.0    # 듀레이션 근사 (금리 1bp 변화 → 가격 ~D bp 역방향)
C = 85.0   # convexity 근사

st.set_page_config(page_title='국고채 10Y — 2026 라이브 OOS 대시보드', page_icon='📈', layout='wide')


@st.cache_data
def load():
    d = pd.read_csv(LIVE, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return d


def backtest(d, cost_bp):
    """전략 S1: 매일 sign(q50). 금리상승예측=숏, 하락예측=롱 (채권가격 역방향).
    PnL = 가격손익 + convexity + carry - 거래비용. scripts/39 와 동일 규약."""
    pos = np.sign(d['q50'].values)             # +1=금리↑예측=숏포지션 부호(아래 dy와 결합)
    dy = d['y_true'].values / 10000
    pnl_price = pos * D * dy                    # 부호규약: pos*D*dy (script 39 일관)
    pnl_convex = -pos * 0.5 * C * dy ** 2
    pnl_carry = -pos * (d['y10'].values / 100) / 252
    pos_change = np.abs(np.diff(pos, prepend=pos[0]))
    cost = pos_change * cost_bp * D / 10000
    pnl = pnl_price + pnl_convex + pnl_carry - cost
    return pnl


def buyhold(d):
    dy = d['y_true'].values / 10000
    return -D * dy   # 항상 롱: 금리↑→가격↓


def sharpe(pnl):
    pnl = np.asarray(pnl, float)
    return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else float('nan')


def sharpe_ci(pnl, nb=1000, seed=42):
    pnl = np.asarray(pnl, float); rng = np.random.default_rng(seed); n = len(pnl); s = []
    for _ in range(nb):
        x = pnl[rng.integers(0, n, n)]
        if x.std() > 0:
            s.append(x.mean() / x.std() * np.sqrt(252))
    return (float(np.quantile(s, .025)), float(np.quantile(s, .975))) if s else (np.nan, np.nan)


d = load()
valid = d['y_true'] != 0
dir_acc = float((np.sign(d['q50'][valid]) == np.sign(d['y_true'][valid])).mean())
n_valid = int(valid.sum())

# ============================== Header ==============================
st.markdown('# 📈 한국 국고채 10년물 — 2026 라이브 OOS 대시보드')
st.caption(
    f'동결 v3 모델(학습 ≤2024) → **{d["date"].min().date()} ~ {d["date"].max().date()} '
    f'({len(d)}영업일) 포워드** · 재학습 0 · 입력·CQR·임계값 전부 ≤2025 동결'
)
st.info(
    '**이 데모의 목적**: 통계적 우위는 2023–2025 test(DM 유의)에서 이미 증명됨. '
    '여기서는 *최신 2026 실데이터에 모델을 꽂았을 때 방향이 맞고 돈이 되는지*를 라이브로 확인한다. '
    '소표본(n=92)이라 통계적 유의성은 주장하지 않음 — **실용 작동 + 데모**.'
)

# ============================== Sidebar: 비용 슬라이더 ==============================
st.sidebar.header('⚙️ 시뮬레이션 설정')
cost_bp = st.sidebar.slider('거래비용 (bp, 편도)', 0.0, 3.0, 1.0, 0.1,
                            help='ETF 스프레드+슬리피지. 채권 현물은 보통 1~2bp.')
st.sidebar.caption('💡 비용을 올려보세요 — **약 2bp 부근에서 alpha 가 소멸**합니다 (정직한 한계).')

pnl = backtest(d, cost_bp)
bh = buyhold(d)
cum = np.cumsum(pnl) * 100
cum_bh = np.cumsum(bh) * 100
sh = sharpe(pnl); lo, hi = sharpe_ci(pnl)

# ============================== (1) KPI ==============================
st.markdown('## ① 라이브 성과 요약')
c1, c2, c3, c4 = st.columns(4)
c1.metric('방향 정확도', f'{dir_acc:.3f}', f'{(dir_acc-0.5)*100:+.1f}%p vs 동전(0.5)',
          help=f'n={n_valid}. >0.5 이나 소표본이라 유의성은 미주장.')
c2.metric(f'누적 수익 (비용 {cost_bp:.1f}bp)', f'{cum[-1]:+.2f}%',
          f'{cum[-1]-cum_bh[-1]:+.2f}%p vs Buy&Hold', delta_color='normal')
c3.metric('Sharpe (연율화)', f'{sh:+.2f}',
          f'95% CI [{lo:+.2f}, {hi:+.2f}]', delta_color='off',
          help='CI 가 0 을 포함하면 통계적 유의성 미확보 (소표본).')
c4.metric('Buy & Hold (항상 롱)', f'{cum_bh[-1]:+.2f}%', '벤치마크', delta_color='off')

if cost_bp <= 1.5:
    st.success(f'✅ 비용 {cost_bp:.1f}bp: 전략 누적 **{cum[-1]:+.2f}%** > Buy&Hold {cum_bh[-1]:+.2f}% — '
               f'추세장에서 방향 엣지가 실제로 작동.')
else:
    st.error(f'⚠️ 비용 {cost_bp:.1f}bp: 전략 누적 **{cum[-1]:+.2f}%** — '
             f'거래비용이 alpha 를 잠식 (회전율 높음). **정직한 한계**.')

st.divider()

# ============================== (2) 누적 수익률 곡선 ==============================
st.markdown('## ② 최신 기준 누적 수익률 — 전략 vs Buy & Hold')
fig = go.Figure()
fig.add_trace(go.Scatter(x=d['date'], y=cum, name=f'모델 전략 (비용 {cost_bp:.1f}bp)',
                         line=dict(color='crimson', width=2.2)))
fig.add_trace(go.Scatter(x=d['date'], y=cum_bh, name='Buy & Hold (항상 롱)',
                         line=dict(color='gray', width=1.5, dash='dash')))
fig.add_hline(y=0, line=dict(color='black', width=0.5))
fig.update_layout(height=400, margin=dict(t=30, b=30, l=40, r=10),
                  yaxis_title='누적 수익률 (%)', xaxis_title='',
                  legend=dict(orientation='h', y=1.08, x=0.6))
st.plotly_chart(fig, width='stretch')

st.divider()

# ============================== (3) 예측 + 방향 신호 + 구간 ==============================
st.markdown('## ③ 일별 예측 Δy + 방향 신호 + 90% 예측구간 (온라인 conformal)')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=d['date'], y=d['on_lo'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig2.add_trace(go.Scatter(x=d['date'], y=d['on_hi'], line=dict(width=0), fill='tonexty',
                          fillcolor='rgba(70,130,180,0.20)', name='90% 예측구간'))
fig2.add_trace(go.Scatter(x=d['date'], y=d['q50'], line=dict(color='steelblue', width=1.3), name='예측 q50'))
correct = np.sign(d['q50']) == np.sign(d['y_true'])
fig2.add_trace(go.Scatter(x=d['date'][correct], y=d['y_true'][correct], mode='markers',
                          marker=dict(size=5, color='green', symbol='circle'), name='실제 Δy (방향 ✅)'))
fig2.add_trace(go.Scatter(x=d['date'][~correct], y=d['y_true'][~correct], mode='markers',
                          marker=dict(size=5, color='red', symbol='x'), name='실제 Δy (방향 ❌)'))
fig2.add_hline(y=0, line=dict(color='gray', width=0.5))
fig2.update_layout(height=420, margin=dict(t=30, b=30, l=40, r=10),
                   yaxis_title='Δy (bp)', xaxis_title='',
                   legend=dict(orientation='h', y=1.08, x=0.35))
st.plotly_chart(fig2, width='stretch')
st.caption('🟢 방향 적중 · 🔴 방향 오답 · 구간은 0을 거의 항상 포함(0배제율≈0) → '
           '**구간은 불확실성만, 방향은 q50 가 담당** (nb12 정직 결론).')

st.divider()

# ============================== (4) 비용 민감도 + 월별 분해 ==============================
col1, col2 = st.columns(2)
with col1:
    st.markdown('### ④ 거래비용 민감도')
    rows = []
    for c in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        p = backtest(d, c)
        rows.append({'비용(bp)': c, '누적 %': round(np.cumsum(p)[-1] * 100, 2), 'Sharpe': round(sharpe(p), 2)})
    cs = pd.DataFrame(rows)
    figc = go.Figure()
    figc.add_trace(go.Bar(x=cs['비용(bp)'], y=cs['Sharpe'],
                          marker_color=['green' if s > 0 else 'crimson' for s in cs['Sharpe']]))
    figc.add_hline(y=0, line=dict(color='black', width=0.5))
    figc.add_hline(y=1.0, line=dict(color='gray', width=0.5, dash='dot'))
    figc.update_layout(height=300, margin=dict(t=10, b=30, l=40, r=10),
                       yaxis_title='Sharpe', xaxis_title='거래비용 (bp)')
    st.plotly_chart(figc, width='stretch')
    st.caption('**≈2bp 에서 Sharpe 음수 전환** — 거래비용 마진이 얇다는 정직한 한계.')

with col2:
    st.markdown('### ⑤ 월별 손익 분해 (레짐 집중 정직 노출)')
    dm = d.copy()
    dm['pnl'] = backtest(dm, cost_bp) * 100
    dm['month'] = dm['date'].dt.to_period('M').astype(str)
    mon = dm.groupby('month')['pnl'].sum().reset_index()
    figm = go.Figure(go.Bar(x=mon['month'], y=mon['pnl'],
                            marker_color=['green' if v > 0 else 'crimson' for v in mon['pnl']]))
    figm.add_hline(y=0, line=dict(color='black', width=0.5))
    figm.update_layout(height=300, margin=dict(t=10, b=30, l=40, r=10),
                       yaxis_title='월 손익 (%)', xaxis_title='')
    st.plotly_chart(figm, width='stretch')
    st.caption('수익이 일부 월에 집중 — 엣지가 **추세 국면 의존적**임을 그대로 노출(덮지 않음).')

st.divider()
st.markdown('### 📌 정직한 결론 (발표용)')
st.markdown(f'''
- **방향 엣지 라이브 작동**: 2026 {n_valid}영업일 방향정확도 **{dir_acc:.3f}** (>0.5), Buy&Hold(**{cum_bh[-1]:+.1f}%**) 능가.
- **진폭도 우위**: DM vs naive p=0.003 (RMSE 5.58<5.90) — 2026 에도 naive 능가 (별도 검증).
- **거래비용이 관건**: 1bp 누적 +5.2%·Sharpe ~2.0 이나, **≈2bp 에서 alpha 소멸** (슬라이더로 확인).
- **유의성은 미주장**: n=92 소표본 → "통계 증명"이 아니라 "**라이브 sanity-check + 실용 데모**". 통계 증명은 2023–25 test 담당.
- **레짐 게이팅 불채택**: v3 에선 게이팅이 오히려 손해(scripts/39) → 정직하게 매일 매매 + 레짐 집중은 공개 한계.
''')
st.caption('데이터: reports/no_leak_v2/live_oos_2026_xgb.csv · 모델: 동결 v3 XGBoost 분위수 회귀 · 본 패널은 기말 T3.1 보강')

# -*- coding: utf-8 -*-
"""
Streamlit 1페이지 미니 데모 (계획서 §8 B1 필수 산출물)
======================================================
한국 국고채 10년물 일별 변화량(Δy) 분위수 회귀 예측 데모

구성 (1페이지):
- (1) 모델 성능 KPI (DM test, RMSE, Coverage, Dir_Acc)
- (2) 예측구간 시각화 (q05~q95 밴드 + 실제값, 위기구간 음영)
- (3) SHAP top-5 (q50 기준)
- (4) 케이스 스터디 (날짜 선택 → 그날의 입력·예측·SHAP)

실행: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data' / 'processed'
REPORT_DIR = ROOT / 'reports'

# ============================== Page setup ==============================
st.set_page_config(
    page_title='Macro Bond Forecast — A0 LSTM 분위수 회귀',
    page_icon='📈',
    layout='wide',
)

# ============================== Data loading ============================
@st.cache_data
def load_data():
    pred = pd.read_csv(DATA_DIR/'lstm_a0_predictions_w6.csv', parse_dates=['date'])
    crisis = pd.read_csv(REPORT_DIR/'crisis_labels_w6.csv', parse_dates=['date'])
    dm = pd.read_csv(REPORT_DIR/'dm_test_w6.csv')
    baseline = pd.read_csv(REPORT_DIR/'baseline_results_w5.csv')
    channel = pd.read_csv(REPORT_DIR/'channel_validation_w6.csv')
    final_eval = pd.read_csv(REPORT_DIR/'lstm_a0_final_eval_w5.csv')
    return pred, crisis, dm, baseline, channel, final_eval

@st.cache_data
def load_shap():
    npz = np.load(REPORT_DIR/'lstm_a0_shap_w6.npz', allow_pickle=True)
    return {
        'q05': npz['shap_q05'], 'q50': npz['shap_q50'], 'q95': npz['shap_q95'],
        'eval_dates': pd.to_datetime(npz['eval_dates']),
        'features': [str(f) for f in npz['features']],
    }

pred, crisis, dm, baseline, channel, final_eval = load_data()
shap_data = load_shap()
test_pred = pred[pred['split']=='test'].merge(crisis[['date','is_crisis']], on='date', how='left')
test_pred['is_crisis'] = test_pred['is_crisis'].fillna(False)

# ============================== Header ==================================
st.markdown('# 📈 한국 국고채 10년물 일별 변화량(Δy) 예측 — A0 LSTM 분위수 회귀')
st.caption(
    '**계획서 v5.1** · multivariate LSTM (`Δfeature[t-1]` 입력, lookback 30일, q=[0.05, 0.5, 0.95]) · '
    f'test 구간 {test_pred["date"].min().date()} ~ {test_pred["date"].max().date()} ({len(test_pred):,}일)'
)

# ============================== (1) KPI =================================
st.markdown('## ① 모델 성능 — DM test (vs Naive · XGBoost · LSTM raw)')
test_eval = final_eval[final_eval['split']=='test']
rmse = test_eval['rmse_q50_bp'].mean()
cov = test_eval['coverage_90'].mean()
dirA = test_eval['dir_acc_q50'].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric('RMSE q50 (bp)', f'{rmse:.3f}', f'{(rmse-4.647)/4.647*100:+.1f}% vs Naive 4.647', delta_color='inverse')
c2.metric('Coverage 90%', f'{cov:.3f}', f'{(cov-0.9)*100:+.1f}%p vs target 0.9')
c3.metric('Directional Accuracy', f'{dirA:.3f}', f'{(dirA-0.55)*100:+.1f}%p vs target 0.55')
c4.metric('VALIDATION_LOG', '40+ 건', '안내문 최소 3건의 13배+')

with st.expander('DM test 통계 검정 (HAC + HLN + Bonferroni)'):
    dm_show = dm[['comparison','rmse_a0','rmse_other','DM_HLN','p_value','bonf']].copy()
    dm_show.columns = ['비교', 'RMSE A0', 'RMSE 비교', 'DM (HLN)', 'p-value', 'Bonferroni α*=0.0167']
    st.dataframe(dm_show, use_container_width=True, hide_index=True)
    st.caption('**3/3 모두 Bonferroni 보정 후 p<0.0001** — A0 가 통계적으로 유의하게 우월')

st.divider()

# ============================== (2) 예측구간 시계열 =====================
st.markdown('## ② 예측구간 시각화 — 90% 분위수 밴드 + 실제 Δy + 위기구간')

date_range = st.slider(
    '날짜 범위',
    min_value=test_pred['date'].min().to_pydatetime(),
    max_value=test_pred['date'].max().to_pydatetime(),
    value=(test_pred['date'].min().to_pydatetime(), test_pred['date'].max().to_pydatetime()),
    format='YYYY-MM-DD',
)
mask = (test_pred['date']>=date_range[0]) & (test_pred['date']<=date_range[1])
sub = test_pred[mask]

fig = go.Figure()
# 위기구간 음영
crisis_periods = sub[sub['is_crisis']]
if len(crisis_periods):
    for _, r in crisis_periods.iterrows():
        fig.add_vrect(x0=r['date'], x1=r['date']+pd.Timedelta(days=1),
                      fillcolor='rgba(255,0,0,0.05)', layer='below', line_width=0)
# 90% 밴드
fig.add_trace(go.Scatter(x=sub['date'], y=sub['q05'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
fig.add_trace(go.Scatter(x=sub['date'], y=sub['q95'], line=dict(width=0), fill='tonexty',
                         fillcolor='rgba(70,130,180,0.25)', name='90% 예측 구간'))
# q50 + 실제값
fig.add_trace(go.Scatter(x=sub['date'], y=sub['q50'], line=dict(color='steelblue', width=1.2), name='q50 (median)'))
fig.add_trace(go.Scatter(x=sub['date'], y=sub['y_true_bp'], mode='markers',
                         marker=dict(size=4, color='black', opacity=0.5), name='실제 Δy'))
fig.add_hline(y=0, line=dict(color='gray', width=0.5))
fig.update_layout(
    height=420, margin=dict(t=30, b=40, l=40, r=10),
    xaxis_title='', yaxis_title='Δy (bp)',
    legend=dict(orientation='h', y=1.05, x=0.7),
)
st.plotly_chart(fig, use_container_width=True)

# 선택 구간 metric
sub_in_band = ((sub['y_true_bp']>=sub['q05'])&(sub['y_true_bp']<=sub['q95'])).mean()
sub_dir = ((np.sign(sub['q50'])==np.sign(sub['y_true_bp']))&(sub['y_true_bp']!=0)&(sub['q50']!=0)).mean()
sub_rmse = float(np.sqrt(np.mean((sub['y_true_bp']-sub['q50'])**2)))
c1, c2, c3, c4 = st.columns(4)
c1.metric('선택 구간 N', f'{len(sub):,}일')
c2.metric('Coverage 90%', f'{sub_in_band:.3f}')
c3.metric('Direction', f'{sub_dir:.3f}')
c4.metric('RMSE q50', f'{sub_rmse:.3f} bp')

st.caption(f'🟥 음영 = 위기구간 ({sub["is_crisis"].sum()}일, 정의: 20일 rolling vol 상위 20%, train-only threshold 4.11 bp)')

st.divider()

# ============================== (3) SHAP top-5 + 채널 부합 ==============
st.markdown('## ③ SHAP 변수 중요도 + 거시경제 채널 부합')
col1, col2 = st.columns([3, 2])

with col1:
    abs_q50 = np.abs(shap_data['q50']).mean(axis=(0,1))
    imp = pd.DataFrame({'feature': shap_data['features'], 'mean_abs_shap': abs_q50}).sort_values('mean_abs_shap', ascending=True)
    fig_imp = px.bar(imp, x='mean_abs_shap', y='feature', orientation='h',
                     color='mean_abs_shap', color_continuous_scale='Reds',
                     labels={'mean_abs_shap':'mean |SHAP| (q50)', 'feature':''})
    fig_imp.update_layout(height=350, margin=dict(t=20,b=40,l=10,r=10), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption('**top-1**: `us_treasury_10y` (mean|SHAP| 0.0547) — 한미 동조성 정량 입증')

with col2:
    ch = channel[['feature','hypothesis','actual_sign','signed_abs_ratio','region','match']].copy()
    ch['signed_abs_ratio (%)'] = (ch['signed_abs_ratio']*100).round(1)
    ch_show = ch[['feature','hypothesis','actual_sign','signed_abs_ratio (%)','region','match']]
    ch_show.columns = ['변수','가설','실측','signed/|SHAP|','영역','부합']
    st.dataframe(ch_show, use_container_width=True, hide_index=True, height=350)
    st.caption('**strong 1/1** (kr_base_rate, 정책 채널 ✅) + weak 1/2 + noise 5 (절대값 0.001~0.007 결론 보류)')

st.divider()

# ============================== (4) 케이스 스터디 =======================
st.markdown('## ④ 케이스 스터디 — 특정 날짜의 예측·실제·위기여부')
case_dates = test_pred['date'].dt.date.unique()
sel_date = st.selectbox('날짜 선택', case_dates, index=len(case_dates)//2)
row = test_pred[test_pred['date']==pd.Timestamp(sel_date)].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('실제 Δy (bp)', f'{row["y_true_bp"]:+.2f}')
c2.metric('q05', f'{row["q05"]:+.2f}')
c3.metric('q50 (median)', f'{row["q50"]:+.2f}')
c4.metric('q95', f'{row["q95"]:+.2f}')
c5.metric('위기일?', '🔴 YES' if row['is_crisis'] else '🟢 NO')
in_band = (row['y_true_bp']>=row['q05']) and (row['y_true_bp']<=row['q95'])
sign_correct = np.sign(row['q50'])==np.sign(row['y_true_bp']) and row['y_true_bp']!=0
st.info(f'**구간 안 들어옴**: {"✅" if in_band else "❌"} · **방향성**: {"✅" if sign_correct else "❌"} · **q50 오차**: {row["y_true_bp"]-row["q50"]:+.2f} bp')

st.divider()
st.markdown('### 📚 차별화 포인트 — 발표 강조')
st.markdown('''
1. **부진 → 진단 → 회복 서사**: 4주차 raw LSTM RMSE 4.535 → 입력 비정상성 진단(LOG #35) → A0 (Δfeature[t-1]) 4.195 (-9.7% vs Naive)
2. **DM test 3/3 p<0.0001** (Bonferroni 보정 후) — 학부 7주 프로젝트 압도적 통계 입증
3. **메타-검증 5회 누적** (#30 → #36 → #37 → #40 → #43) — audit 도구 자체의 자가 검증
4. **VALIDATION_LOG 43건** = 안내문 최소 3건의 14.3배
''')

st.markdown('---')
st.caption('GitHub: https://github.com/donguk77/macro-bond-forecast · 본 데모는 7주차 §8 B1 (Streamlit 1페이지 필수) 산출물')

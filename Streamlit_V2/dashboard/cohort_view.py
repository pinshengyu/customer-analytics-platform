import pandas as pd
import altair as alt
import streamlit as st
from .data_loader import load_cohort
from .common import section_error
from .chart_helpers import show,rank_bar,detail,PALETTE

def render():
    st.header('Cohort 季度留存')
    st.caption('依首次購買季度分組，不使用 K-Means；Q0 為首次購買季度，Q1～Q4 為各後續季度有購買的比例。')
    try:data=load_cohort()
    except Exception as exc:section_error('Cohort',exc);return
    m=data['matrix'];qcols=[f'Q{i}' for i in range(5) if f'Q{i}' in m]
    d=m.melt(id_vars='CohortQuarter',value_vars=qcols,var_name='季度',value_name='Rate');d['標籤']=d.Rate.map(lambda v:'—' if pd.isna(v) else f'{v:.0f}%')
    base=alt.Chart(d).encode(x=alt.X('季度:O',sort=qcols,title='距首次購買季度',axis=alt.Axis(labelAngle=0)),y=alt.Y('CohortQuarter:O',sort='ascending',title='首次購買季度'),tooltip=['CohortQuarter:O','季度:O','標籤:N'])
    color=alt.condition('isValid(datum.Rate)',alt.Color('Rate:Q',scale=alt.Scale(domain=[0,100],scheme='blues'),legend=alt.Legend(title='留存率 (%)')),alt.value('#e5e7eb'))
    show((base.mark_rect(stroke='white',strokeWidth=2).encode(color=color)+base.mark_text(fontSize=13).encode(text='標籤:N',color=alt.condition(alt.datum.Rate>55,alt.value('white'),alt.value('#334155')))).properties(title='顏色越深，該季度留存越高',height=max(360,46*len(m))))
    st.caption('灰色「—」代表尚未到達該季度，並非 0%；各期留存不要求連續購買，因此可能回升。')
    options=m.CohortQuarter.astype(str).tolist();chosen=st.multiselect('比較留存曲線',options,default=options[-min(4,len(options)):],key='cohort_select')
    curve=d.loc[d.CohortQuarter.astype(str).isin(chosen)&d.Rate.notna()]
    if not curve.empty:show(alt.Chart(curve).mark_line(point=True,strokeWidth=3).encode(x=alt.X('季度:O',sort=qcols,title=None),y=alt.Y('Rate:Q',title='留存率 (%)',scale=alt.Scale(domain=[0,100])),color=alt.Color('CohortQuarter:N',scale=alt.Scale(range=PALETTE)),tooltip=['CohortQuarter:N','季度:O',alt.Tooltip('Rate:Q',format='.1f')]).properties(title='所選 Cohort 留存曲線',height=360))
    st.markdown('### 各批新客規模')
    rank_bar(data['size'],'CohortQuarter','CohortSize','各季度首次購買客戶數',unit='客戶數')
    detail(m,'季度留存矩陣')

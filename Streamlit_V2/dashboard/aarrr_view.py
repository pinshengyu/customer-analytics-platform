import pandas as pd
import altair as alt
import streamlit as st
from .data_loader import load_aarrr
from .common import money,number,percent,section_error
from .chart_helpers import show,rank_bar,detail,PALETTE
LABELS={'ActivationRate':'啟用率','RepeatPurchaseRate':'再購率','ReferralRate':'推薦率'}
def render():
    st.header('AARRR 客戶生命週期')
    try:data=load_aarrr()
    except Exception as exc:section_error('AARRR',exc);return
    mode=st.radio('分析範圍',['整體','分群比較','單一客群'],horizontal=True,key='aarrr_mode');c=data['cluster']
    st.caption('三項率的分母依原分析定義，各自衡量不同活動；再購率不等於跨季度留存率。')
    if mode=='分群比較':
        if not c.empty:
            ordered=c.sort_values('ClusterName')
            st.markdown('#### 各客群人數')
            for column, row in zip(st.columns(len(ordered)), ordered.itertuples()):
                column.metric(row.ClusterName, f'{number(row.AcquisitionCustomers)} 人')
            st.caption('人數採用本頁 AARRR 的客群獲客人數，與下方圖表使用相同分析範圍。')
        d=c.melt(id_vars='ClusterName',value_vars=list(LABELS),var_name='指標',value_name='Rate');d['指標']=d['指標'].map(LABELS)
        base=alt.Chart(d).encode(
            x=alt.X('ClusterName:N',title='客群',axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset('指標:N',sort=list(LABELS.values())),
            y=alt.Y('Rate:Q',title='比例（%）',scale=alt.Scale(domain=[0,110])),
            tooltip=[alt.Tooltip('ClusterName:N',title='客群'),'指標:N',alt.Tooltip('Rate:Q',title='比例（%）',format='.1f')])
        bars=base.mark_bar().encode(color=alt.Color('指標:N',scale=alt.Scale(domain=list(LABELS.values()),range=PALETTE[:3])))
        labels=base.mark_text(dy=-8).encode(text=alt.Text('Rate:Q',format='.1f'))
        show((bars+labels).properties(title='客群生命週期指標比較',height=400))
        detail(c,'AARRR客群明細');return
    row=data['overall'].iloc[0]
    if mode=='單一客群':
        name=st.selectbox('選擇客群',c.ClusterName.tolist(),key='aarrr_cluster');row=c.loc[c.ClusterName.eq(name)].iloc[0]
    for col,label,field,fmt in zip(st.columns(5),['獲客數','啟用率','再購率','推薦率','營收'],['AcquisitionCustomers',*LABELS,'Revenue'],[number,percent,percent,percent,money]):col.metric(label,fmt(row[field]))
    d=pd.DataFrame({'指標':list(LABELS.values()),'Rate':[row[k] for k in LABELS]})
    base=alt.Chart(d).encode(y=alt.Y('指標:N',title=None),x=alt.X('Rate:Q',scale=alt.Scale(domain=[0,110]),title='率 (%)'),tooltip=['指標:N',alt.Tooltip('Rate:Q',format='.1f')])
    show((base.mark_bar(color='#2563eb')+base.mark_text(align='left',dx=6).encode(text=alt.Text('Rate:Q',format='.1f'))).properties(title='核心活動參與率',height=300))
    detail(pd.DataFrame([row]),'AARRR摘要')

"""首頁分群摘要：正式客戶對照的唯一客戶分布。"""
import json
import altair as alt
import streamlit as st
from .data_loader import get_project_root, read_csv, segment
from .chart_helpers import show, PALETTE
from .common import section_error


def render():
    st.subheader('K-Means 客戶分群')
    try:
        root=get_project_root()
        model=json.loads((segment.segment_dir(root)/'K_Means_Model.json').read_text(encoding='utf-8'))
        customers=read_csv(str(segment.segment_dir(root)/'K_Means.csv'))
        names=read_csv(str(root/'K-means分群/Cluster/Cluster_Profile_Final.csv'))
        if customers.CustomerID.duplicated().any():raise ValueError('客戶對照包含重複編號')
        counts=customers.groupby('Cluster').CustomerID.nunique().rename('CustomerCount').reset_index()
        data=counts.merge(names[['Cluster','ClusterName']],on='Cluster',validate='one_to_one')
        if len(data)!=len(counts):raise ValueError('客群名稱不完整')
        data['Share']=data.CustomerCount/data.CustomerCount.sum()*100
    except (OSError,ValueError,KeyError) as exc:
        section_error('客戶分群摘要',exc);return
    st.write(f"依 RFM 三個維度分成 **{model['K']} 個客群**。")
    st.caption('R：距最近購買的天數；F：觀察期間訂單數；M：觀察期間累計消費金額。使用最近最多四季的特徵，經 log1p 與標準化後建立分群。')
    st.caption(f"群中心由 {model['TrainingSampleCount']:,} 筆合格訓練快照建立；下圖統計正式客戶對照中的 {len(customers):,} 位不重複客戶，不是訓練快照筆數，也不是單季預測人數。")
    domain=sorted(data.ClusterName.tolist())
    chart=alt.Chart(data).mark_arc().encode(theta=alt.Theta('CustomerCount:Q'),
        color=alt.Color('ClusterName:N',title='客群',scale=alt.Scale(domain=domain,range=PALETTE[:len(domain)])),
        tooltip=[alt.Tooltip('ClusterName:N',title='客群'),alt.Tooltip('CustomerCount:Q',title='人數',format=','),alt.Tooltip('Share:Q',title='占比（%）',format='.2f')])
    show(chart.properties(title='客群人數分布',height=360))
    st.dataframe(data[['ClusterName','CustomerCount','Share']].rename(columns={'ClusterName':'客群','CustomerCount':'人數','Share':'人數占比（%）'}),hide_index=True,width='stretch')

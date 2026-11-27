import pandas as pd
import streamlit as st
from .data_loader import load_operational, load_operational_quarterly_extensions
from .common import money,number,percent,section_error,developer_mode
from .chart_helpers import line,rank_bar,detail,brief


def _extension_notice(message: str | None) -> None:
    if not message:return
    st.info('季度客群資料目前不可用；上方原有營運 KPI 不受影響。')
    if developer_mode():st.caption(message)


def render():
    st.header('營運 KPI')
    try:data=load_operational()
    except Exception as exc:section_error('營運 KPI',exc);return

    # 原有頁面契約與呈現保留。
    mode=st.radio('分析範圍',['整體','分群比較','單一客群'],horizontal=True,key='op_mode')
    c=data['cluster'].copy();trend=data['quarterly']
    c['AverageCustomerSpend']=c.TotalRevenue/c.PurchasingCustomers.where(c.PurchasingCustomers.ne(0))
    if mode=='分群比較':
        if not c.empty:
            top=c.loc[c.TotalRevenue.idxmax()]
            share=100*top.PurchasingCustomers/c.PurchasingCustomers.sum() if c.PurchasingCustomers.sum() else 0
            brief(f"營收最高客群：{top.ClusterName}，營收貢獻 {percent(top.RevenueShare)}；購買客戶 {number(top.PurchasingCustomers)} 人，占全部購買客戶 {percent(share)}。")
        st.markdown('### 客群營收貢獻')
        rank_bar(c,'ClusterName','TotalRevenue','各客群累積營收',cluster=True)
        st.markdown('### 客群消費強度')
        rank_bar(c,'ClusterName','AOV','各客群平均訂單金額',cluster=True)
        st.markdown('### 客群平均個人消費總金額')
        rank_bar(c,'ClusterName','AverageCustomerSpend','各客群平均每人累積消費金額',cluster=True)
        st.caption('計算方式：分析期間的客群累積營收 ÷ 該群不重複購買客戶數；不同於平均每筆訂單金額。')
        st.markdown('### 跨季度變化')
        line(data['cluster_quarterly'],'Quarter','TotalRevenue','各客群季度營收趨勢',series='ClusterName')
        detail(c,'營運客群摘要')
    else:
        row=data['overall'].iloc[0]
        if mode=='單一客群':
            name=st.selectbox('選擇客群',c.ClusterName.tolist(),key='op_cluster');row=c.loc[c.ClusterName.eq(name)].iloc[0]
            trend=data['cluster_quarterly'].loc[data['cluster_quarterly'].ClusterName.eq(name)]
        for col,label,field,fmt in zip(st.columns(4),['總營收','訂單數','購買客戶','平均訂單金額'],['TotalRevenue','TotalOrders','PurchasingCustomers','AOV'],[money,number,number,money]):col.metric(label,fmt(row[field]))
        brief('最新季度營收季增率：'+percent(row.RevenueGrowthRate)+'。')
        st.markdown('### 營收趨勢')
        line(trend,'Quarter','TotalRevenue','季度營收趨勢')
        st.markdown('### 成長速度')
        rank_bar(trend,'Quarter','RevenueGrowthRate','各季度營收季增率',unit='季增率',percent=True,signed=True)
        detail(trend,'營運季度明細')

    # 新增功能獨立載入：失敗只影響此區，不改變上方原有分析。
    st.divider();st.subheader('季度關鍵客戶')
    st.caption('依所選季度的營收或訂單數，查看關鍵客戶。')
    ext=load_operational_quarterly_extensions();source=ext['cluster_quarterly'];customers=ext['customer_quarterly']
    if source is None:
        _extension_notice(ext['cluster_error']);return
    if source.empty:
        st.info('季度客群資料目前沒有資料列。');return
    quarters=sorted(source['Quarter'].dropna().astype(str).unique().tolist())
    if not quarters:
        st.info('季度客群資料沒有可用季度。');return
    q=st.selectbox('查看單季',quarters,index=len(quarters)-1,key='operational_kpi__local_quarter',help='只控制本區塊，不影響上方原有 KPI 與趨勢。')
    if customers is None:
        _extension_notice(ext['customer_error']);return
    if customers.empty:
        st.info('季度客戶資料目前沒有資料列。');return
    st.markdown('#### 關鍵客戶')
    a,b=st.columns(2)
    with a:metric=st.selectbox('排名依據',['營收貢獻','訂單數'],key='operational_kpi__customer_metric')
    with b:n=st.selectbox('顯示筆數',[3,5,10,20],key='operational_kpi__customer_n')
    field={'營收貢獻':'QuarterRevenue','訂單數':'QuarterOrders'}[metric]
    x=customers.loc[customers['Quarter'].astype(str).eq(q)].copy()
    if x.empty:
        st.info(f'{q} 沒有符合條件的客戶資料。');return
    top=x.sort_values(field,ascending=False).head(n)
    show=top[['CustomerID','ClusterName','QuarterRevenue','QuarterOrders','QuarterProfit','Recency','Frequency','Monetary']].copy()
    show.columns=['CustomerID','當季客群','當季營收','當季訂單','當季利潤','Recency','Frequency','Monetary']
    st.dataframe(show,width="stretch",hide_index=True)

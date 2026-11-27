import streamlit as st
from .data_loader import load_historical_clv, load_historical_clv_quarterly_extensions
from .common import money,number,section_error,developer_mode
from .chart_helpers import rank_bar,histogram,detail


def _extension_notice(message: str | None) -> None:
    if not message:return
    st.info('季度利潤分析資料目前不可用；上方累積 Historical CLV 不受影響。')
    if developer_mode():st.caption(message)


def render():
    st.header('歷史客戶價值')
    st.caption('歷史 CLV 為觀察期間累積實際利潤；未來價值請查看預測性分析。')
    try:data=load_historical_clv()
    except Exception as exc:section_error('Historical CLV',exc);return

    # 原有 Historical CLV 定義與頁面呈現保留。
    mode=st.radio('分析範圍',['整體','分群比較','單一客群'],horizontal=True,key='clv_mode');c=data['cluster'];d=data['distribution']
    if mode=='分群比較':
        st.markdown('### 每人歷史價值比較')
        rank_bar(c,'ClusterName','AverageHistoricalCLV','各客群平均每人 Historical CLV',signed=True,cluster=True)
        st.markdown('### 客群累積利潤比較')
        rank_bar(c,'ClusterName','TotalHistoricalProfit','各客群累積已實現利潤',signed=True,cluster=True)
        detail(c,'歷史CLV客群摘要')
    else:
        row=data['overall'].iloc[0]
        if mode=='單一客群':
            name=st.selectbox('選擇客群',c.ClusterName.tolist(),key='clv_cluster');row=c.loc[c.ClusterName.eq(name)].iloc[0];d=d.loc[d.ClusterName.eq(name)]
        for col,label,field,fmt in zip(st.columns(4),['購買客戶','歷史總利潤','平均每人價值','每人價值中位數'],['CustomerCount','TotalHistoricalProfit','AverageHistoricalCLV','MedianHistoricalCLV'],[number,money,money,money]):col.metric(label,fmt(row[field]))
        histogram(d.HistoricalCLV,'客戶價值集中在哪個區間？')
        st.caption('中位數代表典型客戶；平均值容易受到少數高價值客戶影響。負值代表累積虧損。')
        detail(d,'歷史CLV客戶明細')

    # 季度 Profit 是獨立擴充，不覆寫 Historical CLV = Σ Profit。
    st.divider();st.subheader('季度關鍵客戶')
    st.caption('依所選季度的實際利潤或營收，查看關鍵客戶。')
    ext=load_historical_clv_quarterly_extensions();quarterly=ext['quarterly'];customers=ext['customer_quarterly']
    if quarterly is None:
        _extension_notice(ext['quarterly_error']);return
    if quarterly.empty:
        st.info('季度利潤資料目前沒有資料列。');return
    cluster_q=quarterly.loc[quarterly['Scope'].eq('Cluster')].copy()
    quarters=sorted(cluster_q['Quarter'].dropna().astype(str).unique().tolist())
    if not quarters:
        st.info('季度利潤資料沒有可用的客群季度。');return
    q=st.selectbox('查看單季',quarters,index=len(quarters)-1,key='historical_clv__local_quarter',help='只控制本區塊，不影響上方累積 Historical CLV。')
    if customers is None:
        _extension_notice(ext['customer_error']);return
    if customers.empty:
        st.info('季度客戶資料目前沒有資料列。');return
    st.markdown('#### 關鍵客戶')
    a,b=st.columns(2)
    with a:metric=st.selectbox('排名依據',['利潤貢獻','營收貢獻'],key='historical_clv__customer_metric')
    with b:n=st.selectbox('顯示筆數',[3,5,10,20],key='historical_clv__customer_n')
    field={'利潤貢獻':'QuarterProfit','營收貢獻':'QuarterRevenue'}[metric]
    x=customers.loc[customers['Quarter'].astype(str).eq(q)].copy()
    if x.empty:
        st.info(f'{q} 沒有符合條件的客戶資料。');return
    top=x.sort_values(field,ascending=False).head(n)
    show=top[['CustomerID','ClusterName','QuarterProfit','QuarterRevenue','QuarterOrders','Recency','Frequency','Monetary']].copy()
    show.columns=['CustomerID','當季客群','當季利潤','當季營收','當季訂單','Recency','Frequency','Monetary']
    st.dataframe(show,width="stretch",hide_index=True)

"""以現行群中心比較個別客戶的季度快照。"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import quarterly_data as quarterly
import segment_contract as segment
from .common import csv_download
from .data_loader import get_project_root
from .chart_helpers import PALETTE, show


@st.cache_data(show_spinner=False, max_entries=2, ttl=300)
def load_history(root_path: str, signature: tuple) -> pd.DataFrame:
    root = Path(root_path)
    segment.context(root)
    columns = ['CustomerID', 'TargetQuarter', 'ObservationStart', 'ObservationEnd',
               'Recency', 'Frequency', 'Monetary']
    frames = [quarterly.read_samples(root, name)[columns] for name in
              ['Quarterly_Samples', 'Quarterly_Holdout', 'Quarterly_Forecast']]
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['CustomerID', 'TargetQuarter']).any():
        raise ValueError('客戶季度快照重複，請重新執行完整流程。')
    data = quarterly.attach_clusters(root, data)
    data['Quarter'] = pd.to_datetime(data.ObservationEnd).dt.to_period('Q').astype(str)
    data['Group'] = data.Cluster.astype(str) + ' · ' + data.ClusterName
    return data


def customer_timeline(data: pd.DataFrame, customer: str) -> pd.DataFrame:
    selected = data.loc[data.CustomerID.eq(customer)].copy()
    quarters = pd.period_range(data.Quarter.min(), data.Quarter.max(), freq='Q').astype(str)
    selected = selected.set_index('Quarter').reindex(quarters).rename_axis('Quarter').reset_index()
    previous = selected.Cluster.shift()
    changed = selected.Cluster.notna() & previous.notna() & selected.Cluster.ne(previous)
    selected['Change'] = '無快照'
    present = selected.Cluster.notna()
    selected.loc[present, 'Change'] = '首次／中斷後觀察'
    selected.loc[present & previous.notna(), 'Change'] = '維持原群'
    selected.loc[changed, 'Change'] = '換群'
    selected['PreviousGroup'] = selected.Group.shift().where(previous.notna(), '—')
    selected['Group'] = selected.Group.fillna('無快照')
    selected['CustomerID'] = customer
    return selected


def render() -> None:
    st.subheader('個別客戶分群變化')
    st.caption('季度代表觀察期末，例如 2025Q4 的快照用於 2026Q1 預測。')
    st.info('各季 RFM 均套用目前同一組 K-Means 群中心，以比較購買行為變化；不是每季重新訓練，也不代表當時已上線的分群。其他歷史診斷圖表仍使用固定客群對照。')
    root = get_project_root()
    folder = segment.segment_dir(root)
    files = [folder / name for name in ['Quarterly_Samples.csv', 'Quarterly_Holdout.csv',
             'Quarterly_Forecast.csv', 'K_Means_Model.json', 'Segmentation_Manifest.json']]
    files += [root / 'K-means分群' / 'Cluster' / name for name in
              ['Cluster_Profile_Final.csv', 'Naming_Manifest.json']]
    try:
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        data = load_history(str(root), signature)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'無法讀取客戶分群歷程：{exc}')
        return
    if data.empty:
        st.info('目前沒有客戶季度快照。')
        return
    customer_ids = sorted(data.CustomerID.unique(),
                          key=lambda value: (0, int(value), value) if value.isdecimal() else (1, value))
    customer = st.selectbox('客戶編號（可輸入搜尋）', customer_ids,
                            key='diag_customer_history')
    timeline = customer_timeline(data, customer)
    observed = timeline.loc[timeline.Cluster.notna()]
    cols = st.columns(3)
    cols[0].metric('有快照的季度', len(observed))
    cols[1].metric('相鄰季度換群次數', int(timeline.Change.eq('換群').sum()))
    cols[2].metric('最近觀察客群', observed.iloc[-1].Group)
    groups = data[['Cluster', 'Group']].drop_duplicates().sort_values('Cluster').Group.tolist()
    chart = alt.Chart(timeline).mark_circle(size=240).encode(
        x=alt.X('Quarter:O', title='觀察季度末', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Group:N', title='所屬客群', sort=groups + ['無快照']),
        color=alt.Color('Group:N', scale=alt.Scale(domain=groups + ['無快照'],
                       range=PALETTE[:len(groups)] + ['#94a3b8']), legend=None),
        tooltip=[alt.Tooltip('Quarter:N', title='觀察季度'), alt.Tooltip('Group:N', title='客群'),
                 alt.Tooltip('Change:N', title='變動'), alt.Tooltip('Recency:Q', title='距最近購買（天）'),
                 alt.Tooltip('Frequency:Q', title='訂單數（筆）'),
                 alt.Tooltip('Monetary:Q', title='累計消費（元）', format=',.2f')]
    ).properties(height=380, title=f'客戶 {customer} 的季度分群歷程')
    show(chart)
    st.caption('RFM 使用各季度末往前最多四季資料。無快照可能是尚未購買，或觀察窗口內沒有購買；不視為流失或換群。相鄰季度均有快照才計算換群。')
    changes = timeline.loc[timeline.Change.eq('換群')]
    if changes.empty:
        st.info('目前沒有可確認的相鄰季度換群紀錄。')
    else:
        st.write('**換群紀錄**')
        for row in changes.itertuples():
            st.write(f'{row.Quarter}：{row.PreviousGroup} → {row.Group}')
    detail = timeline[['CustomerID', 'Quarter', 'Group', 'Change', 'ObservationStart',
                       'ObservationEnd', 'Recency', 'Frequency', 'Monetary']].rename(columns={
        'CustomerID': '客戶編號', 'Quarter': '觀察季度', 'Group': '客群', 'Change': '變動',
        'ObservationStart': '觀察起日', 'ObservationEnd': '觀察迄日',
        'Recency': '距最近購買（天）', 'Frequency': '訂單數（筆）', 'Monetary': '累計消費（元）'})
    st.dataframe(detail, hide_index=True, width='stretch')
    csv_download(detail, f'Customer_{customer}_Segment_History.csv', 'diag_customer_history_download')

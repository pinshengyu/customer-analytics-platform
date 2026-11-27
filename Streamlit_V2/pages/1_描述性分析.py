import streamlit as st

from dashboard.operational_kpi_view import render as render_operational
from dashboard.aarrr_view import render as render_aarrr
from dashboard.historical_clv_view import render as render_clv
from dashboard.cohort_view import render as render_cohort
from dashboard.customer_segments_view import render as render_customer_segments

st.set_page_config(
    page_title="描述性分析",
    page_icon="📈",
    layout="wide",
)

from dashboard.common import sidebar_status, require_ready
sidebar_status()

st.title("描述性分析")
st.caption("營運指標 · 客戶生命週期 · 歷史客戶價值 · 季度留存")

require_ready()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["營運 KPI", "AARRR", "歷史客戶價值", "Cohort 留存", "客戶分群變化"], key='descriptive_tabs', on_change='rerun'
)

with tab1:
    if tab1.open: render_operational()

with tab2:
    if tab2.open: render_aarrr()

with tab3:
    if tab3.open: render_clv()

with tab4:
    if tab4.open: render_cohort()

with tab5:
    if tab5.open: render_customer_segments()

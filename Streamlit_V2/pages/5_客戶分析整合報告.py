import streamlit as st
from dashboard.common import sidebar_status, require_ready
from dashboard.report_view import render

st.set_page_config(page_title='客戶分析整合報告', page_icon='📄', layout='wide')
sidebar_status()
st.title('客戶分析整合報告')
st.caption('彙整本次分析的重點結果、待驗證行動與限制。')
require_ready()
render()

import streamlit as st
from dashboard.common import sidebar_status, require_ready, developer_mode
from dashboard.predictive_view import render

st.set_page_config(page_title='模型結果分析', page_icon='📋', layout='wide')
sidebar_status()
if not developer_mode():
    st.info('此頁僅在開發者模式使用。')
    st.stop()
st.title('模型結果分析')
st.caption('歷史回測、保留測試集評估與 CLV 模型比較。')
require_ready()
render(evaluation=True)

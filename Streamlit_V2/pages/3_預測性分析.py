import streamlit as st
from dashboard.common import sidebar_status, require_ready
from dashboard.predictive_view import render
st.set_page_config(page_title='預測性分析',page_icon='📉',layout='wide')
sidebar_status()
st.title('預測性分析')
require_ready()
render()

import streamlit as st
from dashboard.common import sidebar_status, require_ready
from dashboard.diagnostic_view import render

st.set_page_config(page_title="診斷性分析", page_icon="🔎", layout="wide")
sidebar_status()
st.title("診斷性分析")
require_ready()
render()

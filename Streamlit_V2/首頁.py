import pandas as pd
import streamlit as st
from dashboard.common import sidebar_status, require_ready, money, number, developer_mode
from dashboard.data_loader import load_diagnostic, load_operational
from dashboard.import_view import render as render_import
from dashboard.segmentation_overview import render as render_segmentation

st.set_page_config(page_title="客戶分析與智慧決策支援平台", page_icon="📊", layout="wide")
sidebar_status()
st.title("客戶分析與智慧決策支援平台")
st.caption("基於商業智慧之客戶分析與智慧決策支援平台 · v5")

left, middle, right = st.columns(3)
with left:
    st.subheader("描述性分析")
    st.write("掌握營收、客戶生命週期、歷史價值與季度留存表現。")
    st.page_link("pages/1_描述性分析.py", label="開啟描述性分析", icon="📈")
with middle:
    st.subheader("診斷性分析")
    st.write("找出指標變化與主要影響來源，協助定位需要進一步檢查的問題。")
    st.page_link("pages/2_診斷性分析.py", label="開啟診斷性分析", icon="🔎")
with right:
    st.subheader("預測性分析")
    st.write("查看下一季營收、客戶流失風險與下一季客戶價值預測。")
    st.page_link("pages/3_預測性分析.py", label="開啟預測性分析", icon="🔮")
    if developer_mode(): st.page_link("pages/4_模型結果分析.py", label="查看模型結果分析", icon="🧪")

st.page_link('pages/5_客戶分析整合報告.py', label='查看客戶分析整合報告', icon='📄')
st.divider()
render_import()
st.divider()
require_ready(show_link=False)
render_segmentation()
st.divider()
try:
    summary = load_diagnostic("summary").sort_values("Quarter")
    if summary.empty:
        raise ValueError("季度摘要尚無資料")
    latest = summary.iloc[-1]
    st.subheader(f"最新季度：{latest['Quarter']}")
    cols = st.columns(4)
    for col, label, field, formatter in [
        (cols[0], "營收", "Revenue", money),
        (cols[1], "訂單數", "Orders", number),
        (cols[2], "購買客戶數", "Buyers", number),
        (cols[3], "AOV", "AOV", money),
    ]:
        col.metric(label, formatter(latest.get(field)))
    try:
        findings = load_diagnostic("findings")
        selected = findings.loc[findings["Period"].astype(str).eq(str(latest["Quarter"]))]
        abnormal = selected.loc[selected["Anomaly"].astype(str).str.lower().eq("true")]
        if not abnormal.empty:
            st.warning("最新季度異常：" + "、".join(abnormal["KPI"].astype(str)))
        elif not selected.empty and selected["ComparisonStatus"].eq("Comparable").all():
            st.info("最新季度沒有超過設定下降門檻的 KPI。")
        else:
            st.info("最新季度尚無完整的可比較異常判斷。")
    except (FileNotFoundError, KeyError):
        st.info("先執行新版診斷分析程式以產生異常判斷。")
except (FileNotFoundError, ValueError, KeyError):
    try:
        overall = load_operational()["overall"].iloc[0]
        st.subheader("歷史營運總覽")
        cols = st.columns(4)
        for col, label, field, formatter in [
            (cols[0], "總營收", "TotalRevenue", money),
            (cols[1], "訂單數", "TotalOrders", number),
            (cols[2], "購買客戶數", "PurchasingCustomers", number),
            (cols[3], "AOV", "AOV", money),
        ]:
            col.metric(label, formatter(overall.get(field)))
    except (FileNotFoundError, ValueError, KeyError, IndexError):
        st.info("尚未讀到分析輸出。請將本資料夾放在專案根目錄內，並先執行對應分析程式。")

st.caption("完成資料處理後，可從各分析頁查看結果與整合報告。")

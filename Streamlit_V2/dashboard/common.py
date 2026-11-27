from __future__ import annotations
import pandas as pd
import streamlit as st


def money(value) -> str:
    return "—" if pd.isna(value) else f"${float(value):,.0f}"

def number(value) -> str:
    return "—" if pd.isna(value) else f"{int(round(float(value))):,}"

def percent(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):,.2f}%"

def section_error(title: str, exc: Exception) -> None:
    st.error(f"{title}資料暫時無法顯示。")
    if developer_mode():
        st.caption(str(exc))

def developer_mode() -> bool:
    return bool(st.session_state.get("developer_mode", False))

def session_data_ready() -> bool:
    return bool(st.session_state.get("session_data_ready", False))

def sidebar_status() -> None:
    from .data_loader import get_project_root
    with st.sidebar:
        st.page_link("首頁.py", label="首頁", icon="🏠")
        st.page_link("pages/1_描述性分析.py", label="描述性分析", icon="📈")
        st.page_link("pages/2_診斷性分析.py", label="診斷性分析", icon="🔎")
        st.page_link("pages/3_預測性分析.py", label="預測性分析", icon="🔮")
        st.page_link("pages/5_客戶分析整合報告.py", label="客戶分析整合報告", icon="📄")
        st.divider()
        st.toggle("開發者模式", key="developer_mode", help="顯示模型評估、系統狀態、路徑與執行紀錄。")
        if developer_mode():
            st.page_link("pages/4_模型結果分析.py", label="模型結果分析", icon="🧪")
            if st.button("清除快取", key="refresh_data", width="stretch"):
                st.cache_data.clear(); st.rerun()
            try:
                r=get_project_root()
                st.caption(f"專案位置：{r}")
                from pipeline_runtime import current_status, publishing_blocked, published_data_available
                status=current_status(r)
                st.caption("批次狀態："+str(status.get("Status","尚未執行")))
                if publishing_blocked(r): st.warning("結果更新中或需復原，請回首頁查看執行狀態。")
                if published_data_available(r):
                    from .data_loader import segment
                    ctx=segment.context(r)
                    st.caption("預測季度："+ctx["TargetQuarter"])
                    st.caption("分群版本："+ctx["SegmentVersion"][:10])
            except (FileNotFoundError,ValueError,KeyError) as exc:
                st.warning(str(exc))
        else:
            st.caption("分析結果會在本次操作完成資料處理後顯示。")

def csv_download(data: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button("下載目前表格 CSV", data.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv", key=key)

def require_ready(show_link=True) -> None:
    """每個新網頁工作階段預設不載入既有分析結果。"""
    from .data_loader import get_project_root
    from pipeline_runtime import publishing_blocked,current_status
    root=get_project_root()
    if not session_data_ready():
        state=current_status(root)
        job=st.session_state.get("session_pipeline_job")
        if job and state.get("JobID")==job and state.get("Status")=="completed":
            st.session_state["session_data_ready"]=True
        else:
            st.info("目前尚未載入本次分析資料。請先從首頁匯入資料並完成分析。")
            if show_link: st.page_link("首頁.py",label="前往首頁匯入資料",icon="📥")
            st.stop()
    if publishing_blocked(root):
        st.info("分析結果正在更新，完成後即可查看。")
        st.stop()

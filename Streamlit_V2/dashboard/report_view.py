from pathlib import Path
import streamlit as st
from .data_loader import get_project_root
import pipeline_runtime as pipeline

@st.cache_data(show_spinner=False,max_entries=8)
def report_bytes(path,modified_ns):return Path(path).read_bytes()

def render():
    root=get_project_root()
    state=pipeline.current_status(root)
    if pipeline.publishing_blocked(root):st.caption('結果正在更新或需要復原，暫停提供報告。');return
    if not pipeline.published_data_available(root):st.caption('首次匯入並完成全流程後提供報告。');return
    report=root/'報告結果'
    summary=pipeline.read_json(report/'報告摘要.json')
    if not summary:st.caption('完整流程完成後提供報告。');return
    st.caption('報告批次：'+summary.get('JobID',''))
    if state and summary.get('JobID')!=state.get('JobID'):st.info('下方是前一次成功執行的報告；本次尚未完成。')
    for col,name,mime in zip(st.columns(3),['客戶分析整合報告.html','客戶分析整合報告.md','報告摘要.json'],['text/html','text/markdown','application/json']):
        file=report/name
        if file.exists():col.download_button('下載 '+{'html':'圖表報告','md':'文字報告','json':'摘要資料'}[file.suffix[1:]],report_bytes(str(file),file.stat().st_mtime_ns),name,mime,key='report_'+file.suffix)
    st.caption('HTML 報告可離線用瀏覽器開啟；需要 PDF 時可使用瀏覽器列印另存 PDF。')
    st.divider()
    html=report/'客戶分析整合報告.html'
    if html.is_file():
        st.iframe(report_bytes(str(html),html.stat().st_mtime_ns).decode('utf-8-sig'),height=950)
    else:
        st.warning('尚未找到圖表報告，請先在 首頁完成整合報告流程。')

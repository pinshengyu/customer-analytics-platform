"""資料匯入與執行頁；UI 不直接執行模型，也不覆寫工作中的資料。"""
from pathlib import Path
import sys
import uuid
import shutil
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import pipeline_runtime as pipeline
from .data_loader import get_project_root
from .common import developer_mode

STATUS={'waiting':'待執行','queued':'準備中','running':'執行中','publishing':'更新結果','completed':'完成','failed':'失敗','interrupted':'中斷'}

def stage_uploads(root,uploads):
    names=[x.name for x in uploads]
    if len(names)!=len(set(names)):raise ValueError('上傳檔名重複，請每張表只選一份。')
    missing=set(pipeline.SCHEMAS)-set(names);extra=set(names)-set(pipeline.SCHEMAS)
    if missing:raise ValueError('請一次提供全部六張表；缺少：'+', '.join(sorted(missing)))
    if extra:raise ValueError('不支援的檔名：'+', '.join(sorted(extra)))
    folder=pipeline.jobs_dir(root)/'待匯入'/uuid.uuid4().hex;folder.mkdir(parents=True)
    for uploaded in uploads:
        with (folder/uploaded.name).open('wb') as handle:handle.write(uploaded.getbuffer())
    return folder

@st.fragment(run_every='3s')
def progress(root):
    state=pipeline.current_status(root)
    if not st.session_state.get('session_pipeline_job') and not developer_mode():
        st.caption('尚未執行本次資料分析。')
        return
    if not state:st.caption('尚未執行匯入流程。');return
    status=state.get('Status')
    previous=st.session_state.get('pipeline_phase')
    st.session_state['pipeline_phase']=(state['JobID'],status)
    if previous and previous[0]==state['JobID'] and previous[1] in pipeline.ACTIVE and status not in pipeline.ACTIVE:st.rerun()
    steps=state.get('Steps',[]);finished=sum(x['Status']=='completed' for x in steps)
    st.subheader('執行進度')
    st.caption('批次：'+state['JobID']);st.progress(finished/max(1,len(steps)),text=f"{STATUS.get(status,status)} · 已完成 {finished}／{len(steps)} 步")
    if status=='completed':
        if st.session_state.get('session_pipeline_job')==state.get('JobID'):
            st.session_state['session_data_ready']=True
        st.success('資料處理完成，可以開始查看分析結果。')
        if st.session_state.get('cleared_pipeline')!=state['JobID']:st.cache_data.clear();st.session_state['cleared_pipeline']=state['JobID']
        st.page_link('pages/3_預測性分析.py',label='查看更新後的預測圖表')
    elif status in ('failed','interrupted'):st.error(state.get('Message','流程停止'))
    else:st.info(state.get('Message',''))
    if not developer_mode():
        if status in ('failed','interrupted'):
            st.caption('處理未完成，請重新執行；如需查看錯誤內容請開啟開發者模式。')
        return
    groups=[steps[:4],steps[4:9],steps[9:]]
    for col,title,items in zip(st.columns(3),['資料與分群','描述與診斷','預測與報告'],groups):
        with col:
            st.markdown('**'+title+'**')
            for item in items:
                icon={'completed':'✓','running':'⏳','failed':'✕'}.get(item['Status'],'·');st.write(icon+' '+item['Name']+'　'+STATUS.get(item['Status'],item['Status']))
    if status in ('failed','interrupted') and pipeline.publishing_blocked(root):
        if st.button('復原中斷的結果更新',key='recover_pipeline'):
            try:pipeline.recover_project(root);st.rerun()
            except Exception as exc:st.error(str(exc))
    if status in ('failed','interrupted'):
        if st.button('重新執行這批資料',key='retry_pipeline'):
            try:
                folder=pipeline.safe_job(root,state['JobID']);request=pipeline.read_json(folder/'request.json');pipeline.submit(root,folder/'inputs',request['Config']);st.rerun()
            except Exception as exc:st.error(str(exc))
    with st.expander('查看執行紀錄與錯誤內容'):
        st.code(pipeline.tail_log(root,state['JobID']) or '尚無輸出',language='text')
        log=pipeline.safe_job(root,state['JobID'])/'worker.log'
        if log.exists():st.download_button('下載執行紀錄',log.read_bytes(),'執行紀錄_'+state['JobID']+'.txt',key='download_pipeline_log')
    st.page_link('pages/5_客戶分析整合報告.py',label='查看客戶分析整合報告',icon='📄')

def render():
    root=get_project_root();st.header('資料匯入與執行')
    st.caption('匯入完整資料後，系統會自動完成分析並產生最新結果。')
    active=pipeline.current_status(root).get('Status') in pipeline.ACTIVE
    from quarterly_plan import QuarterPlan,load_plan
    current=load_plan(root)
    year_col,quarter_col=st.columns(2)
    year=year_col.number_input('預測年份',min_value=2003,max_value=2099,value=int(current.target[:4]),step=1,disabled=active,key='target_year')
    quarter=quarter_col.selectbox('預測季度',['Q1','Q2','Q3','Q4'],index=int(current.target[-1])-1,disabled=active,key='target_quarter')
    plan=QuarterPlan(f'{year}{quarter}')
    st.info(f'本次將分析 {plan.history_start}～{plan.holdout} 的完整資料，並預測 {plan.target}。')
    source_mode='上傳新的六張原始資料表'
    if developer_mode():
        source_mode=st.radio('資料來源',['上傳新的六張原始資料表','使用目前原始資料重新執行'],horizontal=True,disabled=active,key='pipeline_source')
    uploads=[]
    if source_mode.startswith('上傳'):
        uploads=st.file_uploader('一次選取六個 CSV（UTF-8）',type=['csv'],accept_multiple_files=True,disabled=active,key='pipeline_uploads')
        required=set(pipeline.SCHEMAS);received={x.name for x in uploads};st.caption(f'已選取 {len(required&received)}／6 張必要資料表')
        with st.expander('必要檔名與欄位格式'):
            for name,columns in pipeline.SCHEMAS.items():st.markdown('**'+name+'**');st.code(', '.join(columns),language='text')
    else:st.caption('來源：原始資料/RAW_Data；使用相同流程重新產生全部結果。')
    complete=st.checkbox(f'確認 {plan.history_start}～{plan.holdout} 的12季資料皆已完整載入',disabled=active,key='complete_history_'+plan.target)
    if developer_mode(): st.caption(f'{plan.history_start}～{plan.history_end} 按時間與筆數切 70/15/15；{plan.holdout} 獨立回測，{plan.target} 預測。第一季為觀察暖身；特徵最多取最近4季。')
    st.caption('可上傳更長期間的資料；系統會依選定季度擷取所需期間。請確認各季資料皆已完整。')
    llm=False; model='llama3.2:3b'; url='http://localhost:11434'
    if developer_mode():
        with st.expander('報告與本機 LLM 設定'):
            llm=st.checkbox('在整合報告加入 LLM 文字解讀',value=False,disabled=active,key='pipeline_llm')
            model=st.text_input('Ollama 模型',value=model,disabled=active,key='pipeline_model')
            url=st.text_input('本機 Ollama 位址',value=url,disabled=active,key='pipeline_url')
            st.caption('客群命名固定使用 LLM；此區為系統執行設定。')
    if developer_mode(): st.caption('完成後會更新目前原始表與分析結果；舊資料保存在執行紀錄的批次備份。')
    if st.button('檢查資料並執行完整流程',type='primary',disabled=active or not complete,key='start_pipeline'):
        source=None
        try:
            with st.spinner('正在檢查資料並啟動分析…'):
                source=stage_uploads(root,uploads) if source_mode.startswith('上傳') else root/'原始資料'/'RAW_Data'
                config={'TargetQuarter':plan.target,'CompleteHistory':complete,'LLMReport':llm,'Model':model.strip(),'OllamaURL':url.strip()}
                jid=pipeline.submit(root,source,config)
                st.session_state['session_pipeline_job']=jid
                st.session_state['session_data_ready']=False
            st.rerun()
        except Exception as exc:st.error(str(exc))
        finally:
            if source is not None and source_mode.startswith('上傳'):shutil.rmtree(source,ignore_errors=True)
    if active: st.caption('資料正在處理中，完成後即可查看本次分析結果。')
    progress(root)

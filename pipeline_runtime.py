"""可搬移的單一專案流程：匯入檢查、背景工作、發布及可復原備份。"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from urllib.request import urlopen
from urllib.parse import urlparse

SCHEMAS={
 'Customer_Dim.csv':['CustomerID','Gender','Age','City','RegisterDate','ReferralUserID','AcquisitionSource'],
 'Product_Dim.csv':['ProductID','Category','Brand','Price','Cost','PopularityWeight','LaunchDate','ShelfLifeDays'],
 'Order_Fact.csv':['OrderID','CustomerID','ProductID','OrderDate','Quantity','UnitPrice','GrossAmount','DiscountRate','DiscountAmount','Amount','Channel','StoreID','CampaignID'],
 'Event_Fact.csv':['EventID','CustomerID','EventDate','EventType','Channel','SessionID'],
 'Referral_Table.csv':['ReferralID','ReferrerCustomerID','NewCustomerID','ReferralDate'],
 'Marketing_Campaign_Table.csv':['CampaignID','CampaignName','Channel','Budget','StartDate','EndDate','DiscountRate','CampaignType'],
}
DATE_COLUMNS={'Customer_Dim.csv':['RegisterDate'],'Product_Dim.csv':['LaunchDate'],'Order_Fact.csv':['OrderDate'],'Event_Fact.csv':['EventDate'],'Referral_Table.csv':['ReferralDate'],'Marketing_Campaign_Table.csv':['StartDate','EndDate']}
ACTIVE={'queued','running','publishing'}
OUTPUT_DIRS=['ETL_Output','原始資料/Data_Warehouse','K-means分群/K_Means','K-means分群/Cluster','描述性分析/Operational_KPI','描述性分析/AARRR','描述性分析/Historical_CLV','描述性分析/Cohort','描述性分析/Quarterly_Clustering']
# 每次均使用原來的輸出位置；診斷資料夾內的 .py 不屬於輸出。
OUTPUT_DIRS=OUTPUT_DIRS+['預測性分析/預測結果/整體季度營收趨勢預測','預測性分析/預測結果/流失預測','預測性分析/預測結果/CLV價值預測','報告結果']
SCRIPT_STEPS=[
 ('ETL 資料清理','原始資料/ETL.py'),('資料倉儲','原始資料/Data_Warehouse.py'),
 ('季度訓練集分群','K-means分群/K_Means_Clustering.py'),('LLM 客群命名','K-means分群/Cluster_Profile.py'),
 ('營運 KPI','描述性分析/Operational_KPI.py'),('AARRR','描述性分析/AARRR.py'),
 ('歷史 CLV','描述性分析/Historical_CLV.py'),('季度固定群中心分析','描述性分析/Quarterly_Descriptive.py'),('季度 Cohort','描述性分析/Cohort_Analysis.py'),
 ('診斷分析','診斷性分析/診斷性分析.py'),('季度營收預測','預測性分析/整體季度營收趨勢預測.py'),
 ('流失預測','預測性分析/流失預測.py'),('CLV 價值預測','預測性分析/CLV價值預測.py'),
 ('整合報告','產生整合報告.py')]

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def jobs_dir(root):return Path(root)/'執行紀錄'
def read_json(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return {} if default is None else default

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    try:replace_path(temp,path)
    finally:
        if temp.exists():temp.unlink()

def replace_path(source,target):
    # Windows 的短暫讀取鎖不應讓整個發布失敗。
    for attempt in range(16):
        try:os.replace(source,target);return
        except PermissionError:
            if attempt==15:raise
            time.sleep(.2)

@contextmanager
def project_lock(root,wait_seconds=0):
    """OS 檔案鎖：程序結束自動釋放，Windows 與 POSIX 皆可用。"""
    folder=jobs_dir(root);folder.mkdir(parents=True,exist_ok=True)
    handle=open(folder/'pipeline.lock','a+b');handle.seek(0,2)
    if handle.tell()==0:handle.write(b'0');handle.flush()
    deadline=time.monotonic()+wait_seconds;locked=False
    try:
        while True:
            try:
                handle.seek(0)
                if os.name=='nt':
                    import msvcrt;msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl;fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True;break
            except OSError:
                if time.monotonic()>=deadline:raise RuntimeError('目前有流程正在執行，請等待完成。')
                time.sleep(.1)
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name=='nt':
                import msvcrt;msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl;fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        handle.close()

def process_alive(pid):
    if not isinstance(pid,int) or pid<=0:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:return ctypes.get_last_error()==5
        code=wintypes.DWORD()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:
        os.kill(pid,0)
        stat=Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().rsplit(')',1)[-1].split()[0]=='Z':return False
        return True
    except ProcessLookupError:return False
    except PermissionError:return True

def safe_job(root,job_id):
    if not job_id or any(c not in '0123456789abcdef-' for c in job_id):raise ValueError('無效的執行批次。')
    return jobs_dir(root)/job_id

def current_status(root):
    pointer=read_json(jobs_dir(root)/'current.json');jid=pointer.get('JobID')
    if not jid:return {}
    folder=safe_job(root,jid);state=read_json(folder/'status.json')
    process=read_json(folder/'process.json')
    if state.get('Status') in ACTIVE:
        created=datetime.fromisoformat(state['CreatedAt']);age=(datetime.now(timezone.utc)-created).total_seconds()
        if not process_alive(process.get('PID')) and (process or age>15):
            state={**state,'Status':'interrupted','Message':'背景程序已停止；請重新執行此批次。若在更新結果時停止，系統將先復原。'}
    return state

def published_data_available(root):
    # 單獨重跑模組可能留下同批次編號的舊報告，必須核對分群與目標季度。
    import segment_contract as segment
    try:
        marker=read_json(jobs_dir(root)/'已發布批次.json')
        if not marker.get('JobID'):return False
        report=read_json(Path(root)/'報告結果'/'報告摘要.json')
        return report.get('JobID')==marker['JobID'] and marker.get('Context')==report.get('Context')==segment.context(root)
    except (OSError,ValueError,KeyError):return False

def publishing_blocked(root):
    s=current_status(root)
    if s.get('Status')=='publishing':return True
    if s.get('JobID'):
        return bool(read_json(safe_job(root,s['JobID'])/'publish.json').get('Pending'))
    return False

def validate_inputs(folder,target_quarter='2026Q1'):
    """分塊讀取全部 CSV，確認實際 ETL 格式，避免只檢查前幾列。"""
    import pandas as pd
    from quarterly_plan import QuarterPlan
    plan=QuarterPlan(target_quarter)
    folder=Path(folder);result=[]
    for name,cols in SCHEMAS.items():
        path=folder/name
        if not path.is_file():raise ValueError('缺少原始資料：'+name)
        with path.open(encoding='utf-8-sig',newline='') as handle:
            header=next(csv.reader(handle),[])
        if len(header)!=len(set(header)):raise ValueError(f'{name} 有重複欄名。')
        missing=set(cols)-set(header)
        if missing:raise ValueError(f'{name} 缺少欄位：'+', '.join(sorted(missing)))
        rows=0;low=None;high=None;quarters=set()
        for chunk in pd.read_csv(path,encoding='utf-8-sig',chunksize=50000,low_memory=False):
            rows+=len(chunk)
            for col in DATE_COLUMNS[name]:
                dates=pd.to_datetime(chunk[col],errors='coerce')
                if dates.isna().any():raise ValueError(f'{name} 的 {col} 有空白或無效日期；請修正再匯入。')
                if name=='Order_Fact.csv':
                    low=min(low,dates.min()) if low is not None else dates.min();high=max(high,dates.max()) if high is not None else dates.max();quarters.update(dates.dt.to_period('Q').astype(str))
        if rows==0 and name!='Referral_Table.csv':raise ValueError(f'{name} 沒有資料列。')
        item={'File':name,'Rows':rows,'SizeMB':round(path.stat().st_size/1024**2,2)}
        if low is not None:
            item.update(DateStart=str(low.date()),DateEnd=str(high.date()))
            needed=set(plan.quarters)
            if not needed.issubset(quarters):raise ValueError(f'預測 {plan.target} 需要 {plan.history_start}～{plan.holdout} 的完整12季資料；缺少季度：'+ '、'.join(sorted(needed-quarters)))
            item.update(RequiredStart=plan.start_date,RequiredEnd=plan.score_end,TargetQuarter=plan.target)
        result.append(item)
    return result

def input_digest(folder):
    hashes={}
    for name in SCHEMAS:
        h=hashlib.sha256()
        with (Path(folder)/name).open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        hashes[name]=h.hexdigest()
    return hashes

def ollama_url(value):
    value=value.rstrip('/');p=urlparse(value)
    if p.scheme not in ('http','https') or p.hostname not in ('localhost','127.0.0.1','::1') or p.username or p.password or p.query or p.fragment:raise ValueError('目前使用本機 Ollama，網址須為 http(s)://localhost 或 127.0.0.1 的服務位置。')
    return value

def preflight(root,config):
    missing=[p for _,p in SCRIPT_STEPS if not (Path(root)/p).is_file()]
    if missing:raise ValueError('缺少流程程式：'+', '.join(missing))
    deps=[m for m in ['pandas','numpy','sklearn','xgboost','matplotlib','streamlit','altair'] if importlib.util.find_spec(m) is None]
    if deps:raise ValueError('目前 Python 環境缺少套件：'+', '.join(deps)+'；請先執行 Streamlit_V2/install_requirements.bat。')
    url=ollama_url(config['OllamaURL'])
    try:
        with urlopen(url+'/api/tags',timeout=5) as response:data=json.load(response)
    except Exception as exc:raise ValueError('本機 Ollama 尚未連線；請先啟動 Ollama。'+str(exc)) from exc
    models={m.get('name',m.get('model','')) for m in data.get('models',[])}
    model=config['Model'].strip()
    if model not in models and model+':latest' not in models:raise ValueError(f'Ollama 尚未安裝模型 {model}；請先執行 ollama pull {model}。')


def submit(root,source,config):
    root=Path(root).resolve();source=Path(source).resolve()
    from quarterly_plan import QuarterPlan
    config=dict(config)
    plan=QuarterPlan(config.get('TargetQuarter','2026Q1'))
    config['TargetQuarter']=plan.target
    if not config.get('CompleteHistory',config.get('CompleteThrough2025',False)):raise ValueError(f'請確認 {plan.history_start}～{plan.holdout} 的12季資料已完整載入。')
    preflight(root,config)
    validation=validate_inputs(source,plan.target)
    with project_lock(root):
        recover_pending(root)
        current=current_status(root)
        if current.get('Status') in ACTIVE:raise RuntimeError('目前已有執行批次，請等待完成。')
        jid=datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')+'-'+uuid.uuid4().hex[:12]
        folder=safe_job(root,jid);inputs=folder/'inputs';inputs.mkdir(parents=True)
        for name in SCHEMAS:shutil.copy2(source/name,inputs/name)
        manifest={'JobID':jid,'CreatedAt':now(),'Config':config,'Validation':validation,'InputHashes':input_digest(inputs)}
        write_json(folder/'request.json',manifest)
        steps=[{'Name':name,'Script':path,'Status':'waiting'} for name,path in SCRIPT_STEPS]+[{'Name':'更新網頁結果','Script':'','Status':'waiting'}]
        write_json(folder/'status.json',{'JobID':jid,'CreatedAt':manifest['CreatedAt'],'Status':'queued','Message':'已排入背景執行','Steps':steps})
        write_json(jobs_dir(root)/'current.json',{'JobID':jid})
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        kwargs={'cwd':str(root),'env':env,'stdin':subprocess.DEVNULL}
        if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
        else:kwargs['start_new_session']=True
        try:
            with (folder/'worker.log').open('ab') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(root/'pipeline_runner.py'),'--project-root',str(root),'--job-id',jid],stdout=log,stderr=subprocess.STDOUT,**kwargs)
            write_json(folder/'process.json',{'PID':proc.pid})
        except Exception as exc:
            state=read_json(folder/'status.json');state.update(Status='failed',Message='啟動失敗：'+str(exc));write_json(folder/'status.json',state);raise
    return jid

def build_workspace(root,folder):
    work=folder/'work';work.mkdir()
    # 複製現用完整程式，暫存工作區沒有另外的客群邏輯。
    for current,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ('執行紀錄','.venv','venv','__pycache__','.git')]
        for name in files:
            if not name.endswith('.py'):continue
            source=Path(current)/name;relative=source.relative_to(root)
            target=work/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    raw=work/'原始資料'/'RAW_Data';raw.mkdir(parents=True,exist_ok=True)
    for name in SCHEMAS:shutil.copy2(folder/'inputs'/name,raw/name)
    config=read_json(folder/'request.json')['Config']
    write_json(work/'季度設定.json',{'TargetQuarter':config['TargetQuarter'],'HistoryQuarters':12,'FeatureQuarters':4})
    return work

def publish_targets(work):
    result=['原始資料/RAW_Data']+OUTPUT_DIRS+['季度設定.json','執行紀錄/已發布批次.json']
    diag=Path(work)/'診斷性分析'
    result += [str(p.relative_to(work)) for p in sorted(diag.iterdir()) if p.suffix not in ('.py','.pyc') and p.name!='__pycache__']
    for rel in result:
        if not (Path(work)/rel).exists():raise FileNotFoundError('流程未產生必要輸出：'+rel)
    return result

def rollback(root,folder,journal):
    for item in reversed(journal['Items']):
        rel=item['Path'];dst=Path(root)/rel;backup=folder/'backup'/rel
        # Prepared 在移動之前記錄；備份已存在即代表舊結果已被移走。
        if backup.exists():
            if dst.is_dir():shutil.rmtree(dst)
            elif dst.exists():dst.unlink()
            dst.parent.mkdir(parents=True,exist_ok=True);replace_path(backup,dst)
        elif not item['Existed']:
            if dst.is_dir():shutil.rmtree(dst)
            elif dst.exists():dst.unlink()
    journal.update(Pending=False,RestoredAt=now());write_json(folder/'publish.json',journal)

def recover_pending(root):
    state=current_status(root)
    if state.get('Status') in ACTIVE:return
    if state.get('JobID'):
        folder=safe_job(root,state['JobID']);journal=read_json(folder/'publish.json')
        if journal.get('Pending'):rollback(root,folder,journal)

def publish(root,folder,work):
    targets=publish_targets(work);journal={'Pending':True,'Items':[]};write_json(folder/'publish.json',journal)
    try:
        for rel in targets:
            dst=Path(root)/rel;src=Path(work)/rel;bak=folder/'backup'/rel
            journal['Items'].append({'Path':rel,'Existed':dst.exists()});write_json(folder/'publish.json',journal)
            if dst.exists():bak.parent.mkdir(parents=True,exist_ok=True);replace_path(dst,bak)
            dst.parent.mkdir(parents=True,exist_ok=True);replace_path(src,dst)
        journal.update(Pending=False,PublishedAt=now());write_json(folder/'publish.json',journal)
    except Exception:
        rollback(root,folder,journal);raise

def tail_log(root,job_id,max_bytes=18000):
    path=safe_job(root,job_id)/'worker.log'
    if not path.exists():return ''
    with path.open('rb') as f:f.seek(0,2);f.seek(max(0,f.tell()-max_bytes));return f.read().decode('utf-8',errors='replace')

def recover_project(root):
    with project_lock(root):
        state=current_status(root)
        if state.get('Status') in ACTIVE:raise RuntimeError('工作仍在執行，不能復原。')
        recover_pending(root)
        if state.get('JobID'):
            state.update(Status='failed',Message='已處理中斷的發布；可重新執行這批資料。');write_json(safe_job(root,state['JobID'])/'status.json',state)

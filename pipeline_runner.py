"""獨立背景程序：網頁重新整理不會重新啟動任務。"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from pipeline_runtime import *

def run(root,job_id):
    folder=safe_job(root,job_id)
    with project_lock(root,wait_seconds=15):
        if read_json(jobs_dir(root)/'current.json').get('JobID')!=job_id:raise RuntimeError('此批次已非目前工作。')
        state=read_json(folder/'status.json');request=read_json(folder/'request.json');config=request['Config']
        def save():state['UpdatedAt']=now();write_json(folder/'status.json',state)
        state.update(Status='running',StartedAt=now(),Message='準備本次資料');save()
        try:
            if input_digest(folder/'inputs')!=request['InputHashes']:raise ValueError('匯入資料在開始後被修改，請重新匯入。')
            work=build_workspace(root,folder)
            for index,(name,relative) in enumerate(SCRIPT_STEPS):
                row=state['Steps'][index];row.update(Status='running',StartedAt=now());state['Message']='執行中：'+name;save();start=time.monotonic()
                args=[sys.executable,'-u',str(work/relative)]
                if relative.endswith('Cluster_Profile.py'):args+=['--model',config['Model'],'--ollama-url',config['OllamaURL'],'--timeout','180','--retries','3']
                if relative=='產生整合報告.py':
                    args+=['--job-id',job_id,'--source-manifest',str(folder/'request.json')]
                    if config.get('LLMReport'):args+=['--llm','--model',config['Model'],'--ollama-url',config['OllamaURL']]
                print('\n執行：'+name,flush=True)
                proc=subprocess.Popen(args,cwd=work,stdin=subprocess.DEVNULL)
                row['PID']=proc.pid;save()
                while proc.poll() is None:save();time.sleep(1)
                row.update(DurationSeconds=round(time.monotonic()-start,1),ExitCode=proc.returncode);row.pop('PID',None)
                if proc.returncode:row['Status']='failed';save();raise RuntimeError(f'{name} 執行失敗（代碼 {proc.returncode}），請查看執行紀錄。')
                row.update(Status='completed',FinishedAt=now());save()
            state.update(Message='驗證模型與報告，準備更新網頁結果');state['Steps'][-1].update(Status='running',StartedAt=now());save()
            import importlib.util
            spec=importlib.util.spec_from_file_location('current_work_segment',work/'segment_contract.py');contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
            for relative in ['描述性分析/Operational_KPI','描述性分析/AARRR','描述性分析/Historical_CLV','診斷性分析','預測性分析/預測結果/整體季度營收趨勢預測','預測性分析/預測結果/流失預測','預測性分析/預測結果/CLV價值預測']:contract.validate_stage(work,work/relative)
            subprocess.run([sys.executable,str(work/'quarterly_validation.py'),'--project-root',str(work)],cwd=work,check=True)
            write_json(jobs_dir(work)/'已發布批次.json',{'JobID':job_id,'PublishedAt':now(),'Context':contract.context(work)})
            state.update(Status='publishing',Message='更新既有網頁結果');state['Steps'][-1]['Status']='running';save()
            publish(root,folder,work)
            state['Steps'][-1].update(Status='completed',FinishedAt=now());state.update(Status='completed',FinishedAt=now(),Message='全流程完成，圖表與報告已更新。');save()
        except Exception as exc:
            # 狀態寫入等非模型錯誤也可能發生在子程序執行期間。
            if 'proc' in locals() and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for row in state['Steps']:
                if row.get('Status')=='running':row.update(Status='failed',FinishedAt=now())
            state.update(Status='failed',FinishedAt=now(),Message=str(exc));save();print('流程停止：'+str(exc),flush=True)
            return 1
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--job-id',required=True);args=parser.parse_args()
    sys.exit(run(args.project_root.resolve(),args.job_id))

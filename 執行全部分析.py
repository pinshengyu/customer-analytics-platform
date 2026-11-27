"""命令列入口：與網頁共用隔離工作區、完整驗證及可復原發布。"""
from pathlib import Path
import argparse
import sys
import time
import pipeline_runtime as pipeline
from quarterly_plan import load_plan

def main():
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-quarter',default=load_plan(root).target)
    parser.add_argument('--confirm-complete-history',action='store_true',help='確認目標季度之前12季資料已完整匯入')
    parser.add_argument('--model',default='llama3.2:3b')
    parser.add_argument('--ollama-url',default='http://localhost:11434')
    parser.add_argument('--llm-report',action='store_true')
    args=parser.parse_args()
    if not args.confirm_complete_history:
        parser.error('請先確認三年資料完整，再加上 --confirm-complete-history；或從 main 首頁執行。')
    try:
        job=pipeline.submit(root,root/'原始資料/RAW_Data',{'TargetQuarter':args.target_quarter,'CompleteHistory':True,'Model':args.model,'OllamaURL':args.ollama_url,'LLMReport':args.llm_report})
        print('批次：'+job+'；關閉此視窗不會停止背景流程。',flush=True)
        previous=None
        while True:
            state=pipeline.current_status(root)
            if state.get('JobID')!=job:raise RuntimeError('目前批次已切換，請至網頁查看執行紀錄。')
            message=(state.get('Status'),state.get('Message'))
            if message!=previous:print(message[1],flush=True);previous=message
            if message[0] not in pipeline.ACTIVE:return 0 if message[0]=='completed' else 1
            time.sleep(2)
    except KeyboardInterrupt:
        print('已停止等待，背景流程仍會繼續，可在網頁查看。');return 130
    except (ValueError,RuntimeError,OSError) as exc:
        print('流程未完成：'+str(exc));return 1

if __name__=='__main__':sys.exit(main())

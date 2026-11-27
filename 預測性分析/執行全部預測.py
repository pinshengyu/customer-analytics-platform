from pathlib import Path
import subprocess
import sys
if __name__=='__main__':
    folder=Path(__file__).resolve().parent
    failures=[]
    for name in ['整體季度營收趨勢預測','流失預測','CLV價值預測']:
        result=subprocess.run([sys.executable,str(folder/(name+'.py')),*sys.argv[1:]],check=False)
        if result.returncode: failures.append(name)
    if failures:
        print('未完成：'+ '、'.join(failures)); sys.exit(1)
    print('三個預測模組全部完成。')

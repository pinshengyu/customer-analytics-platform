"""開啟已執行的網站，避免重複啟動 Streamlit。"""
from pathlib import Path
import os
import socket
import subprocess
import sys
import webbrowser
import msvcrt

ROOT = Path(__file__).resolve().parent
PORT = 8515
URL = f'http://127.0.0.1:{PORT}'

def main():
    folder = ROOT / '執行紀錄'
    folder.mkdir(exist_ok=True)
    with (folder / 'website.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if not lock.tell():
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('網站已在啟動或執行，請稍候。')
            webbrowser.open(URL)
            return 0
        try:
            with socket.create_connection(('127.0.0.1', PORT), timeout=2):
                print(f'{PORT} 已有本版網站服務，直接開啟，不重複啟動。')
                webbrowser.open(URL)
                return 0
        except OSError:
            pass
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        return subprocess.call([
            sys.executable, '-m', 'streamlit', 'run', '首頁.py',
            '--server.address', '127.0.0.1', '--server.port', str(PORT),
            '--server.fileWatcherType', 'none',
        ], cwd=ROOT / 'Streamlit_V2', env=env)

if __name__ == '__main__':
    sys.exit(main())

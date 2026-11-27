"""檢查專案檔案與已完成批次的 Streamlit 頁面；不重新訓練模型。"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / '檢查結果'


def inventory():
    import pandas as pd
    from PIL import Image
    rows = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or OUTPUT in path.parents:
            continue
        row = {'Path': str(path.relative_to(ROOT)), 'Bytes': path.stat().st_size,
               'SHA256': '', 'Check': 'readable', 'Rows': '', 'Error': ''}
        try:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            row['SHA256'] = digest.hexdigest()
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8-sig'))
                row['Check'] = 'python syntax'
            elif path.suffix == '.json':
                json.loads(path.read_text(encoding='utf-8-sig'))
                row['Check'] = 'json parse'
            elif path.suffix == '.csv':
                row['Rows'] = sum(len(chunk) for chunk in pd.read_csv(
                    path, encoding='utf-8-sig', chunksize=50000, low_memory=False))
                row['Check'] = 'csv full parse'
            elif path.suffix == '.png':
                with Image.open(path) as picture:
                    picture.verify()
                row['Check'] = 'png verify'
            elif path.suffix == '.xlsx':
                with zipfile.ZipFile(path) as book:
                    bad = book.testzip()
                    if bad:
                        raise ValueError('Invalid XLSX member: ' + bad)
                row['Check'] = 'xlsx archive integrity'
        except Exception as exc:
            row['Error'] = str(exc)
        rows.append(row)
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / '檔案清單.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = [row for row in rows if row['Error']]
    print(json.dumps({'Files': len(rows), 'Bytes': sum(row['Bytes'] for row in rows),
                      'Errors': errors}, ensure_ascii=False, indent=2))
    return not errors


def ui():
    import pipeline_runtime as pipeline
    from streamlit.testing.v1 import AppTest
    assert pipeline.published_data_available(ROOT), '完整批次尚未發布'
    sys.path.insert(0, str(ROOT / 'Streamlit_V2'))
    from dashboard.diagnostic_view import KPI_NAMES, DIMENSIONS
    records = []

    def check(app, label):
        errors = [str(x.value) for x in app.exception] + [str(x.value) for x in app.error]
        records.append({'Page': label, 'Errors': errors,
                        'Metrics': len(app.metric), 'Tables': len(app.dataframe)})
        print(label, 'PASS' if not errors else errors, flush=True)
        assert not errors, errors

    files = [ROOT / 'Streamlit_V2' / '首頁.py'] + sorted((ROOT / 'Streamlit_V2' / 'pages').glob('*.py'))
    for path in files:
        app = AppTest.from_file(str(ROOT / 'Streamlit_V2' / '首頁.py'), default_timeout=90).run()
        if path.name != '首頁.py':
            app.switch_page('pages/' + path.name).run()
        check(app, path.name)
        tab_sets = {
            '1_描述性分析.py': ('descriptive_tabs', ['營運 KPI', 'AARRR', 'Historical CLV', 'Cohort']),
            '2_診斷性分析.py': ('diagnostic_tabs', ['異常總覽', '原因鏈與貢獻度', '客群交叉分析', '原有診斷報表', '資料品質']),
            '3_預測性分析.py': ('predictive_tabs', ['整體季度營收趨勢預測', '流失預測', 'CLV價值預測']),
            '4_模型結果分析.py': ('evaluation_tabs', ['整體季度營收趨勢預測', '流失預測', 'CLV價值預測']),
        }
        tab_key, labels = tab_sets.get(path.name, (None, [None]))
        for label in labels:
            def rerun():
                if tab_key:
                    app.session_state[tab_key] = label
                app.run()
            if tab_key:
                app.session_state[tab_key] = label
                app.run()
                check(app, f'{path.name}: {label}')
            for radio in list(app.radio):
                for option in radio.options[1:]:
                    app.radio(key=radio.key).set_value(option); rerun()
                    check(app, f'{path.name}: {radio.key} = {option}')
            # 單一客群模式也會建立額外選單。格式化選單使用原始值。
            for widget in list(app.selectbox):
                key = widget.key
                options = list(widget.options)
                if key in ('diag_kpi', 'diag_trend_metric'):
                    options = list(KPI_NAMES)
                elif key == 'diag_driver_dimension':
                    options = list(DIMENSIONS)
                elif key == 'risk_eval':
                    options = ['Recall', 'Precision', 'F1']
                elif key in ('流失預測', 'CLV價值預測'):
                    import pandas as pd
                    data = pd.read_csv(ROOT / '預測性分析' / '預測結果' / key / 'Customer_Predictions.csv')
                    options = ['整體'] + sorted(data.Cluster.unique().tolist())
                for option in options:
                    app.selectbox(key=key).set_value(option); rerun()
                    check(app, f'{path.name}: {key} = {option}')
                app.selectbox(key=key).set_value(options[0]); rerun()
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / '網頁驗證.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ui', action='store_true', help='驗證已完成批次的頁面與選單')
    args = parser.parse_args()
    sys.exit(0 if (ui() if args.ui else inventory()) else 1)

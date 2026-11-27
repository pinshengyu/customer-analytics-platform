# 客戶分析與智慧決策支援平台

國立臺中科技大學人工智慧應用工程學士學位學程 畢業專題（A21）

把會員、交易與行銷活動資料整合成一個 Streamlit 網站：從營運現況、問題追查、客戶分群到下一季預測，最後由本機 LLM 產出可下載的分析報告，減少營運會議前整理資料的時間。

## 功能

| 頁面 | 內容 |
| --- | --- |
| 資料匯入 | 上傳 6 類 CSV（客戶、商品、訂單、行為事件、推薦關係、行銷活動），檢查欄位、日期、重複與關聯 |
| 描述性分析 | 營運 KPI、AARRR、Cohort 留存、歷史 CLV |
| 診斷性分析 | 季度指標變化、變化來源拆解、客群交叉分析 |
| 預測性分析 | 下一季營收、客戶流失風險、客戶利潤（CLV）預測 |
| 模型結果分析 | 營收回測、流失分類評估、CLV 誤差與版本比較 |
| 整合報告 | LLM 撰寫報告解讀，可下載 HTML／Markdown／JSON |

範例輸出：[docs/sample-report.html](docs/sample-report.html)

## 技術重點

- **資料管線**：pandas ETL → 資料倉儲表 → 各分析模組，每次執行建立獨立批次並在驗證通過後才發布。
- **客戶分群**：RFM 特徵（log1p + 標準化）搭配 K-Means，在 K=3～7 間以輪廓係數選擇群數。
- **預測模型**：XGBoost 預測流失（LogLoss 選模）與 CLV（MAE 選模），另建立季度營收預測。
- **避免資料洩漏**：依目標季度往前取 12 季，按時間切分 70/15/15，排除跨界未到期標籤，最後一季獨立回測。細節見 [季度模型說明](季度模型說明.md)。
- **LLM 整合**：本機 Ollama（`llama3.2:3b`）為客群命名與撰寫報告解讀；Prompt 限定只能引用已計算的數字，不得虛構因果或成效。
- **測試**：`test_quarterly.py`（季度切分與標籤）、`test_pipeline_safety.py`（混版報告、模型殘留、發布失敗復原）。

## 快速開始

需求：Python 3.11 以上；客群命名與報告解讀需本機 [Ollama](https://ollama.com/) 並下載 `llama3.2:3b`。

```bash
pip install -r Streamlit_V2/requirements.txt
python -m unittest test_quarterly test_pipeline_safety
```

Windows 可雙擊 `start.bat` 啟動網站（`launch_site.py` 使用 Windows 的檔案鎖），或直接執行：

```bash
cd Streamlit_V2
streamlit run 首頁.py
```

在首頁選擇預測季度並上傳 6 類 CSV 後執行完整流程。命令列版本：

```bash
python 執行全部分析.py --target-quarter 2026Q1 --confirm-complete-history
```

> 專題使用的模擬資料（約 2 萬名客戶、2023～2025 年）檔案過大，未收錄於此 repo。資料格式與生成規則見 [原始資料/生成規則.md](原始資料/生成規則.md)。

## 專案結構

```
原始資料/        ETL、資料倉儲、模擬資料生成器
描述性分析/      KPI、AARRR、Cohort、歷史 CLV
診斷性分析/      季度變化與來源拆解
K-means分群/     RFM 分群與 LLM 客群命名
預測性分析/      流失、CLV、營收預測入口
quarterly_*.py   季度切分、建模、回測、驗證
pipeline_*.py    背景執行、批次管理與發布
Streamlit_V2/    網站介面
```

完整操作說明見 [使用說明.md](使用說明.md)。

## 團隊

組員：鄭○○、余品陞、蕭○○　指導老師：陳志騰

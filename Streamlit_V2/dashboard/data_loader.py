from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import segment_contract as segment


def get_project_root() -> Path:
    """
    自動向上尋找真正的專題根目錄。

    支援：
    專題根目錄/
    ├─ Streamlit_V2/
    │  ├─ 首頁.py
    │  └─ dashboard/
    ├─ 描述性分析/
    ├─ 原始資料/
    └─ K-means分群/

    因此 Streamlit 資料夾不需要拆開搬到專題根目錄。
    """
    start = Path(__file__).resolve().parent

    for candidate in [start, *start.parents]:
        descriptive = candidate / "描述性分析"
        raw_dw = candidate / "原始資料" / "Data_Warehouse"

        if descriptive.is_dir() and (raw_dw.is_dir() or (candidate / "原始資料" / "ETL.py").is_file()):
            return candidate

    raise FileNotFoundError(
        "無法自動找到專題根目錄。\n"
        "請確認 Streamlit_V2 資料夾位於專題根目錄內，且專題根目錄包含：\n"
        "1. 描述性分析/\n"
        "2. 原始資料/Data_Warehouse/ 或 ETL_Output/"
    )


def require_csv(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"找不到分析輸出 CSV：{path}")
    return path


def read_csv(path: str) -> pd.DataFrame:
    from pipeline_runtime import publishing_blocked, published_data_available
    if not st.session_state.get("session_data_ready", False):
        raise ValueError("本次工作階段尚未完成資料匯入與分析。")
    if publishing_blocked(get_project_root()):
        raise ValueError("結果正在更新或更新中斷；請至首頁的資料匯入區查看進度／復原。")
    if not published_data_available(get_project_root()):
        raise ValueError("尚未匯入並完成分析，不載入既有CSV。")
    file_path = require_csv(Path(path))
    info = file_path.stat()
    return _read_csv_cached(str(file_path), info.st_mtime_ns, info.st_size)


@st.cache_data(show_spinner=False, max_entries=32, ttl=300)
def _read_csv_cached(path: str, modified_ns: int, size_bytes: int) -> pd.DataFrame:
    file_path = Path(path)

    try:
        return pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            low_memory=False,
        )



class OptionalDataError(RuntimeError):
    """非核心擴充資料無法使用；保留原因供 UI/開發者診斷。"""


def load_optional_csv(path: Path, required_columns: set[str] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """讀取非必要擴充資料。缺檔/格式錯誤不影響核心 Loader，但不偽裝成空資料。"""
    from pipeline_runtime import publishing_blocked, published_data_available
    if not st.session_state.get("session_data_ready", False):
        raise ValueError("本次工作階段尚未完成資料匯入與分析。")
    if publishing_blocked(get_project_root()):
        raise ValueError("結果正在更新或更新中斷；請至首頁的資料匯入區查看進度／復原。")
    if not published_data_available(get_project_root()):
        raise ValueError("尚未匯入並完成分析，不載入既有CSV。")
    path = Path(path)
    if not path.is_file():
        return None, f"尚未產生季度分析資料：{path.name}"
    try:
        info=path.stat();data=_read_csv_cached(str(path),info.st_mtime_ns,info.st_size)
    except (FileNotFoundError, UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        return None, f"{path.name} 無法載入：{type(exc).__name__}: {exc}"
    if required_columns:
        missing = sorted(set(required_columns) - set(data.columns))
        if missing:
            return None, f"{path.name} 缺少必要欄位：{', '.join(missing)}"
    return data, None


def load_operational_quarterly_extensions() -> dict[str, object]:
    root = get_project_root()
    cluster, cluster_error = load_optional_csv(
        root / "描述性分析" / "Operational_KPI" / "Operational_Cluster_Quarterly_Rolling.csv",
        {"Quarter","ClusterName","TotalRevenue","TotalOrders","PurchasingCustomers","AOV"},
    )
    customer, customer_error = load_optional_csv(
        root / "描述性分析" / "Quarterly_Clustering" / "Quarterly_Customer_KPI.csv",
        {"Quarter","CustomerID","ClusterName","Recency","Frequency","Monetary","QuarterRevenue","QuarterProfit","QuarterOrders"},
    )
    return {"cluster_quarterly": cluster, "customer_quarterly": customer,
            "cluster_error": cluster_error, "customer_error": customer_error}


def load_historical_clv_quarterly_extensions() -> dict[str, object]:
    root = get_project_root()
    quarterly, quarterly_error = load_optional_csv(
        root / "描述性分析" / "Historical_CLV" / "Historical_CLV_Quarterly.csv",
        {"Scope","Quarter","QuarterlyProfit","QuarterlyRevenue","PurchasingCustomers"},
    )
    customer, customer_error = load_optional_csv(
        root / "描述性分析" / "Quarterly_Clustering" / "Quarterly_Customer_KPI.csv",
        {"Quarter","CustomerID","ClusterName","Recency","Frequency","Monetary","QuarterRevenue","QuarterProfit","QuarterOrders"},
    )
    return {"quarterly": quarterly, "customer_quarterly": customer,
            "quarterly_error": quarterly_error, "customer_error": customer_error}

def load_operational() -> dict[str, pd.DataFrame]:
    project_root = get_project_root()
    base = project_root / "描述性分析" / "Operational_KPI"
    segment.validate_stage(project_root, base)

    return {
        "overall": read_csv(str(base / "Operational_Overall_KPI.csv")),
        "cluster": read_csv(str(base / "Operational_Cluster_KPI.csv")),
        "quarterly": read_csv(str(base / "Operational_Quarterly_Trend.csv")),
        "cluster_quarterly": read_csv(
            str(base / "Operational_Cluster_Quarterly_Trend.csv")
        ),
    }


def load_aarrr() -> dict[str, pd.DataFrame]:
    project_root = get_project_root()
    base = project_root / "描述性分析" / "AARRR"
    segment.validate_stage(project_root, base)

    return {
        "overall": read_csv(str(base / "AARRR_Overall_KPI.csv")),
        "cluster": read_csv(str(base / "AARRR_Cluster_KPI.csv")),
        "diagnosis": read_csv(str(base / "AARRR_Stage_Diagnosis.csv")),
    }


def load_historical_clv() -> dict[str, pd.DataFrame]:
    project_root = get_project_root()
    base = project_root / "描述性分析" / "Historical_CLV"
    segment.validate_stage(project_root, base)

    return {
        "overall": read_csv(str(base / "Historical_CLV_Overall.csv")),
        "cluster": read_csv(str(base / "Historical_CLV_Cluster.csv")),
        "distribution": read_csv(
            str(base / "Historical_CLV_Distribution.csv")
        ),
    }


def load_cohort() -> dict[str, pd.DataFrame]:
    project_root = get_project_root()
    base = project_root / "描述性分析" / "Cohort"

    return {
        "size": read_csv(str(base / "Cohort_Size.csv")),
        "matrix": read_csv(str(base / "Cohort_Retention_Matrix.csv")),
        "long": read_csv(str(base / "Cohort_Retention_Long.csv")),
    }


DIAGNOSTIC_FILES = {
    "summary": "營收驅動/Diagnostic_Quarterly_Summary.csv",
    "findings": "Diagnostic_Findings.csv",
    "drivers": "Diagnostic_Driver_Contributions.csv",
    "cluster_category": "Diagnostic_Cluster_Category.csv",
    "cluster_channel": "Diagnostic_Cluster_Channel.csv",
    "cluster_brand": "Diagnostic_Cluster_Brand.csv",
    "cluster_campaign": "Diagnostic_Cluster_Campaign.csv",
    "cluster_category_channel": "Diagnostic_Cluster_Category_Channel.csv",
    "cluster_trends": "客群趨勢/Diagnostic_Cluster_Quarterly.csv",
    "category": "商品與折扣/Diagnostic_Product_Category.csv",
    "brand": "商品與折扣/Diagnostic_Product_Brand.csv",
    "channel": "商品與折扣/Diagnostic_Product_Channel.csv",
    "campaign": "活動歸因/Diagnostic_Campaign_Attribution.csv",
    "funnel": "來源漏斗/Diagnostic_Acquisition_Funnel.csv",
    "quality": "Diagnostic_Data_Quality.csv",
}


def load_diagnostic(key: str) -> pd.DataFrame:
    base = get_project_root() / "診斷性分析"
    segment.validate_stage(get_project_root(), base)
    return read_csv(str(base / DIAGNOSTIC_FILES[key]))


def diagnostic_file_status() -> pd.DataFrame:
    base = get_project_root() / "診斷性分析"
    return pd.DataFrame([
        {"資料": key, "檔案": filename, "已產生": (base / filename).is_file()}
        for key, filename in DIAGNOSTIC_FILES.items()
    ])

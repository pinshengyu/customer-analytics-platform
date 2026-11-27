from __future__ import annotations

"""
營運 KPI 描述性分析 V1
============================================================

目的
----
掌握：
1. 整體營運規模
2. 季度成長趨勢
3. 各命名客群的營收貢獻
4. 為後續 Streamlit 提供可直接切換：
   - 整體
   - 分群比較
   - 單一客群
   的標準化資料。

目前核心 KPI
------------
1. Total Revenue
   = SUM(Fact_Sales.Amount)

2. Total Orders
   = NUNIQUE(Fact_Sales.OrderID)

3. Purchasing Customers
   = 有至少 1 筆有效訂單的不重複客戶數

4. AOV
   = Total Revenue / Total Orders

5. Revenue Growth Rate
   = (本季 Revenue - 上季 Revenue) / 上季 Revenue * 100
   採季度 QoQ。

6. Revenue Share
   = 客群 Revenue / 整體 Revenue * 100
   用於分群營收貢獻比較。

注意
----
目前尚未正式納入 Cost，因此本版不計算：
- Gross Profit
- Gross Margin
- Profit Share

未來 Cost 進入資料生成 / ETL / DW 後，可再擴充。

資料粒度
--------
Fact_Sales 為 Order Line 粒度，因此：
- 訂單數一定使用 OrderID.nunique()
- 不可直接使用 Fact_Sales row count

Cluster
-------
Cluster 編號只作為資料關聯鍵。
所有展示名稱均讀取：
K-means分群/Cluster/Cluster_Profile_Final.csv
中的 ClusterName。

預設專案結構
------------
專題v2/
├─ 原始資料/
│  └─ Data_Warehouse/
│     ├─ Dim_Customer.csv
│     ├─ Dim_Time.csv
│     └─ Fact_Sales.csv
│
├─ K-means分群/
│  ├─ K_Means/
│  │  └─ K_Means.csv
│  └─ Cluster/
│     └─ Cluster_Profile_Final.csv
│
└─ 描述性分析/
   ├─ Operational_KPI.py
   └─ Operational_KPI/
      ├─ Operational_Overall_KPI.csv
      ├─ Operational_Cluster_KPI.csv
      ├─ Operational_Quarterly_Trend.csv
      ├─ Operational_Cluster_Quarterly_Trend.csv
      ├─ Operational_Dashboard_KPI.csv
      ├─ Operational_Quarterly_Revenue.png
      ├─ Operational_Quarterly_Growth.png
      ├─ Operational_Cluster_Revenue.png
      └─ Operational_Cluster_Revenue_Share.png
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment_contract as segment



PROGRAM_TITLE = "OPERATIONAL KPI ANALYSIS V1"


# ============================================================
# 1. DISPLAY
# ============================================================

def banner(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)


def section(title: str) -> None:
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


# ============================================================
# 2. BASIC UTILITIES
# ============================================================

def normalize_column(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    table_name: str,
    required: bool = True,
) -> str | None:
    column_map = {
        normalize_column(column): column
        for column in df.columns
    }

    for candidate in candidates:
        matched = column_map.get(normalize_column(candidate))
        if matched is not None:
            return matched

    if required:
        raise ValueError(
            f"{table_name} 找不到必要欄位。\n"
            f"候選欄位：{list(candidates)}\n"
            f"實際欄位：{list(df.columns)}"
        )

    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            low_memory=False,
        )


def clean_identifier(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    return text.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )


def safe_divide(
    numerator: float | int,
    denominator: float | int,
) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return float(numerator) / float(denominator)


def round_output(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns].round(4)
    )

    return result


# ============================================================
# 3. PATHS
# ============================================================

def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        # 專題v2/描述性分析/Operational_KPI.py
        project_root = script_dir.parent

    dw_dir = project_root / "原始資料" / "Data_Warehouse"

    fact_sales_path = (
        args.fact_sales.resolve()
        if args.fact_sales
        else dw_dir / "Fact_Sales.csv"
    )

    dim_customer_path = (
        args.dim_customer.resolve()
        if args.dim_customer
        else dw_dir / "Dim_Customer.csv"
    )

    dim_time_path = (
        args.dim_time.resolve()
        if args.dim_time
        else dw_dir / "Dim_Time.csv"
    )

    kmeans_path = (
        args.kmeans.resolve()
        if args.kmeans
        else (
            project_root
            / "K-means分群"
            / "K_Means"
            / "K_Means.csv"
        )
    )

    cluster_profile_path = (
        args.cluster_profile.resolve()
        if args.cluster_profile
        else (
            project_root
            / "K-means分群"
            / "Cluster"
            / "Cluster_Profile_Final.csv"
        )
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else script_dir / "Operational_KPI"
    )

    required = (
        fact_sales_path,
        dim_customer_path,
        dim_time_path,
        kmeans_path,
        cluster_profile_path,
    )

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到必要輸入檔案：{path}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        project_root,
        dw_dir,
        fact_sales_path,
        dim_customer_path,
        dim_time_path,
        kmeans_path,
        cluster_profile_path,
        output_dir,
    )


# ============================================================
# 4. DIM CUSTOMER
# ============================================================

def prepare_customers(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    customer_key_col = find_column(
        raw,
        ("CustomerKey", "Customer_Key", "CustomerSK"),
        "Dim_Customer",
    )

    customer_id_col = find_column(
        raw,
        ("CustomerID", "Customer_ID", "CustomerBK"),
        "Dim_Customer",
    )

    result = raw[
        [customer_key_col, customer_id_col]
    ].copy()

    result.columns = [
        "CustomerKey",
        "CustomerID",
    ]

    result["CustomerKey"] = clean_identifier(
        result["CustomerKey"]
    )

    result["CustomerID"] = clean_identifier(
        result["CustomerID"]
    )

    result = result.dropna(
        subset=["CustomerKey", "CustomerID"]
    ).copy()

    if result["CustomerKey"].duplicated().any():
        raise ValueError(
            "Dim_Customer.CustomerKey 必須唯一。"
        )

    if result["CustomerID"].duplicated().any():
        raise ValueError(
            "Dim_Customer.CustomerID 必須唯一。"
        )

    return result.reset_index(drop=True)


# ============================================================
# 5. DIM TIME
# ============================================================

def prepare_time(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    date_key_col = find_column(
        raw,
        ("DateKey", "Date_Key"),
        "Dim_Time",
    )

    date_col = find_column(
        raw,
        ("Date", "FullDate", "CalendarDate"),
        "Dim_Time",
    )

    result = raw[
        [date_key_col, date_col]
    ].copy()

    result.columns = [
        "DateKey",
        "Date",
    ]

    result["DateKey"] = clean_identifier(
        result["DateKey"]
    )

    result["Date"] = pd.to_datetime(
        result["Date"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["DateKey", "Date"]
    ).copy()

    if result["DateKey"].duplicated().any():
        raise ValueError(
            "Dim_Time.DateKey 必須唯一。"
        )

    result["Year"] = result["Date"].dt.year
    result["QuarterNumber"] = (
        result["Date"].dt.quarter
    )

    result["Quarter"] = (
        result["Year"].astype(str)
        + " Q"
        + result["QuarterNumber"].astype(str)
    )

    result["QuarterPeriod"] = (
        result["Date"].dt.to_period("Q")
    )

    return result.reset_index(drop=True)


# ============================================================
# 6. K-MEANS
# ============================================================

def prepare_kmeans(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    customer_col = find_column(
        raw,
        ("CustomerID", "Customer_ID"),
        "K_Means.csv",
    )

    cluster_col = find_column(
        raw,
        ("Cluster", "ClusterID", "Cluster_Label", "Label"),
        "K_Means.csv",
    )

    result = raw[
        [customer_col, cluster_col]
    ].copy()

    result.columns = [
        "CustomerID",
        "Cluster",
    ]

    result["CustomerID"] = clean_identifier(
        result["CustomerID"]
    )

    result["Cluster"] = pd.to_numeric(
        result["Cluster"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["CustomerID", "Cluster"]
    ).copy()

    result["Cluster"] = result["Cluster"].astype(int)

    if result["CustomerID"].duplicated().any():
        raise ValueError(
            "K_Means.csv 的 CustomerID 必須唯一。"
        )

    return result.reset_index(drop=True)


# ============================================================
# 7. CLUSTER NAMES
# ============================================================

def prepare_cluster_names(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    cluster_col = find_column(
        raw,
        ("Cluster", "ClusterID", "Cluster_ID", "ClusterLabel"),
        "Cluster_Profile_Final.csv",
    )

    name_col = find_column(
        raw,
        (
            "ClusterName",
            "Cluster_Name",
            "SegmentName",
            "Segment_Name",
            "CustomerSegment",
            "Customer_Segment",
            "Name",
        ),
        "Cluster_Profile_Final.csv",
    )

    result = raw[
        [cluster_col, name_col]
    ].copy()

    result.columns = [
        "Cluster",
        "ClusterName",
    ]

    result["Cluster"] = pd.to_numeric(
        result["Cluster"],
        errors="coerce",
    )

    result["ClusterName"] = (
        result["ClusterName"]
        .astype("string")
        .str.strip()
    )

    result = result.dropna(
        subset=["Cluster", "ClusterName"]
    ).copy()

    result["Cluster"] = result["Cluster"].astype(int)

    result = result[
        result["ClusterName"].ne("")
    ].copy()

    if result["Cluster"].duplicated().any():
        raise ValueError(
            "Cluster_Profile_Final.csv 的 Cluster 必須唯一。"
        )

    return result.reset_index(drop=True)


# ============================================================
# 8. FACT SALES
# ============================================================

def prepare_sales(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    customer_key_col = find_column(
        raw,
        ("CustomerKey", "Customer_Key", "CustomerSK"),
        "Fact_Sales",
    )

    date_key_col = find_column(
        raw,
        ("DateKey", "Date_Key"),
        "Fact_Sales",
    )

    order_col = find_column(
        raw,
        (
            "OrderID",
            "Order_ID",
            "SalesOrderID",
            "TransactionID",
        ),
        "Fact_Sales",
    )

    amount_col = find_column(
        raw,
        (
            "Amount",
            "SalesAmount",
            "Revenue",
            "NetAmount",
            "TotalAmount",
        ),
        "Fact_Sales",
    )

    result = raw[
        [
            customer_key_col,
            date_key_col,
            order_col,
            amount_col,
        ]
    ].copy()

    result.columns = [
        "CustomerKey",
        "DateKey",
        "OrderID",
        "Amount",
    ]

    result["CustomerKey"] = clean_identifier(
        result["CustomerKey"]
    )

    result["DateKey"] = clean_identifier(
        result["DateKey"]
    )

    result["OrderID"] = clean_identifier(
        result["OrderID"]
    )

    result["Amount"] = pd.to_numeric(
        result["Amount"],
        errors="coerce",
    )

    invalid = (
        result[
            ["CustomerKey", "DateKey", "OrderID", "Amount"]
        ]
        .isna()
        .any(axis=1)
        |
        (result["Amount"] <= 0)
    )

    before = len(result)

    result = result.loc[
        ~invalid
    ].copy()

    print(
        f"Fact_Sales rows before cleaning : {before:,}"
    )
    print(
        f"Fact_Sales rows after cleaning  : {len(result):,}"
    )

    if result.empty:
        raise ValueError(
            "Fact_Sales 清理後沒有有效交易。"
        )

    return result


# ============================================================
# 9. BUILD ANALYSIS BASE
# ============================================================

def build_analysis_base(
    sales: pd.DataFrame,
    customers: pd.DataFrame,
    time_dim: pd.DataFrame,
    kmeans: pd.DataFrame,
    cluster_names: pd.DataFrame,
) -> pd.DataFrame:
    cluster_assignment = kmeans.merge(
        cluster_names,
        on="Cluster",
        how="left",
        validate="many_to_one",
    )

    if cluster_assignment["ClusterName"].isna().any():
        missing = sorted(
            cluster_assignment.loc[
                cluster_assignment["ClusterName"].isna(),
                "Cluster",
            ]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "以下 Cluster 找不到命名結果："
            f"{missing}"
        )

    customer_map = customers.merge(
        cluster_assignment,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    result = sales.merge(
        customer_map,
        on="CustomerKey",
        how="left",
        validate="many_to_one",
    )

    result = result.merge(
        time_dim[
            [
                "DateKey",
                "Date",
                "Year",
                "QuarterNumber",
                "Quarter",
                "QuarterPeriod",
            ]
        ],
        on="DateKey",
        how="left",
        validate="many_to_one",
    )

    if result["CustomerID"].isna().any():
        raise ValueError(
            "部分 Fact_Sales.CustomerKey "
            "無法對應 Dim_Customer。"
        )

    if result["Date"].isna().any():
        raise ValueError(
            "部分 Fact_Sales.DateKey "
            "無法對應 Dim_Time。"
        )

    return result


# ============================================================
# 10. KPI FUNCTIONS
# ============================================================

def aggregate_kpi(
    data: pd.DataFrame,
) -> dict[str, float | int]:
    revenue = float(
        data["Amount"].sum()
    )

    orders = int(
        data["OrderID"].nunique()
    )

    purchasing_customers = int(
        data["CustomerID"].nunique()
    )

    aov = safe_divide(
        revenue,
        orders,
    )

    return {
        "TotalRevenue": revenue,
        "TotalOrders": orders,
        "PurchasingCustomers": purchasing_customers,
        "AOV": aov,
    }


def calculate_overall_kpi(
    base: pd.DataFrame,
) -> pd.DataFrame:
    values = aggregate_kpi(base)

    # Overall 的 Revenue Growth Rate
    # 採最新完整資料季度相對前一季度 QoQ。
    quarterly = (
        base
        .groupby(
            "QuarterPeriod",
            as_index=False,
            observed=True,
        )
        .agg(
            Revenue=("Amount", "sum")
        )
        .sort_values("QuarterPeriod")
    )

    quarterly["RevenueGrowthRate"] = (
        quarterly["Revenue"].pct_change() * 100.0
    )

    latest_growth = (
        float(
            quarterly.iloc[-1]["RevenueGrowthRate"]
        )
        if len(quarterly) >= 2
        else np.nan
    )

    row = {
        "Scope": "Overall",
        "Cluster": pd.NA,
        "ClusterName": pd.NA,
        "DisplayName": "整體",
        **values,
        "RevenueGrowthRate": latest_growth,
        "RevenueShare": 100.0,
    }

    return pd.DataFrame([row])


def calculate_cluster_kpi(
    base: pd.DataFrame,
) -> pd.DataFrame:
    clustered = base.dropna(
        subset=["Cluster", "ClusterName"]
    ).copy()

    clustered["Cluster"] = (
        clustered["Cluster"].astype(int)
    )

    overall_revenue = float(
        base["Amount"].sum()
    )

    rows: list[dict[str, Any]] = []

    for (
        cluster,
        cluster_name,
    ), group in clustered.groupby(
        ["Cluster", "ClusterName"],
        sort=True,
        observed=True,
    ):
        values = aggregate_kpi(group)

        cluster_quarterly = (
            group
            .groupby(
                "QuarterPeriod",
                as_index=False,
                observed=True,
            )
            .agg(
                Revenue=("Amount", "sum")
            )
            .sort_values("QuarterPeriod")
        )

        cluster_quarterly[
            "RevenueGrowthRate"
        ] = (
            cluster_quarterly["Revenue"]
            .pct_change()
            * 100.0
        )

        latest_growth = (
            float(
                cluster_quarterly.iloc[-1][
                    "RevenueGrowthRate"
                ]
            )
            if len(cluster_quarterly) >= 2
            else np.nan
        )

        rows.append(
            {
                "Scope": "Cluster",
                "Cluster": int(cluster),
                "ClusterName": str(cluster_name),
                "DisplayName": str(cluster_name),
                **values,
                "RevenueGrowthRate": latest_growth,
                "RevenueShare": (
                    safe_divide(
                        values["TotalRevenue"],
                        overall_revenue,
                    )
                    * 100.0
                    if overall_revenue > 0
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ValueError(
            "沒有可用的 Cluster KPI。"
        )

    return result


# ============================================================
# 11. QUARTERLY TREND
# ============================================================

def calculate_quarterly_trend(
    base: pd.DataFrame,
) -> pd.DataFrame:
    result = (
        base
        .groupby(
            [
                "QuarterPeriod",
                "Quarter",
                "Year",
                "QuarterNumber",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            TotalRevenue=("Amount", "sum"),
            TotalOrders=("OrderID", "nunique"),
            PurchasingCustomers=("CustomerID", "nunique"),
        )
        .sort_values(
            ["Year", "QuarterNumber"]
        )
        .reset_index(drop=True)
    )

    result["AOV"] = (
        result["TotalRevenue"]
        / result["TotalOrders"]
    )

    result["RevenueGrowthRate"] = (
        result["TotalRevenue"]
        .pct_change()
        * 100.0
    )

    result.insert(
        0,
        "Scope",
        "Overall",
    )

    result.insert(
        1,
        "DisplayName",
        "整體",
    )

    result = result.drop(
        columns=["QuarterPeriod"]
    )

    return result


def calculate_cluster_quarterly_trend(
    base: pd.DataFrame,
) -> pd.DataFrame:
    clustered = base.dropna(
        subset=["Cluster", "ClusterName"]
    ).copy()

    clustered["Cluster"] = (
        clustered["Cluster"].astype(int)
    )

    result = (
        clustered
        .groupby(
            [
                "Cluster",
                "ClusterName",
                "QuarterPeriod",
                "Quarter",
                "Year",
                "QuarterNumber",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            TotalRevenue=("Amount", "sum"),
            TotalOrders=("OrderID", "nunique"),
            PurchasingCustomers=("CustomerID", "nunique"),
        )
        .sort_values(
            [
                "Cluster",
                "Year",
                "QuarterNumber",
            ]
        )
        .reset_index(drop=True)
    )

    result["AOV"] = (
        result["TotalRevenue"]
        / result["TotalOrders"]
    )

    result["RevenueGrowthRate"] = (
        result
        .groupby(
            "Cluster",
            observed=True,
        )["TotalRevenue"]
        .pct_change()
        * 100.0
    )

    result.insert(
        0,
        "Scope",
        "Cluster",
    )

    result.insert(
        3,
        "DisplayName",
        result["ClusterName"],
    )

    result = result.drop(
        columns=["QuarterPeriod"]
    )

    return result


# ============================================================
# 12. CHARTS
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    plt.rcParams["axes.unicode_minus"] = False


def save_quarterly_revenue_chart(
    trend: pd.DataFrame,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.plot(
        trend["Quarter"],
        trend["TotalRevenue"],
        marker="o",
    )

    ax.set_title(
        "Quarterly Revenue Trend"
    )
    ax.set_xlabel(
        "Quarter"
    )
    ax.set_ylabel(
        "Revenue"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "Operational_Quarterly_Revenue.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def save_quarterly_growth_chart(
    trend: pd.DataFrame,
    output_dir: Path,
) -> Path:
    data = trend.dropna(
        subset=["RevenueGrowthRate"]
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        data["Quarter"],
        data["RevenueGrowthRate"],
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Quarter-over-Quarter Revenue Growth"
    )
    ax.set_xlabel(
        "Quarter"
    )
    ax.set_ylabel(
        "Growth Rate (%)"
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "Operational_Quarterly_Growth.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def save_cluster_revenue_chart(
    cluster_kpi: pd.DataFrame,
    output_dir: Path,
) -> Path:
    data = (
        cluster_kpi
        .sort_values(
            "TotalRevenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        data["ClusterName"],
        data["TotalRevenue"],
    )

    ax.set_title(
        "Revenue by Customer Segment"
    )
    ax.set_ylabel(
        "Revenue"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment(
            "right"
        )

    for bar, value in zip(
        bars,
        data["TotalRevenue"],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    path = (
        output_dir
        / "Operational_Cluster_Revenue.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def save_cluster_revenue_share_chart(
    cluster_kpi: pd.DataFrame,
    output_dir: Path,
) -> Path:
    data = (
        cluster_kpi
        .sort_values(
            "RevenueShare",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        data["ClusterName"],
        data["RevenueShare"],
    )

    ax.set_title(
        "Revenue Contribution by Customer Segment"
    )
    ax.set_ylabel(
        "Revenue Share (%)"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment(
            "right"
        )

    for bar, value in zip(
        bars,
        data["RevenueShare"],
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    path = (
        output_dir
        / "Operational_Cluster_Revenue_Share.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


# ============================================================
# 13. ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "建立可供 Streamlit 使用的營運 KPI 描述性分析資料。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
    )

    parser.add_argument(
        "--fact-sales",
        type=Path,
    )

    parser.add_argument(
        "--dim-customer",
        type=Path,
    )

    parser.add_argument(
        "--dim-time",
        type=Path,
    )

    parser.add_argument(
        "--kmeans",
        type=Path,
    )

    parser.add_argument(
        "--cluster-profile",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
    )

    return parser.parse_args()


# ============================================================
# 14. MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    (
        project_root,
        dw_dir,
        fact_sales_path,
        dim_customer_path,
        dim_time_path,
        kmeans_path,
        cluster_profile_path,
        output_dir,
    ) = resolve_paths(args)

    segment.ensure_paths(project_root, kmeans_path, cluster_profile_path)
    segment.begin_stage(project_root, output_dir)

    banner(PROGRAM_TITLE)

    print(
        f"Project Root        : {project_root}"
    )
    print(
        f"DW Input            : {dw_dir}"
    )
    print(
        f"Fact Sales          : {fact_sales_path}"
    )
    print(
        f"Dim Customer        : {dim_customer_path}"
    )
    print(
        f"Dim Time            : {dim_time_path}"
    )
    print(
        f"K-Means             : {kmeans_path}"
    )
    print(
        f"Cluster Names       : {cluster_profile_path}"
    )
    print(
        f"Output              : {output_dir}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    section(
        "1. LOADING INPUT"
    )

    sales = prepare_sales(
        read_csv(fact_sales_path)
    )

    customers = prepare_customers(
        read_csv(dim_customer_path)
    )

    time_dim = prepare_time(
        read_csv(dim_time_path)
    )

    kmeans = prepare_kmeans(
        read_csv(kmeans_path)
    )

    cluster_names = prepare_cluster_names(
        read_csv(cluster_profile_path)
    )

    print(
        f"Customers       : {len(customers):,}"
    )
    print(
        f"K-Means members : {len(kmeans):,}"
    )
    print(
        f"Named clusters  : {len(cluster_names):,}"
    )

    print()
    print("Cluster naming:")
    print(
        cluster_names
        .sort_values("Cluster")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Analysis Base
    # --------------------------------------------------------

    section(
        "2. BUILDING ANALYSIS BASE"
    )

    base = build_analysis_base(
        sales,
        customers,
        time_dim,
        kmeans,
        cluster_names,
    )

    if (base.Cluster.isna()).any():
        raise ValueError('購買客戶缺少共用基準期分群，請先重新執行K-Means及命名。')

    print(
        f"Sales rows           : {len(base):,}"
    )
    print(
        f"Unique orders        : {base['OrderID'].nunique():,}"
    )
    print(
        f"Purchasing customers : {base['CustomerID'].nunique():,}"
    )
    print(
        f"Date range           : "
        f"{base['Date'].min().date()} ~ "
        f"{base['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    section(
        "3. CALCULATING OPERATIONAL KPI"
    )

    overall = calculate_overall_kpi(
        base
    )

    cluster_kpi = calculate_cluster_kpi(
        base
    )

    quarterly_trend = (
        calculate_quarterly_trend(
            base
        )
    )

    cluster_quarterly_trend = (
        calculate_cluster_quarterly_trend(
            base
        )
    )

    dashboard_kpi = pd.concat(
        [
            overall,
            cluster_kpi,
        ],
        ignore_index=True,
        sort=False,
    )

    print("Overall KPI:")
    print(
        overall[
            [
                "TotalRevenue",
                "TotalOrders",
                "PurchasingCustomers",
                "AOV",
                "RevenueGrowthRate",
            ]
        ].to_string(index=False)
    )

    print()
    print("Cluster KPI:")
    print(
        cluster_kpi[
            [
                "ClusterName",
                "TotalRevenue",
                "RevenueShare",
                "TotalOrders",
                "PurchasingCustomers",
                "AOV",
                "RevenueGrowthRate",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    section(
        "4. SAVING OUTPUT"
    )

    overall_path = (
        output_dir
        / "Operational_Overall_KPI.csv"
    )

    cluster_path = (
        output_dir
        / "Operational_Cluster_KPI.csv"
    )

    quarterly_path = (
        output_dir
        / "Operational_Quarterly_Trend.csv"
    )

    cluster_quarterly_path = (
        output_dir
        / "Operational_Cluster_Quarterly_Trend.csv"
    )

    dashboard_path = (
        output_dir
        / "Operational_Dashboard_KPI.csv"
    )

    round_output(
        overall
    ).to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(
        cluster_kpi
    ).to_csv(
        cluster_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(
        quarterly_trend
    ).to_csv(
        quarterly_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(
        cluster_quarterly_trend
    ).to_csv(
        cluster_quarterly_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(
        dashboard_kpi
    ).to_csv(
        dashboard_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Overall KPI             : {overall_path}"
    )
    print(
        f"Cluster KPI             : {cluster_path}"
    )
    print(
        f"Quarterly Trend         : {quarterly_path}"
    )
    print(
        f"Cluster Quarterly Trend : {cluster_quarterly_path}"
    )
    print(
        f"Dashboard KPI           : {dashboard_path}"
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    section(
        "5. GENERATING STATIC CHECK CHARTS"
    )

    configure_matplotlib()

    chart_paths = [
        save_quarterly_revenue_chart(
            quarterly_trend,
            output_dir,
        ),
        save_quarterly_growth_chart(
            quarterly_trend,
            output_dir,
        ),
        save_cluster_revenue_chart(
            cluster_kpi,
            output_dir,
        ),
        save_cluster_revenue_share_chart(
            cluster_kpi,
            output_dir,
        ),
    ]

    for path in chart_paths:
        print(
            f"Saved Chart : {path}"
        )

    # --------------------------------------------------------
    # Dashboard
    # --------------------------------------------------------

    section(
        "6. STREAMLIT-READY OUTPUT"
    )

    print(
        "Overall:"
    )
    print(
        "  Operational_Overall_KPI.csv"
    )

    print(
        "Cluster comparison / single cluster:"
    )
    print(
        "  Operational_Cluster_KPI.csv"
    )

    print(
        "Overall quarterly trend:"
    )
    print(
        "  Operational_Quarterly_Trend.csv"
    )

    print(
        "Single-cluster quarterly trend:"
    )
    print(
        "  Operational_Cluster_Quarterly_Trend.csv"
    )

    print(
        "Unified KPI source:"
    )
    print(
        "  Operational_Dashboard_KPI.csv"
    )

    print()
    print(
        "Operational KPI Analysis completed successfully."
    )

    segment.finish_stage(project_root, output_dir)
    return 0


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        pd.errors.ParserError,
    ) as exc:
        print()
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)

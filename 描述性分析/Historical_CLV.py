from __future__ import annotations

"""
Historical CLV 描述性分析 Final
============================================================

本專題定義
----------
Historical CLV = 客戶於歷史觀察期間內已實現的累積實際利潤。

Customer-level:
    HistoricalCLV_i = SUM(Profit_ij)

其中 Profit 直接使用 Data Warehouse / Fact_Sales 中的實際 Profit 欄位。
程式同時驗證：
    Profit ≈ Amount - TotalCost

AOV / Purchase Frequency / Observed Lifetime 僅作為價值差異的解釋指標，
不參與 Historical CLV 主公式。

層級
----
1. Customer
2. Overall
3. Cluster

Historical CLV 與 Predictive CLV 分離：
- Historical CLV：描述性分析，衡量過去已實現利潤。
- Predictive CLV：由預測性分析的分群 XGBoost／兩階段模型預測未來12個月利潤。

預設專案結構
------------
專題v2/
├─ 原始資料/
│  └─ Data_Warehouse/
│     ├─ Fact_Sales.csv
│     ├─ Dim_Customer.csv
│     └─ Dim_Time.csv
├─ K-means分群/
│  ├─ K_Means/
│  │  └─ K_Means.csv
│  └─ Cluster/
│     └─ Cluster_Profile_Final.csv
└─ 描述性分析/
   ├─ Historical_CLV.py
   └─ Historical_CLV/
      ├─ Historical_CLV_Customer.csv
      ├─ Historical_CLV_Overall.csv
      ├─ Historical_CLV_Cluster.csv
      ├─ Historical_CLV_Dashboard.csv
      └─ Historical_CLV_Distribution.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment_contract as segment


PROGRAM_TITLE = "HISTORICAL CLV ANALYSIS - REALIZED PROFIT"


def banner(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)


def section(title: str) -> None:
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


def normalize_column(name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    table_name: str,
    required: bool = True,
) -> str | None:
    column_map = {normalize_column(c): c for c in df.columns}
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
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def clean_identifier(series: pd.Series) -> pd.Series:
    text = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    return text.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
    )


def safe_divide(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def round_output(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(4)
    return result


def resolve_paths(args: argparse.Namespace):
    script_dir = Path(__file__).resolve().parent
    project_root = (
        args.project_root.resolve()
        if args.project_root
        else script_dir.parent
    )

    dw_dir = project_root / "原始資料" / "Data_Warehouse"

    fact_sales = (
        args.fact_sales.resolve()
        if args.fact_sales
        else dw_dir / "Fact_Sales.csv"
    )
    dim_customer = (
        args.dim_customer.resolve()
        if args.dim_customer
        else dw_dir / "Dim_Customer.csv"
    )
    dim_time = (
        args.dim_time.resolve()
        if args.dim_time
        else dw_dir / "Dim_Time.csv"
    )
    kmeans = (
        args.kmeans.resolve()
        if args.kmeans
        else project_root / "K-means分群" / "K_Means" / "K_Means.csv"
    )
    cluster_profile = (
        args.cluster_profile.resolve()
        if args.cluster_profile
        else project_root / "K-means分群" / "Cluster" / "Cluster_Profile_Final.csv"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else script_dir / "Historical_CLV"
    )

    for path in (fact_sales, dim_customer, dim_time, kmeans, cluster_profile):
        if not path.exists():
            raise FileNotFoundError(f"找不到必要輸入檔案：{path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        project_root, dw_dir, fact_sales, dim_customer, dim_time,
        kmeans, cluster_profile, output_dir
    )


def prepare_customers(raw: pd.DataFrame) -> pd.DataFrame:
    key = find_column(raw, ("CustomerKey", "Customer_Key"), "Dim_Customer")
    cid = find_column(raw, ("CustomerID", "Customer_ID"), "Dim_Customer")
    result = raw[[key, cid]].copy()
    result.columns = ["CustomerKey", "CustomerID"]
    result["CustomerKey"] = clean_identifier(result["CustomerKey"])
    result["CustomerID"] = clean_identifier(result["CustomerID"])
    result = result.dropna().drop_duplicates("CustomerKey")
    if result["CustomerID"].duplicated().any():
        raise ValueError("Dim_Customer.CustomerID 必須唯一。")
    return result.reset_index(drop=True)


def prepare_time(raw: pd.DataFrame) -> pd.DataFrame:
    key = find_column(raw, ("DateKey", "Date_Key"), "Dim_Time")
    date = find_column(raw, ("Date", "FullDate"), "Dim_Time")
    result = raw[[key, date]].copy()
    result.columns = ["DateKey", "Date"]
    result["DateKey"] = clean_identifier(result["DateKey"])
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    result = result.dropna().drop_duplicates("DateKey")
    return result.reset_index(drop=True)


def prepare_kmeans(raw: pd.DataFrame) -> pd.DataFrame:
    cid = find_column(raw, ("CustomerID", "Customer_ID"), "K_Means.csv")
    cluster = find_column(
        raw, ("Cluster", "ClusterID", "Cluster_Label"), "K_Means.csv"
    )
    result = raw[[cid, cluster]].copy()
    result.columns = ["CustomerID", "Cluster"]
    result["CustomerID"] = clean_identifier(result["CustomerID"])
    result["Cluster"] = pd.to_numeric(result["Cluster"], errors="coerce")
    result = result.dropna()
    result["Cluster"] = result["Cluster"].astype(int)
    if result["CustomerID"].duplicated().any():
        raise ValueError("K_Means.csv 的 CustomerID 必須唯一。")
    return result.reset_index(drop=True)


def prepare_cluster_names(raw: pd.DataFrame) -> pd.DataFrame:
    cluster = find_column(
        raw, ("Cluster", "ClusterID", "Cluster_ID"), "Cluster_Profile_Final.csv"
    )
    name = find_column(
        raw,
        ("ClusterName", "Cluster_Name", "SegmentName", "CustomerSegment", "Name"),
        "Cluster_Profile_Final.csv",
    )
    result = raw[[cluster, name]].copy()
    result.columns = ["Cluster", "ClusterName"]
    result["Cluster"] = pd.to_numeric(result["Cluster"], errors="coerce")
    result["ClusterName"] = result["ClusterName"].astype("string").str.strip()
    result = result.dropna()
    result["Cluster"] = result["Cluster"].astype(int)
    if result["Cluster"].duplicated().any():
        raise ValueError("Cluster_Profile_Final.csv 的 Cluster 必須唯一。")
    return result.reset_index(drop=True)


def prepare_sales(raw: pd.DataFrame) -> pd.DataFrame:
    customer = find_column(raw, ("CustomerKey", "Customer_Key"), "Fact_Sales")
    date = find_column(raw, ("DateKey", "Date_Key"), "Fact_Sales")
    order = find_column(raw, ("OrderID", "Order_ID"), "Fact_Sales")
    amount = find_column(
        raw, ("Amount", "Revenue", "SalesAmount", "NetAmount"), "Fact_Sales"
    )
    total_cost = find_column(
        raw, ("TotalCost", "Cost", "Total_Cost"), "Fact_Sales"
    )
    profit = find_column(
        raw, ("Profit", "GrossProfit", "Gross_Profit"), "Fact_Sales"
    )

    result = raw[[customer, date, order, amount, total_cost, profit]].copy()
    result.columns = [
        "CustomerKey", "DateKey", "OrderID",
        "Revenue", "TotalCost", "Profit"
    ]

    for col in ("CustomerKey", "DateKey", "OrderID"):
        result[col] = clean_identifier(result[col])

    for col in ("Revenue", "TotalCost", "Profit"):
        result[col] = pd.to_numeric(result[col], errors="coerce")

    invalid = result[
        ["CustomerKey", "DateKey", "OrderID", "Revenue", "TotalCost", "Profit"]
    ].isna().any(axis=1)

    before = len(result)
    result = result.loc[~invalid].copy()

    print(f"Fact_Sales rows before cleaning : {before:,}")
    print(f"Fact_Sales rows after cleaning  : {len(result):,}")

    if result.empty:
        raise ValueError("Fact_Sales 清理後沒有有效交易。")

    # Data quality validation only. Profit remains the DW source of truth.
    expected_profit = result["Revenue"] - result["TotalCost"]
    difference = (result["Profit"] - expected_profit).abs()
    tolerance = 0.01
    mismatch = difference > tolerance

    print(f"Profit validation tolerance     : {tolerance}")
    print(f"Profit mismatch rows            : {int(mismatch.sum()):,}")
    print(f"Maximum profit difference       : {difference.max():,.6f}")

    if mismatch.any():
        raise ValueError(
            "Fact_Sales Profit 驗證失敗：存在 Profit != Revenue - TotalCost 的資料。"
        )

    return result


def build_customer_level(
    sales: pd.DataFrame,
    customers: pd.DataFrame,
    time_dim: pd.DataFrame,
    kmeans: pd.DataFrame,
    cluster_names: pd.DataFrame,
) -> pd.DataFrame:
    tx = sales.merge(
        customers, on="CustomerKey", how="left", validate="many_to_one"
    )
    tx = tx.merge(
        time_dim, on="DateKey", how="left", validate="many_to_one"
    )

    if tx["CustomerID"].isna().any():
        raise ValueError("部分 Fact_Sales.CustomerKey 無法對應 Dim_Customer。")
    if tx["Date"].isna().any():
        raise ValueError("部分 Fact_Sales.DateKey 無法對應 Dim_Time。")

    cluster_map = kmeans.merge(
        cluster_names, on="Cluster", how="left", validate="many_to_one"
    )
    if cluster_map["ClusterName"].isna().any():
        raise ValueError("部分 Cluster 找不到 ClusterName。")

    customer = (
        tx.groupby(["CustomerKey", "CustomerID"], as_index=False)
        .agg(
            TotalRevenue=("Revenue", "sum"),
            TotalCost=("TotalCost", "sum"),
            HistoricalCLV=("Profit", "sum"),
            TotalOrders=("OrderID", "nunique"),
            FirstPurchaseDate=("Date", "min"),
            LastPurchaseDate=("Date", "max"),
        )
    )

    customer["AOV"] = customer["TotalRevenue"] / customer["TotalOrders"]
    customer["PurchaseFrequency"] = customer["TotalOrders"]
    customer["ObservedLifetimeDays"] = (
        customer["LastPurchaseDate"] - customer["FirstPurchaseDate"]
    ).dt.days
    customer["ObservedLifetimeYears"] = (
        customer["ObservedLifetimeDays"] / 365.25
    )
    customer["ProfitMargin"] = np.where(
        customer["TotalRevenue"] != 0,
        customer["HistoricalCLV"] / customer["TotalRevenue"] * 100.0,
        np.nan,
    )

    customer = customer.merge(
        cluster_map, on="CustomerID", how="left", validate="one_to_one"
    )

    return customer


def summary_row(data: pd.DataFrame, scope: str,
                cluster=None, cluster_name=None) -> dict[str, Any]:
    total_profit = float(data["HistoricalCLV"].sum())
    total_revenue = float(data["TotalRevenue"].sum())
    total_cost = float(data["TotalCost"].sum())
    total_orders = int(data["TotalOrders"].sum())
    count = int(data["CustomerID"].nunique())

    return {
        "Scope": scope,
        "Cluster": cluster,
        "ClusterName": cluster_name,
        "DisplayName": "整體" if scope == "Overall" else cluster_name,
        "CustomerCount": count,
        "TotalHistoricalRevenue": total_revenue,
        "TotalHistoricalCost": total_cost,
        "TotalHistoricalProfit": total_profit,
        "AverageHistoricalCLV": float(data["HistoricalCLV"].mean()),
        "MedianHistoricalCLV": float(data["HistoricalCLV"].median()),
        "MinHistoricalCLV": float(data["HistoricalCLV"].min()),
        "MaxHistoricalCLV": float(data["HistoricalCLV"].max()),
        "AOV": safe_divide(total_revenue, total_orders),
        "AveragePurchaseFrequency": float(data["PurchaseFrequency"].mean()),
        "AverageObservedLifetimeYears": float(
            data["ObservedLifetimeYears"].mean()
        ),
        "ProfitMargin": (
            safe_divide(total_profit, total_revenue) * 100.0
            if total_revenue != 0 else np.nan
        ),
    }


def calculate_summaries(customer: pd.DataFrame):
    overall = pd.DataFrame([summary_row(customer, "Overall")])
    overall["ProfitShare"] = 100.0

    clustered = customer.dropna(subset=["Cluster", "ClusterName"]).copy()
    clustered["Cluster"] = clustered["Cluster"].astype(int)

    rows = []
    for (cluster, name), group in clustered.groupby(
        ["Cluster", "ClusterName"], sort=True, observed=True
    ):
        rows.append(summary_row(
            group, "Cluster", int(cluster), str(name)
        ))

    cluster_df = pd.DataFrame(rows)
    overall_profit = float(overall.iloc[0]["TotalHistoricalProfit"])

    cluster_df["ProfitShare"] = (
        cluster_df["TotalHistoricalProfit"] / overall_profit * 100.0
        if overall_profit != 0 else np.nan
    )

    dashboard = pd.concat(
        [overall, cluster_df], ignore_index=True, sort=False
    )
    return overall, cluster_df, dashboard



def build_distribution_source(customer: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit Historical CLV Distribution 的資料來源。

    保留每位購買客戶一列，可依 Cluster / ClusterName 篩選，
    不在分析層預先切固定區間，避免限制後續互動式圖表。
    """
    columns = [
        "CustomerID",
        "Cluster",
        "ClusterName",
        "HistoricalCLV",
        "TotalRevenue",
        "TotalCost",
        "TotalOrders",
        "AOV",
        "PurchaseFrequency",
        "ObservedLifetimeYears",
        "ProfitMargin",
    ]
    return customer[columns].copy()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical CLV：以歷史實際 Profit 衡量已實現客戶價值。"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--fact-sales", type=Path)
    parser.add_argument("--dim-customer", type=Path)
    parser.add_argument("--dim-time", type=Path)
    parser.add_argument("--kmeans", type=Path)
    parser.add_argument("--cluster-profile", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    (
        project_root, dw_dir, fact_sales_path, dim_customer_path,
        dim_time_path, kmeans_path, cluster_profile_path, output_dir
    ) = resolve_paths(args)

    segment.ensure_paths(project_root, kmeans_path, cluster_profile_path)
    segment.begin_stage(project_root, output_dir)

    banner(PROGRAM_TITLE)
    print("Project Root        :", project_root)
    print("DW Input            :", dw_dir)
    print("Fact Sales          :", fact_sales_path)
    print("Dim Customer        :", dim_customer_path)
    print("Dim Time            :", dim_time_path)
    print("K-Means             :", kmeans_path)
    print("Cluster Names       :", cluster_profile_path)
    print("Output              :", output_dir)

    section("1. LOADING AND VALIDATING INPUT")
    sales = prepare_sales(read_csv(fact_sales_path))
    customers = prepare_customers(read_csv(dim_customer_path))
    time_dim = prepare_time(read_csv(dim_time_path))
    kmeans = prepare_kmeans(read_csv(kmeans_path))
    cluster_names = prepare_cluster_names(read_csv(cluster_profile_path))

    section("2. BUILDING CUSTOMER HISTORICAL CLV")
    customer = build_customer_level(
        sales, customers, time_dim, kmeans, cluster_names
    )

    if (customer.Cluster.isna()).any():
        raise ValueError('購買客戶缺少共用基準期分群，請先重新執行K-Means及命名。')
    print(f"Purchasing customers : {len(customer):,}")
    print(f"Total historical profit: {customer['HistoricalCLV'].sum():,.2f}")

    section("3. CALCULATING OVERALL / CLUSTER SUMMARY")
    overall, cluster, dashboard = calculate_summaries(customer)
    distribution = build_distribution_source(customer)

    display = [
        "CustomerCount", "TotalHistoricalProfit",
        "AverageHistoricalCLV", "MedianHistoricalCLV",
        "AOV", "AveragePurchaseFrequency",
        "AverageObservedLifetimeYears", "ProfitMargin"
    ]

    print("Overall:")
    print(overall[display].to_string(index=False))

    print()
    print("By Customer Segment:")
    print(cluster[
        ["ClusterName", "ProfitShare", *display]
    ].to_string(index=False))

    section("4. SAVING OUTPUT")
    paths = {
        "Customer": output_dir / "Historical_CLV_Customer.csv",
        "Overall": output_dir / "Historical_CLV_Overall.csv",
        "Cluster": output_dir / "Historical_CLV_Cluster.csv",
        "Dashboard": output_dir / "Historical_CLV_Dashboard.csv",
        "Distribution": output_dir / "Historical_CLV_Distribution.csv",
    }

    customer_out = customer.copy()
    for col in ("FirstPurchaseDate", "LastPurchaseDate"):
        customer_out[col] = customer_out[col].dt.strftime("%Y-%m-%d")

    round_output(customer_out).to_csv(
        paths["Customer"], index=False, encoding="utf-8-sig"
    )
    round_output(overall).to_csv(
        paths["Overall"], index=False, encoding="utf-8-sig"
    )
    round_output(cluster).to_csv(
        paths["Cluster"], index=False, encoding="utf-8-sig"
    )
    round_output(dashboard).to_csv(
        paths["Dashboard"], index=False, encoding="utf-8-sig"
    )
    round_output(distribution).to_csv(
        paths["Distribution"], index=False, encoding="utf-8-sig"
    )

    for name, path in paths.items():
        print(f"{name:<10}: {path}")

    print()
    print("Historical CLV definition:")
    print("  Customer HistoricalCLV = SUM(Fact_Sales.Profit)")
    print("  AOV / Frequency / Lifetime are explanatory indicators only.")
    print()
    print("Dashboard primary KPI:")
    print("  Overall -> CustomerCount / TotalHistoricalProfit / AverageHistoricalCLV / MedianHistoricalCLV")
    print("  Cluster -> TotalHistoricalProfit / ProfitShare / AverageHistoricalCLV / MedianHistoricalCLV")
    print("  Drivers -> AOV / AveragePurchaseFrequency / AverageObservedLifetimeYears")
    print("  Distribution -> Historical_CLV_Distribution.csv")
    print()
    print("Historical CLV Analysis completed successfully.")
    segment.finish_stage(project_root, output_dir)
    return 0


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
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

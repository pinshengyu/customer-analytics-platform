# -*- coding: utf-8 -*-
"""
Cohort Analysis - Quarterly Period Retention

定義
----
1. CohortQuarter：
   依每位客戶「首次購買季度」分組。

2. QuarterIndex：
   Q0 = 首次購買季度
   Q1 = 首次購買後第 1 個季度
   Q2 = 首次購買後第 2 個季度
   ...

3. Period Retention：
   Retention(c, q)
   = Cohort c 在第 q 個季度有購買的客戶數 / Cohort c 初始客戶數

4. 本分析不結合 K-Means。
   僅比較不同首次購買季度 Cohort 的後續留存。

資料來源
--------
專題根目錄/
└─ 原始資料/
   └─ Data_Warehouse/
      ├─ Fact_Sales.csv
      ├─ Dim_Customer.csv
      └─ Dim_Time.csv

輸出
----
描述性分析/
└─ Cohort/
   ├─ Cohort_Retention_Matrix.csv
   ├─ Cohort_Retention_Long.csv
   ├─ Cohort_Size.csv
   └─ Cohort_Dashboard.csv

注意
----
- Fact_Sales 為 Order Line 粒度，因此購買行為先以
  CustomerKey + OrderID 去重，再判斷客戶在哪個季度有購買。
- Q0 理論上為 100%。
- 預設輸出 Q0 ~ Q4。
- 尚未經歷完整後續季度的 Cohort，其未成熟季度保留為空值，
  不當成 0% 留存。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM_TITLE = "COHORT ANALYSIS - QUARTERLY PERIOD RETENTION"
MAX_QUARTER_INDEX = 4


def print_header(title: str) -> None:
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_section(number: int, title: str) -> None:
    print()
    print("-" * 90)
    print(f"{number}. {title}")
    print("-" * 90)


def find_project_root(start: Path | None = None) -> Path:
    """
    從目前程式位置向上尋找專題根目錄。

    判斷依據：
    - 原始資料/Data_Warehouse 存在
    或
    - ETL_Output 存在
    """
    current = (start or Path(__file__).resolve().parent).resolve()

    candidates = [current, *current.parents]

    for candidate in candidates:
        if (candidate / "原始資料" / "Data_Warehouse").is_dir():
            return candidate

    for candidate in candidates:
        if (candidate / "ETL_Output").is_dir():
            return candidate

    raise FileNotFoundError(
        "找不到專題根目錄。請確認程式位於專題資料夾內，且存在 "
        "'原始資料/Data_Warehouse' 或 'ETL_Output'。"
    )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在：{path}")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} 缺少必要欄位：{missing}\n"
            f"目前欄位：{list(df.columns)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quarterly Cohort Period Retention Analysis"
    )
    parser.add_argument(
        "--max-quarter",
        type=int,
        default=MAX_QUARTER_INDEX,
        help="輸出的最大 QuarterIndex，預設 4（Q0~Q4）。",
    )
    return parser.parse_args()


def load_data(
    fact_sales_path: Path,
    dim_customer_path: Path,
    dim_time_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(fact_sales_path, "Fact_Sales")
    require_file(dim_customer_path, "Dim_Customer")
    require_file(dim_time_path, "Dim_Time")

    fact_sales = pd.read_csv(fact_sales_path, low_memory=False)
    dim_customer = pd.read_csv(dim_customer_path, low_memory=False)
    dim_time = pd.read_csv(dim_time_path, low_memory=False)

    require_columns(
        fact_sales,
        ["CustomerKey", "OrderID", "DateKey"],
        "Fact_Sales",
    )
    require_columns(
        dim_customer,
        ["CustomerKey", "CustomerID"],
        "Dim_Customer",
    )
    require_columns(
        dim_time,
        ["DateKey", "Date"],
        "Dim_Time",
    )

    return fact_sales, dim_customer, dim_time


def prepare_purchase_activity(
    fact_sales: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_time: pd.DataFrame,
) -> pd.DataFrame:
    """
    建立 Customer × Purchase Quarter 活動資料。

    Fact_Sales 是 line-level，因此先去除無效鍵值，
    再以 CustomerKey + OrderID 去重為訂單粒度。
    """
    sales = fact_sales[["CustomerKey", "OrderID", "DateKey"]].copy()

    before = len(sales)

    sales = sales.dropna(subset=["CustomerKey", "OrderID", "DateKey"]).copy()
    sales = sales.drop_duplicates(subset=["CustomerKey", "OrderID"], keep="first")

    time_map = dim_time[["DateKey", "Date"]].drop_duplicates("DateKey").copy()
    time_map["Date"] = pd.to_datetime(time_map["Date"], errors="coerce")

    if time_map["Date"].isna().any():
        bad = int(time_map["Date"].isna().sum())
        raise ValueError(f"Dim_Time.Date 有 {bad:,} 筆無法轉換為日期。")

    sales = sales.merge(
        time_map,
        on="DateKey",
        how="left",
        validate="many_to_one",
    )

    missing_date = int(sales["Date"].isna().sum())
    if missing_date:
        raise ValueError(
            f"Fact_Sales 有 {missing_date:,} 筆訂單找不到對應 Dim_Time 日期。"
        )

    valid_customer_keys = set(
        dim_customer["CustomerKey"].dropna().unique().tolist()
    )
    invalid_customer = ~sales["CustomerKey"].isin(valid_customer_keys)

    if invalid_customer.any():
        count = int(invalid_customer.sum())
        raise ValueError(
            f"Fact_Sales 有 {count:,} 筆訂單的 CustomerKey "
            "不存在於 Dim_Customer。"
        )

    sales["PurchaseQuarter"] = sales["Date"].dt.to_period("Q")

    # Cohort retention 只需要知道該客戶該季度是否有購買，
    # 同一季度購買多次仍只計為 1 位 Active Customer。
    activity = (
        sales[["CustomerKey", "PurchaseQuarter"]]
        .drop_duplicates()
        .sort_values(["CustomerKey", "PurchaseQuarter"])
        .reset_index(drop=True)
    )

    print(f"Fact_Sales rows                  : {before:,}")
    print(f"Unique orders                   : {len(sales):,}")
    print(f"Purchasing customers            : {sales['CustomerKey'].nunique():,}")
    print(f"Customer-quarter activity rows  : {len(activity):,}")
    print(
        "Purchase date range             : "
        f"{sales['Date'].min().date()} ~ {sales['Date'].max().date()}"
    )

    return activity


def add_cohort_information(activity: pd.DataFrame) -> pd.DataFrame:
    cohort = (
        activity.groupby("CustomerKey", as_index=False)["PurchaseQuarter"]
        .min()
        .rename(columns={"PurchaseQuarter": "CohortQuarter"})
    )

    result = activity.merge(
        cohort,
        on="CustomerKey",
        how="left",
        validate="many_to_one",
    )

    result["QuarterIndex"] = (
        (result["PurchaseQuarter"].dt.year - result["CohortQuarter"].dt.year) * 4
        + (result["PurchaseQuarter"].dt.quarter - result["CohortQuarter"].dt.quarter)
    ).astype(int)

    if (result["QuarterIndex"] < 0).any():
        raise ValueError("發現 QuarterIndex < 0，Cohort 計算異常。")

    return result


def calculate_cohort_retention(
    activity: pd.DataFrame,
    max_quarter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if max_quarter < 0:
        raise ValueError("--max-quarter 不可小於 0。")

    cohort_sizes = (
        activity.loc[activity["QuarterIndex"] == 0]
        .groupby("CohortQuarter")["CustomerKey"]
        .nunique()
        .rename("CohortSize")
        .reset_index()
        .sort_values("CohortQuarter")
        .reset_index(drop=True)
    )

    active = (
        activity.loc[activity["QuarterIndex"].between(0, max_quarter)]
        .groupby(["CohortQuarter", "QuarterIndex"])["CustomerKey"]
        .nunique()
        .rename("ActiveCustomers")
        .reset_index()
    )

    observed_last_quarter = activity["PurchaseQuarter"].max()

    # 建立完整 Cohort × Q0~Qmax 網格。
    # 若某 Cohort 尚未走到該季度，RetentionRate 保留 NaN；
    # 若已走到該季度但沒有任何客戶回購，才記為 0%。
    grid = pd.MultiIndex.from_product(
        [
            cohort_sizes["CohortQuarter"].tolist(),
            list(range(max_quarter + 1)),
        ],
        names=["CohortQuarter", "QuarterIndex"],
    ).to_frame(index=False)

    long_df = (
        grid.merge(active, on=["CohortQuarter", "QuarterIndex"], how="left")
        .merge(cohort_sizes, on="CohortQuarter", how="left")
    )

    long_df["TargetQuarter"] = (
        long_df["CohortQuarter"] + long_df["QuarterIndex"]
    )
    long_df["IsMature"] = long_df["TargetQuarter"] <= observed_last_quarter

    long_df["ActiveCustomers"] = np.where(
        long_df["IsMature"],
        long_df["ActiveCustomers"].fillna(0),
        np.nan,
    )

    long_df["RetentionRate"] = np.where(
        long_df["IsMature"],
        long_df["ActiveCustomers"] / long_df["CohortSize"] * 100.0,
        np.nan,
    )

    # Q0 必須為 100%，否則 cohort 建構有問題。
    q0 = long_df.loc[long_df["QuarterIndex"] == 0, "RetentionRate"]
    if not np.allclose(q0.to_numpy(dtype=float), 100.0):
        raise ValueError("Q0 Retention 並非全部 100%，請檢查 Cohort 建構。")

    # 顯示用字串。
    long_df["CohortQuarter"] = long_df["CohortQuarter"].astype(str)
    long_df["TargetQuarter"] = long_df["TargetQuarter"].astype(str)
    long_df["QuarterLabel"] = "Q" + long_df["QuarterIndex"].astype(str)

    cohort_sizes_out = cohort_sizes.copy()
    cohort_sizes_out["CohortQuarter"] = cohort_sizes_out["CohortQuarter"].astype(str)

    matrix = (
        long_df.pivot(
            index="CohortQuarter",
            columns="QuarterLabel",
            values="RetentionRate",
        )
        .reset_index()
    )

    desired_q_columns = [f"Q{i}" for i in range(max_quarter + 1)]
    for column in desired_q_columns:
        if column not in matrix.columns:
            matrix[column] = np.nan

    matrix = matrix[["CohortQuarter", *desired_q_columns]]
    matrix = cohort_sizes_out.merge(matrix, on="CohortQuarter", how="left")

    # Dashboard 採 long format，最適合 Streamlit 動態折線圖/Heatmap。
    dashboard = long_df[
        [
            "CohortQuarter",
            "CohortSize",
            "QuarterIndex",
            "QuarterLabel",
            "TargetQuarter",
            "ActiveCustomers",
            "RetentionRate",
            "IsMature",
        ]
    ].copy()

    long_out = dashboard.copy()

    return cohort_sizes_out, matrix, long_out, dashboard


def round_outputs(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    result[numeric_cols] = result[numeric_cols].round(4)
    return result


def main() -> None:
    args = parse_args()

    print_header(PROGRAM_TITLE)

    project_root = find_project_root()
    dw_dir = project_root / "原始資料" / "Data_Warehouse"
    output_dir = project_root / "描述性分析" / "Cohort"
    output_dir.mkdir(parents=True, exist_ok=True)

    fact_sales_path = dw_dir / "Fact_Sales.csv"
    dim_customer_path = dw_dir / "Dim_Customer.csv"
    dim_time_path = dw_dir / "Dim_Time.csv"

    print(f"Project Root        : {project_root}")
    print(f"DW Input            : {dw_dir}")
    print(f"Fact Sales          : {fact_sales_path}")
    print(f"Dim Customer        : {dim_customer_path}")
    print(f"Dim Time            : {dim_time_path}")
    print(f"Output              : {output_dir}")
    print(f"Quarter Range       : Q0 ~ Q{args.max_quarter}")

    print_section(1, "LOADING AND VALIDATING INPUT")
    fact_sales, dim_customer, dim_time = load_data(
        fact_sales_path,
        dim_customer_path,
        dim_time_path,
    )

    print_section(2, "BUILDING QUARTERLY CUSTOMER ACTIVITY")
    activity = prepare_purchase_activity(
        fact_sales,
        dim_customer,
        dim_time,
    )
    activity = add_cohort_information(activity)

    print_section(3, "CALCULATING COHORT RETENTION")
    cohort_size, matrix, long_df, dashboard = calculate_cohort_retention(
        activity,
        max_quarter=args.max_quarter,
    )

    print("Cohort Retention Matrix (%):")
    print(matrix.to_string(index=False))

    print_section(4, "SAVING OUTPUT")

    paths = {
        "Cohort Size": output_dir / "Cohort_Size.csv",
        "Retention Matrix": output_dir / "Cohort_Retention_Matrix.csv",
        "Retention Long": output_dir / "Cohort_Retention_Long.csv",
        "Dashboard": output_dir / "Cohort_Dashboard.csv",
    }

    round_outputs(cohort_size).to_csv(
        paths["Cohort Size"], index=False, encoding="utf-8-sig"
    )
    round_outputs(matrix).to_csv(
        paths["Retention Matrix"], index=False, encoding="utf-8-sig"
    )
    round_outputs(long_df).to_csv(
        paths["Retention Long"], index=False, encoding="utf-8-sig"
    )
    round_outputs(dashboard).to_csv(
        paths["Dashboard"], index=False, encoding="utf-8-sig"
    )

    for name, path in paths.items():
        print(f"{name:<16}: {path}")

    print()
    print("Cohort definition:")
    print("  CohortQuarter = customer's first purchase quarter")
    print("  Q0 = first purchase quarter (100%)")
    print("  Q1~Q4 = period retention in subsequent quarters")
    print("  RetentionRate = ActiveCustomers / CohortSize * 100")
    print("  Immature future quarters remain blank, not 0%.")
    print("  K-Means / Cluster results are not used.")
    print()
    print("Cohort Analysis completed successfully.")


if __name__ == "__main__":
    main()

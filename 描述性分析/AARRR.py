from __future__ import annotations

"""
AARRR 描述性分析 V2
============================================================

目的
----
1. 計算整體 AARRR KPI。
2. 延續 K-Means 分群結果，計算各客群 AARRR KPI。
3. 串接 Cluster_Profile_Final.csv 的 LLM 客群命名結果。
4. 輸出可直接供後續 Streamlit 使用的標準化資料。
5. 保留靜態 PNG 圖表，方便目前分析、報告與結果檢查。

AARRR 核心定義
--------------
Acquisition
    Dim_Customer 中的有效客戶數。

Activation Rate
    至少完成 1 筆有效訂單的客戶
    / Acquisition Customers。

Repeat Purchase Rate
    至少完成 2 筆「不同 OrderID」的客戶
    / Activated Customers。

    注意：
    這裡只衡量「是否再次購買」，不衡量時間留存。
    後續 Cohort Analysis 會另外使用季度 Q1/Q2/Q3/Q4
    Retention，因此兩者刻意分開。

Referral Rate
    在 Activated Customers 中，
    至少成功推薦 1 位新客戶的客戶
    / Activated Customers。

Revenue
    Fact_Sales.Amount 總和。

Streamlit 設計
--------------
後續 Dashboard 可直接支援：
- 整體
- 分群比較
- 單一客群

Cluster 編號保留作為資料關聯鍵；
展示時以 Cluster_Profile_Final.csv 的 ClusterName 為主。

預設專案結構
------------
專題v2/
├─ 原始資料/
│  └─ Data_Warehouse/
│     ├─ Dim_Customer.csv
│     ├─ Fact_Sales.csv
│     └─ Fact_Referral.csv
│
├─ K-means分群/
│  ├─ K_Means/
│  │  └─ K_Means.csv
│  └─ Cluster/
│     └─ Cluster_Profile_Final.csv
│
└─ 描述性分析/
   ├─ AARRR_Analysis.py
   └─ AARRR/
      ├─ AARRR_Overall_KPI.csv
      ├─ AARRR_Cluster_KPI.csv
      ├─ AARRR_Dashboard_KPI.csv
      ├─ AARRR_Customer_Base.csv
      ├─ AARRR_Stage_Diagnosis.csv
      ├─ AARRR_Lifecycle_Counts.png
      ├─ AARRR_Core_Rates.png
      ├─ AARRR_Cluster_Rates.png
      └─ AARRR_Cluster_Revenue.png
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



PROGRAM_TITLE = "AARRR CUSTOMER LIFECYCLE ANALYSIS V2"


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


def safe_rate(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return float(numerator) / float(denominator) * 100.0


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
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
    Path,
]:
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent

    if args.project_root:
        project_root = args.project_root.resolve()
    else:
        # AARRR_Analysis.py 放在：
        # 專題v2/描述性分析/AARRR_Analysis.py
        project_root = script_dir.parent

    dw_dir = project_root / "原始資料" / "Data_Warehouse"

    dim_customer_path = (
        args.dim_customer.resolve()
        if args.dim_customer
        else dw_dir / "Dim_Customer.csv"
    )

    fact_sales_path = (
        args.fact_sales.resolve()
        if args.fact_sales
        else dw_dir / "Fact_Sales.csv"
    )

    fact_referral_path = (
        args.fact_referral.resolve()
        if args.fact_referral
        else dw_dir / "Fact_Referral.csv"
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
        else script_dir / "AARRR"
    )

    required_paths = (
        dim_customer_path,
        fact_sales_path,
        fact_referral_path,
        kmeans_path,
        cluster_profile_path,
    )

    for path in required_paths:
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
        dim_customer_path,
        fact_sales_path,
        fact_referral_path,
        kmeans_path,
        cluster_profile_path,
        output_dir,
    )


# ============================================================
# 4. PREPARE DIM CUSTOMER
# ============================================================

def prepare_customers(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    key_col = find_column(
        raw,
        (
            "CustomerKey",
            "Customer_Key",
            "CustomerSK",
        ),
        "Dim_Customer",
    )

    id_col = find_column(
        raw,
        (
            "CustomerID",
            "Customer_ID",
            "CustomerBK",
        ),
        "Dim_Customer",
    )

    register_col = find_column(
        raw,
        (
            "RegisterDate",
            "RegistrationDate",
            "SignupDate",
        ),
        "Dim_Customer",
        required=False,
    )

    acquisition_col = find_column(
        raw,
        (
            "Acquisition_Channel",
            "AcquisitionChannel",
            "AcquisitionSource",
        ),
        "Dim_Customer",
        required=False,
    )

    columns = [key_col, id_col]

    if register_col:
        columns.append(register_col)

    if acquisition_col:
        columns.append(acquisition_col)

    customers = raw[columns].copy()

    rename_map = {
        key_col: "CustomerKey",
        id_col: "CustomerID",
    }

    if register_col:
        rename_map[register_col] = "RegisterDate"

    if acquisition_col:
        rename_map[acquisition_col] = (
            "AcquisitionChannel"
        )

    customers = customers.rename(
        columns=rename_map
    )

    customers["CustomerKey"] = clean_identifier(
        customers["CustomerKey"]
    )

    customers["CustomerID"] = clean_identifier(
        customers["CustomerID"]
    )

    customers = customers.dropna(
        subset=[
            "CustomerKey",
            "CustomerID",
        ]
    ).copy()

    if customers["CustomerKey"].duplicated().any():
        raise ValueError(
            "Dim_Customer.CustomerKey 必須唯一。"
        )

    if customers["CustomerID"].duplicated().any():
        raise ValueError(
            "Dim_Customer.CustomerID 必須唯一。"
        )

    if "RegisterDate" in customers.columns:
        customers["RegisterDate"] = pd.to_datetime(
            customers["RegisterDate"],
            errors="coerce",
        )

    return customers.reset_index(drop=True)


# ============================================================
# 5. PREPARE K-MEANS ASSIGNMENT
# ============================================================

def prepare_kmeans(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    customer_col = find_column(
        raw,
        (
            "CustomerID",
            "Customer_ID",
        ),
        "K_Means.csv",
    )

    cluster_col = find_column(
        raw,
        (
            "Cluster",
            "ClusterID",
            "Cluster_Label",
            "Label",
        ),
        "K_Means.csv",
    )

    result = raw[
        [
            customer_col,
            cluster_col,
        ]
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
        subset=[
            "CustomerID",
            "Cluster",
        ]
    ).copy()

    result["Cluster"] = (
        result["Cluster"].astype(int)
    )

    if result["CustomerID"].duplicated().any():
        raise ValueError(
            "K_Means.csv 的 CustomerID 必須唯一。"
        )

    return result.reset_index(drop=True)


# ============================================================
# 6. PREPARE CLUSTER NAMES
# ============================================================

def prepare_cluster_names(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    cluster_col = find_column(
        raw,
        (
            "Cluster",
            "ClusterID",
            "Cluster_ID",
            "ClusterLabel",
        ),
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
        [
            cluster_col,
            name_col,
        ]
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
        subset=[
            "Cluster",
            "ClusterName",
        ]
    ).copy()

    result["Cluster"] = (
        result["Cluster"].astype(int)
    )

    result = result[
        result["ClusterName"].ne("")
    ].copy()

    if result["Cluster"].duplicated().any():
        duplicated = (
            result.loc[
                result["Cluster"].duplicated(
                    keep=False
                ),
                "Cluster",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Cluster_Profile_Final.csv 中 Cluster "
            f"不可重複：{duplicated}"
        )

    return result.reset_index(drop=True)


# ============================================================
# 7. PREPARE FACT SALES
# ============================================================

def prepare_sales(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    customer_key_col = find_column(
        raw,
        (
            "CustomerKey",
            "Customer_Key",
            "CustomerSK",
        ),
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

    sales = raw[
        [
            customer_key_col,
            order_col,
            amount_col,
        ]
    ].copy()

    sales.columns = [
        "CustomerKey",
        "OrderID",
        "Amount",
    ]

    sales["CustomerKey"] = clean_identifier(
        sales["CustomerKey"]
    )

    sales["OrderID"] = clean_identifier(
        sales["OrderID"]
    )

    sales["Amount"] = pd.to_numeric(
        sales["Amount"],
        errors="coerce",
    )

    invalid = (
        sales[
            [
                "CustomerKey",
                "OrderID",
                "Amount",
            ]
        ]
        .isna()
        .any(axis=1)
        |
        (sales["Amount"] <= 0)
    )

    before = len(sales)
    sales = sales.loc[~invalid].copy()

    print(
        f"Fact_Sales rows before cleaning : "
        f"{before:,}"
    )

    print(
        f"Fact_Sales rows after cleaning  : "
        f"{len(sales):,}"
    )

    if sales.empty:
        raise ValueError(
            "Fact_Sales 清理後沒有有效資料。"
        )

    return sales


# ============================================================
# 8. PREPARE FACT REFERRAL
# ============================================================

def prepare_referral(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    referrer_col = find_column(
        raw,
        (
            "ReferrerCustomerKey",
            "Referrer_Customer_Key",
            "ReferrerKey",
        ),
        "Fact_Referral",
    )

    new_customer_col = find_column(
        raw,
        (
            "NewCustomerKey",
            "New_Customer_Key",
            "ReferredCustomerKey",
        ),
        "Fact_Referral",
        required=False,
    )

    selected = [referrer_col]

    if new_customer_col:
        selected.append(new_customer_col)

    referral = raw[selected].copy()

    rename_map = {
        referrer_col: "ReferrerCustomerKey",
    }

    if new_customer_col:
        rename_map[new_customer_col] = (
            "NewCustomerKey"
        )

    referral = referral.rename(
        columns=rename_map
    )

    referral["ReferrerCustomerKey"] = (
        clean_identifier(
            referral["ReferrerCustomerKey"]
        )
    )

    if "NewCustomerKey" in referral.columns:
        referral["NewCustomerKey"] = (
            clean_identifier(
                referral["NewCustomerKey"]
            )
        )

    referral = referral.dropna(
        subset=["ReferrerCustomerKey"]
    ).copy()

    return referral


# ============================================================
# 9. BUILD CUSTOMER-LEVEL AARRR BASE
# ============================================================

def build_customer_base(
    customers: pd.DataFrame,
    kmeans: pd.DataFrame,
    cluster_names: pd.DataFrame,
    sales: pd.DataFrame,
    referral: pd.DataFrame,
) -> pd.DataFrame:
    # 先把 Cluster assignment 與 LLM 命名結果合併。
    cluster_assignment = kmeans.merge(
        cluster_names,
        on="Cluster",
        how="left",
        validate="many_to_one",
    )

    missing_names = (
        cluster_assignment["ClusterName"]
        .isna()
    )

    if missing_names.any():
        missing_clusters = sorted(
            cluster_assignment.loc[
                missing_names,
                "Cluster",
            ]
            .drop_duplicates()
            .astype(int)
            .tolist()
        )

        raise ValueError(
            "以下 Cluster 在 Cluster_Profile_Final.csv "
            f"找不到命名結果：{missing_clusters}"
        )

    base = customers.merge(
        cluster_assignment,
        on="CustomerID",
        how="left",
        validate="one_to_one",
    )

    # Fact_Sales 是 Order Line 粒度，
    # 因此 Frequency / OrderCount 必須使用唯一 OrderID。
    customer_sales = (
        sales
        .groupby(
            "CustomerKey",
            as_index=False,
        )
        .agg(
            OrderCount=("OrderID", "nunique"),
            Revenue=("Amount", "sum"),
        )
    )

    # 每位客戶成功推薦的筆數。
    customer_referral = (
        referral
        .groupby(
            "ReferrerCustomerKey",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "ReferrerCustomerKey":
                    "CustomerKey",
                "size":
                    "ReferralCount",
            }
        )
    )

    base = base.merge(
        customer_sales,
        on="CustomerKey",
        how="left",
        validate="one_to_one",
    )

    base = base.merge(
        customer_referral,
        on="CustomerKey",
        how="left",
        validate="one_to_one",
    )

    base["OrderCount"] = (
        base["OrderCount"]
        .fillna(0)
        .astype(int)
    )

    base["Revenue"] = (
        base["Revenue"]
        .fillna(0.0)
        .astype(float)
    )

    base["ReferralCount"] = (
        base["ReferralCount"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # AARRR 狀態旗標
    # --------------------------------------------------------

    base["IsAcquired"] = 1

    base["IsActivated"] = (
        base["OrderCount"] >= 1
    ).astype(int)

    base["IsRepeatPurchaser"] = (
        base["OrderCount"] >= 2
    ).astype(int)

    # Referral Rate 的分母是 Activated Customers，
    # 因此 numerator 也限定為 Activated 且成功推薦的客戶，
    # 確保生命週期邏輯一致。
    base["IsReferrer"] = (
        (base["IsActivated"] == 1)
        &
        (base["ReferralCount"] >= 1)
    ).astype(int)

    return base


# ============================================================
# 10. CALCULATE AARRR KPI
# ============================================================

def calculate_kpi(
    data: pd.DataFrame,
    scope: str,
    cluster: int | None = None,
    cluster_name: str | None = None,
) -> dict[str, Any]:
    acquisition = int(
        data["IsAcquired"].sum()
    )

    activated = int(
        data["IsActivated"].sum()
    )

    repeat = int(
        data["IsRepeatPurchaser"].sum()
    )

    referrers = int(
        data["IsReferrer"].sum()
    )

    revenue = float(
        data["Revenue"].sum()
    )

    return {
        "Scope": scope,
        "Cluster": cluster,
        "ClusterName": cluster_name,
        "DisplayName": (
            "整體"
            if scope == "Overall"
            else cluster_name
        ),
        "AcquisitionCustomers": acquisition,
        "ActivatedCustomers": activated,
        "ActivationRate": safe_rate(
            activated,
            acquisition,
        ),
        "RepeatPurchaseCustomers": repeat,
        "RepeatPurchaseRate": safe_rate(
            repeat,
            activated,
        ),
        "ReferrerCustomers": referrers,
        "ReferralRate": safe_rate(
            referrers,
            activated,
        ),
        "Revenue": revenue,
        "RevenuePerActivatedCustomer": (
            revenue / activated
            if activated > 0
            else np.nan
        ),
    }


def calculate_aarrr(
    base: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------
    overall = pd.DataFrame(
        [
            calculate_kpi(
                base,
                scope="Overall",
            )
        ]
    )

    # --------------------------------------------------------
    # Cluster
    # --------------------------------------------------------
    # K-Means 是由有交易紀錄的客戶建立，
    # 因此 Cluster 層級的 Activation Rate 通常會接近或等於 100%。
    # 仍保留此欄位，以維持 AARRR 結構一致；
    # Dashboard 解讀時，Cluster 比較應更重視
    # Repeat Purchase / Referral / Revenue。
    clustered = base.dropna(
        subset=[
            "Cluster",
            "ClusterName",
        ]
    ).copy()

    clustered["Cluster"] = (
        clustered["Cluster"].astype(int)
    )

    cluster_rows: list[dict[str, Any]] = []

    for (
        cluster,
        cluster_name,
    ), group in clustered.groupby(
        [
            "Cluster",
            "ClusterName",
        ],
        sort=True,
        observed=True,
    ):
        cluster_rows.append(
            calculate_kpi(
                group,
                scope="Cluster",
                cluster=int(cluster),
                cluster_name=str(cluster_name),
            )
        )

    cluster_kpi = pd.DataFrame(
        cluster_rows
    )

    if cluster_kpi.empty:
        raise ValueError(
            "沒有可用的 Cluster AARRR 結果。"
        )

    total_cluster_revenue = float(
        cluster_kpi["Revenue"].sum()
    )

    cluster_kpi["RevenueShare"] = (
        cluster_kpi["Revenue"]
        / total_cluster_revenue
        * 100.0
        if total_cluster_revenue > 0
        else np.nan
    )

    total_cluster_customers = int(
        cluster_kpi[
            "AcquisitionCustomers"
        ].sum()
    )

    cluster_kpi["CustomerShare"] = (
        cluster_kpi[
            "AcquisitionCustomers"
        ]
        / total_cluster_customers
        * 100.0
        if total_cluster_customers > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Dashboard-ready unified KPI
    # --------------------------------------------------------
    # Overall + 所有 Cluster 使用同一套欄位，
    # Streamlit 可以直接依 Scope / Cluster / DisplayName 篩選。
    dashboard_kpi = pd.concat(
        [
            overall,
            cluster_kpi,
        ],
        ignore_index=True,
        sort=False,
    )

    return (
        overall,
        cluster_kpi,
        dashboard_kpi,
    )


# ============================================================
# 11. STAGE DIAGNOSIS
# ============================================================

def build_stage_diagnosis(
    overall: pd.DataFrame,
) -> pd.DataFrame:
    row = overall.iloc[0]

    stages = pd.DataFrame(
        [
            {
                "Stage": "Acquisition",
                "Metric": "AcquisitionCustomers",
                "Value":
                    row["AcquisitionCustomers"],
                "Unit": "Customers",
                "Definition":
                    "Dim_Customer 中有效客戶數",
            },
            {
                "Stage": "Activation",
                "Metric": "ActivationRate",
                "Value":
                    row["ActivationRate"],
                "Unit": "Percent",
                "Definition":
                    "至少 1 筆訂單客戶 / Acquisition Customers",
            },
            {
                "Stage": "Repeat Purchase",
                "Metric": "RepeatPurchaseRate",
                "Value":
                    row["RepeatPurchaseRate"],
                "Unit": "Percent",
                "Definition":
                    "至少 2 筆不同訂單客戶 / Activated Customers",
            },
            {
                "Stage": "Referral",
                "Metric": "ReferralRate",
                "Value":
                    row["ReferralRate"],
                "Unit": "Percent",
                "Definition":
                    "Activated 且至少成功推薦 1 人客戶 / Activated Customers",
            },
            {
                "Stage": "Revenue",
                "Metric": "Revenue",
                "Value":
                    row["Revenue"],
                "Unit": "Currency",
                "Definition":
                    "Fact_Sales.Amount 總和",
            },
        ]
    )

    # Acquisition 是人數、Revenue 是金額，
    # 不應與百分比直接比較。
    # 因此只在三個 rate 中標記最低值。
    rate_mask = stages["Unit"].eq(
        "Percent"
    )

    stages["IsLowestRateStage"] = False

    valid_rates = stages.loc[
        rate_mask
        & stages["Value"].notna(),
        "Value",
    ]

    if not valid_rates.empty:
        lowest_index = (
            valid_rates.astype(float).idxmin()
        )

        stages.loc[
            lowest_index,
            "IsLowestRateStage",
        ] = True

    return stages


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


def save_lifecycle_counts_chart(
    overall: pd.DataFrame,
    output_dir: Path,
) -> Path:
    row = overall.iloc[0]

    labels = [
        "Acquisition",
        "Activated",
        "Repeat Purchase",
        "Referral",
    ]

    values = [
        row["AcquisitionCustomers"],
        row["ActivatedCustomers"],
        row["RepeatPurchaseCustomers"],
        row["ReferrerCustomers"],
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "AARRR Customer Lifecycle Counts"
    )
    ax.set_ylabel(
        "Customers"
    )
    ax.set_ylim(
        bottom=0
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    path = (
        output_dir
        / "AARRR_Lifecycle_Counts.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def save_core_rates_chart(
    overall: pd.DataFrame,
    output_dir: Path,
) -> Path:
    row = overall.iloc[0]

    labels = [
        "Activation Rate",
        "Repeat Purchase Rate",
        "Referral Rate",
    ]

    values = [
        row["ActivationRate"],
        row["RepeatPurchaseRate"],
        row["ReferralRate"],
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(
        "AARRR Core Rates"
    )
    ax.set_ylabel(
        "Rate (%)"
    )
    ax.set_ylim(
        0,
        100,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        if pd.notna(value):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()

    path = (
        output_dir
        / "AARRR_Core_Rates.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def save_cluster_rates_chart(
    cluster_kpi: pd.DataFrame,
    output_dir: Path,
) -> Path:
    data = (
        cluster_kpi
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    x = np.arange(
        len(data)
    )

    width = 0.26

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        x - width,
        data["ActivationRate"],
        width,
        label="Activation Rate",
    )

    ax.bar(
        x,
        data["RepeatPurchaseRate"],
        width,
        label="Repeat Purchase Rate",
    )

    ax.bar(
        x + width,
        data["ReferralRate"],
        width,
        label="Referral Rate",
    )

    ax.set_title(
        "AARRR Rates by Customer Segment"
    )
    ax.set_ylabel(
        "Rate (%)"
    )
    ax.set_xlabel(
        "Customer Segment"
    )

    ax.set_xticks(
        x,
        data["ClusterName"].astype(str),
        rotation=15,
        ha="right",
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        / "AARRR_Cluster_Rates.png"
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
            "Revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    labels = (
        data["ClusterName"]
        .astype(str)
        .tolist()
    )

    values = (
        data["Revenue"]
        .astype(float)
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        labels,
        values,
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
        values,
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
        / "AARRR_Cluster_Revenue.png"
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
            "建立整體、分群比較與單群可用的 "
            "AARRR 描述性分析資料。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help=(
            "專案根目錄；預設為 "
            "描述性分析/AARRR_Analysis.py 的上一層。"
        ),
    )

    parser.add_argument(
        "--dim-customer",
        type=Path,
        help="Dim_Customer.csv 完整路徑。",
    )

    parser.add_argument(
        "--fact-sales",
        type=Path,
        help="Fact_Sales.csv 完整路徑。",
    )

    parser.add_argument(
        "--fact-referral",
        type=Path,
        help="Fact_Referral.csv 完整路徑。",
    )

    parser.add_argument(
        "--kmeans",
        type=Path,
        help="K_Means.csv 完整路徑。",
    )

    parser.add_argument(
        "--cluster-profile",
        type=Path,
        help=(
            "Cluster_Profile_Final.csv 完整路徑。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "輸出資料夾；預設為 "
            "描述性分析/AARRR。"
        ),
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
        dim_customer_path,
        fact_sales_path,
        fact_referral_path,
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
        f"Customer Input      : {dim_customer_path}"
    )
    print(
        f"Sales Input         : {fact_sales_path}"
    )
    print(
        f"Referral Input      : {fact_referral_path}"
    )
    print(
        f"K-Means Input       : {kmeans_path}"
    )
    print(
        f"Cluster Name Input  : {cluster_profile_path}"
    )
    print(
        f"Output Dir          : {output_dir}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    section(
        "1. LOADING AND VALIDATING INPUT"
    )

    customers = prepare_customers(
        read_csv(dim_customer_path)
    )

    kmeans = prepare_kmeans(
        read_csv(kmeans_path)
    )

    cluster_names = prepare_cluster_names(
        read_csv(cluster_profile_path)
    )

    sales = prepare_sales(
        read_csv(fact_sales_path)
    )

    referral = prepare_referral(
        read_csv(fact_referral_path)
    )

    print(
        f"Dim_Customer customers : "
        f"{len(customers):,}"
    )

    print(
        f"K-Means customers      : "
        f"{len(kmeans):,}"
    )

    print(
        f"Cluster count          : "
        f"{kmeans['Cluster'].nunique():,}"
    )

    print(
        f"Named clusters         : "
        f"{len(cluster_names):,}"
    )

    print()
    print("Cluster naming:")
    print(
        cluster_names
        .sort_values("Cluster")
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Customer Base
    # --------------------------------------------------------

    section(
        "2. BUILDING CUSTOMER AARRR BASE"
    )

    base = build_customer_base(
        customers,
        kmeans,
        cluster_names,
        sales,
        referral,
    )

    if (base.IsActivated.eq(1) & base.Cluster.isna()).any():
        raise ValueError('購買客戶缺少共用基準期分群，請先重新執行K-Means及命名。')

    print(
        f"Customer base rows : {len(base):,}"
    )

    print(
        "Activated customers: "
        f"{base['IsActivated'].sum():,}"
    )

    print(
        "Repeat purchasers  : "
        f"{base['IsRepeatPurchaser'].sum():,}"
    )

    print(
        "Activated referrers: "
        f"{base['IsReferrer'].sum():,}"
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    section(
        "3. CALCULATING AARRR KPI"
    )

    (
        overall,
        cluster_kpi,
        dashboard_kpi,
    ) = calculate_aarrr(base)

    diagnosis = build_stage_diagnosis(
        overall
    )

    display_columns = [
        "AcquisitionCustomers",
        "ActivationRate",
        "RepeatPurchaseRate",
        "ReferralRate",
        "Revenue",
    ]

    print("Overall:")
    print(
        overall[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("By Customer Segment:")
    print(
        cluster_kpi[
            [
                "ClusterName",
                *display_columns,
            ]
        ].to_string(index=False)
    )

    weakest = diagnosis.loc[
        diagnosis["IsLowestRateStage"]
    ]

    if not weakest.empty:
        weak_row = weakest.iloc[0]

        print()
        print(
            "Lowest comparable rate stage : "
            f"{weak_row['Stage']} "
            f"({weak_row['Value']:.2f}%)"
        )

    # --------------------------------------------------------
    # CSV Output
    # --------------------------------------------------------

    section(
        "4. SAVING ANALYSIS OUTPUT"
    )

    overall_path = (
        output_dir
        / "AARRR_Overall_KPI.csv"
    )

    cluster_path = (
        output_dir
        / "AARRR_Cluster_KPI.csv"
    )

    dashboard_path = (
        output_dir
        / "AARRR_Dashboard_KPI.csv"
    )

    customer_base_path = (
        output_dir
        / "AARRR_Customer_Base.csv"
    )

    diagnosis_path = (
        output_dir
        / "AARRR_Stage_Diagnosis.csv"
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
        dashboard_kpi
    ).to_csv(
        dashboard_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Customer-level base 保留給未來 Streamlit
    # 做更細的互動篩選，不需重新計算 AARRR。
    base_output = base.copy()

    if "RegisterDate" in base_output.columns:
        base_output["RegisterDate"] = (
            base_output["RegisterDate"]
            .dt.strftime("%Y-%m-%d")
        )

    round_output(
        base_output
    ).to_csv(
        customer_base_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(
        diagnosis
    ).to_csv(
        diagnosis_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Overall KPI     : {overall_path}"
    )

    print(
        f"Cluster KPI     : {cluster_path}"
    )

    print(
        f"Dashboard KPI   : {dashboard_path}"
    )

    print(
        f"Customer Base   : {customer_base_path}"
    )

    print(
        f"Stage Diagnosis : {diagnosis_path}"
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    section(
        "5. GENERATING STATIC CHECK CHARTS"
    )

    configure_matplotlib()

    chart_paths = [
        save_lifecycle_counts_chart(
            overall,
            output_dir,
        ),
        save_core_rates_chart(
            overall,
            output_dir,
        ),
        save_cluster_rates_chart(
            cluster_kpi,
            output_dir,
        ),
        save_cluster_revenue_chart(
            cluster_kpi,
            output_dir,
        ),
    ]

    for path in chart_paths:
        print(
            f"Saved Chart : {path}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    section(
        "6. STREAMLIT-READY OUTPUT"
    )

    print(
        "Dashboard mode support:"
    )
    print(
        "  Overall         -> "
        "AARRR_Overall_KPI.csv"
    )
    print(
        "  Cluster Compare -> "
        "AARRR_Cluster_KPI.csv"
    )
    print(
        "  Single Cluster  -> "
        "filter AARRR_Cluster_KPI.csv by ClusterName"
    )
    print(
        "  Unified Source  -> "
        "AARRR_Dashboard_KPI.csv"
    )
    print(
        "  Future Filters  -> "
        "AARRR_Customer_Base.csv"
    )

    print()
    print(
        "AARRR Analysis completed successfully."
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

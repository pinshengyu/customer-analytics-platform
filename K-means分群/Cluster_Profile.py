from __future__ import annotations

"""
Cluster Profile 描述性分析

實際專案結構：
專案根目錄/
├─ Data_Warehouse/
└─ K-means分群/
   ├─ K_Means_Clustering.py
   ├─ Cluster_Profile.py
   └─ K_Means/              # 分群結果統一存放位置
      ├─ K_Means.csv
      ├─ Cluster_Profile.csv
      ├─ Cluster_Summary.csv
      ├─ Customer_RFM.csv
      └─ K_Silhouette_Score.csv

本程式輸出到 K-means分群/Cluster/：
- Cluster_Profile_KPI.csv
- Cluster_Overall_KPI.csv
- Cluster_Profile_LLM.json
- Cluster_Profile_Final.csv

流程：
1. 讀取 K_Means.csv 與 Data_Warehouse/Fact_Sales.csv。
2. 計算 Cluster 規模、RFM、訂單、營收、AOV 與整體基準差異。
3. Python 自動比較各群，Ollama 依可用特徵詞選擇名稱及命名依據。
   特徵與描述由 Python 依當次 KPI 產生，輸出 DescriptionSource 記錄來源。
4. 驗證 LLM JSON 輸出。
5. LLM 失敗時明確報錯，不使用規則式名稱冒充 LLM 結果。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment_contract as segment



# ============================================================
# 1. CONFIG
# ============================================================

PROGRAM_TITLE = "CLUSTER PROFILE ANALYSIS + OLLAMA NAMING V6"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


# ============================================================
# 2. DISPLAY
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
# 3. BASIC UTILITIES
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
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def clean_identifier(series: pd.Series) -> pd.Series:
    """統一 1、1.0、'1' 等常見 CSV 主鍵格式。"""
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


def safe_ratio(
    numerator: pd.Series | float,
    denominator: float,
) -> pd.Series | float:
    if pd.isna(denominator) or denominator == 0:
        if isinstance(numerator, pd.Series):
            return pd.Series(np.nan, index=numerator.index)
        return np.nan

    return numerator / denominator


# ============================================================
# 4. PATHS
# ============================================================

def resolve_paths(args):
    project_root = args.project_root.resolve() if args.project_root else segment.project_root(__file__)
    dw_dir = project_root / '原始資料' / 'Data_Warehouse'
    kmeans_path = (project_root / 'K-means分群' / 'K_Means' / 'K_Means.csv')
    if args.kmeans and args.kmeans.resolve() != kmeans_path.resolve():
        raise ValueError('只能使用全專案共用K_Means.csv。')
    fact_sales_path = dw_dir / 'Fact_Sales.csv'
    if args.fact_sales and args.fact_sales.resolve() != fact_sales_path.resolve():
        raise ValueError('命名必須使用同一DW的基準期交易。')
    output_dir = project_root / 'K-means分群' / 'Cluster'
    if args.output_dir and args.output_dir.resolve() != output_dir.resolve():
        raise ValueError('命名只能寫入全專案共用Cluster資料夾。')
    segment.manifest(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / 'Naming_Manifest.json'
    if marker.exists(): marker.unlink()
    return project_root, dw_dir, kmeans_path, fact_sales_path, output_dir


# ============================================================
# 5. PREPARE K-MEANS DATA
# ============================================================

def prepare_kmeans(raw: pd.DataFrame) -> pd.DataFrame:
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

    recency_col = find_column(
        raw,
        ("Recency",),
        "K_Means.csv",
    )

    frequency_col = find_column(
        raw,
        ("Frequency",),
        "K_Means.csv",
    )

    monetary_col = find_column(
        raw,
        ("Monetary",),
        "K_Means.csv",
    )

    data = raw[
        [
            customer_col,
            cluster_col,
            recency_col,
            frequency_col,
            monetary_col,
        ]
    ].copy()

    data.columns = [
        "CustomerID",
        "Cluster",
        "Recency",
        "Frequency",
        "Monetary",
    ]

    data["CustomerID"] = clean_identifier(data["CustomerID"])

    for column in (
        "Cluster",
        "Recency",
        "Frequency",
        "Monetary",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    invalid = data[
        [
            "CustomerID",
            "Cluster",
            "Recency",
            "Frequency",
            "Monetary",
        ]
    ].isna().any(axis=1)

    if invalid.any():
        print(
            f"Warning: K_Means.csv 移除 {int(invalid.sum()):,} 筆必要欄位無效資料"
        )
        data = data.loc[~invalid].copy()

    if data.empty:
        raise ValueError("K_Means.csv 清理後沒有可分析資料。")

    if data["CustomerID"].duplicated().any():
        duplicate_count = int(data["CustomerID"].duplicated().sum())
        raise ValueError(
            "K_Means.csv 的 CustomerID 必須唯一。\n"
            f"目前重複筆數：{duplicate_count:,}"
        )

    if (data["Recency"] < 0).any():
        raise ValueError("Recency 不得出現負數。")

    if (data["Frequency"] <= 0).any():
        raise ValueError("Frequency 必須大於 0。")

    if (data["Monetary"] <= 0).any():
        raise ValueError("Monetary 必須大於 0。")

    data["Cluster"] = data["Cluster"].astype(int)

    return (
        data
        .sort_values("CustomerID")
        .reset_index(drop=True)
    )


# ============================================================
# 6. CUSTOMER KEY LOOKUP
# ============================================================

def build_customer_key_map(dw_dir: Path) -> pd.DataFrame:
    dim_path = dw_dir / "Dim_Customer.csv"

    if not dim_path.exists():
        raise FileNotFoundError(
            "Fact_Sales 只有 CustomerKey，但找不到 Dim_Customer.csv：\n"
            f"{dim_path}"
        )

    dim = read_csv(dim_path)

    key_col = find_column(
        dim,
        ("CustomerKey", "Customer_Key", "CustomerSK"),
        "Dim_Customer",
    )

    id_col = find_column(
        dim,
        ("CustomerID", "Customer_ID", "CustomerBK", "CustomerCode"),
        "Dim_Customer",
    )

    result = dim[
        [
            key_col,
            id_col,
        ]
    ].copy()

    result.columns = [
        "CustomerKey",
        "CustomerID",
    ]

    result["CustomerKey"] = clean_identifier(result["CustomerKey"])
    result["CustomerID"] = clean_identifier(result["CustomerID"])

    result = (
        result
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if result["CustomerKey"].duplicated().any():
        raise ValueError(
            "Dim_Customer.CustomerKey 無法唯一對應 CustomerID。"
        )

    return result


# ============================================================
# 7. PREPARE SALES DATA
# ============================================================

def prepare_sales(
    raw: pd.DataFrame,
    dw_dir: Path,
) -> pd.DataFrame:
    customer_id_col = find_column(
        raw,
        ("CustomerID", "Customer_ID"),
        "Fact_Sales",
        required=False,
    )

    customer_key_col = find_column(
        raw,
        ("CustomerKey", "Customer_Key", "CustomerSK"),
        "Fact_Sales",
        required=False,
    )

    order_col = find_column(
        raw,
        ("OrderID", "Order_ID", "SalesOrderID", "TransactionID"),
        "Fact_Sales",
    )

    amount_col = find_column(
        raw,
        ("Amount", "SalesAmount", "Revenue", "TotalAmount", "OrderAmount"),
        "Fact_Sales",
    )

    if customer_id_col is None and customer_key_col is None:
        raise ValueError(
            "Fact_Sales 同時缺少 CustomerID 與 CustomerKey。"
        )

    selected_customer_col = (
        customer_id_col
        if customer_id_col is not None
        else customer_key_col
    )

    sales = raw[
        [
            order_col,
            amount_col,
            selected_customer_col,
        ]
    ].copy()

    sales = sales.rename(
        columns={
            order_col: "OrderID",
            amount_col: "Amount",
        }
    )

    sales["OrderID"] = clean_identifier(sales["OrderID"])
    sales["Amount"] = pd.to_numeric(
        sales["Amount"],
        errors="coerce",
    )

    if customer_id_col is not None:
        sales = sales.rename(
            columns={customer_id_col: "CustomerID"}
        )
        sales["CustomerID"] = clean_identifier(
            sales["CustomerID"]
        )

    else:
        sales = sales.rename(
            columns={customer_key_col: "CustomerKey"}
        )
        sales["CustomerKey"] = clean_identifier(
            sales["CustomerKey"]
        )

        customer_map = build_customer_key_map(dw_dir)

        sales = sales.merge(
            customer_map,
            on="CustomerKey",
            how="left",
            validate="many_to_one",
        )

        missing_customer = int(
            sales["CustomerID"].isna().sum()
        )

        if missing_customer > 0:
            raise ValueError(
                "Fact_Sales 中存在無法由 CustomerKey 對應 CustomerID 的資料。\n"
                f"未對應筆數：{missing_customer:,}"
            )

    before_rows = len(sales)

    invalid = (
        sales[
            [
                "CustomerID",
                "OrderID",
                "Amount",
            ]
        ]
        .isna()
        .any(axis=1)
        |
        (sales["Amount"] <= 0)
    )

    sales = sales.loc[
        ~invalid,
        [
            "CustomerID",
            "OrderID",
            "Amount",
        ],
    ].copy()

    print(
        f"Fact_Sales rows before cleaning : {before_rows:,}"
    )
    print(
        f"Fact_Sales rows after cleaning  : {len(sales):,}"
    )

    if sales.empty:
        raise ValueError(
            "Fact_Sales 清理後沒有有效銷售資料。"
        )

    return sales


# ============================================================
# 8. CALCULATE CLUSTER PROFILE
# ============================================================

def calculate_profiles(
    kmeans: pd.DataFrame,
    sales: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assignments = kmeans[
        [
            "CustomerID",
            "Cluster",
        ]
    ].copy()

    sales_clustered = sales.merge(
        assignments,
        on="CustomerID",
        how="left",
        validate="many_to_one",
    )

    unmatched_rows = int(
        sales_clustered["Cluster"].isna().sum()
    )

    unmatched_revenue = float(
        sales_clustered.loc[
            sales_clustered["Cluster"].isna(),
            "Amount",
        ].sum()
    )

    matched_sales = (
        sales_clustered
        .dropna(subset=["Cluster"])
        .copy()
    )

    matched_sales["Cluster"] = (
        matched_sales["Cluster"].astype(int)
    )

    if matched_sales.empty:
        raise ValueError(
            "Fact_Sales 與 K_Means.csv 沒有任何可對應的 CustomerID。"
        )

    # --------------------------------------------------------
    # RFM Profile
    # --------------------------------------------------------

    rfm_profile = (
        kmeans
        .groupby("Cluster", as_index=False)
        .agg(
            CustomerCount=("CustomerID", "nunique"),
            AvgRecency=("Recency", "mean"),
            MedianRecency=("Recency", "median"),
            AvgFrequency=("Frequency", "mean"),
            MedianFrequency=("Frequency", "median"),
            AvgMonetary=("Monetary", "mean"),
            MedianMonetary=("Monetary", "median"),
        )
    )

    # --------------------------------------------------------
    # Sales Profile
    # --------------------------------------------------------

    sales_profile = (
        matched_sales
        .groupby("Cluster", as_index=False)
        .agg(
            TotalOrders=("OrderID", "nunique"),
            TotalRevenue=("Amount", "sum"),
        )
    )

    profile = rfm_profile.merge(
        sales_profile,
        on="Cluster",
        how="left",
        validate="one_to_one",
    )

    profile[
        [
            "TotalOrders",
            "TotalRevenue",
        ]
    ] = profile[
        [
            "TotalOrders",
            "TotalRevenue",
        ]
    ].fillna(0)

    # --------------------------------------------------------
    # Overall Totals
    # --------------------------------------------------------

    total_customers = int(
        kmeans["CustomerID"].nunique()
    )

    total_orders = int(
        matched_sales["OrderID"].nunique()
    )

    total_revenue = float(
        matched_sales["Amount"].sum()
    )

    overall_aov = safe_ratio(
        total_revenue,
        total_orders,
    )

    # --------------------------------------------------------
    # Shares + AOV
    # --------------------------------------------------------

    profile["CustomerPercentage"] = (
        safe_ratio(
            profile["CustomerCount"],
            total_customers,
        )
        * 100
    )

    profile["OrderShare"] = (
        safe_ratio(
            profile["TotalOrders"],
            total_orders,
        )
        * 100
    )

    profile["RevenueShare"] = (
        safe_ratio(
            profile["TotalRevenue"],
            total_revenue,
        )
        * 100
    )

    profile["AvgOrderValue"] = np.where(
        profile["TotalOrders"] > 0,
        profile["TotalRevenue"] / profile["TotalOrders"],
        np.nan,
    )

    # --------------------------------------------------------
    # Overall Benchmark
    # --------------------------------------------------------

    overall_values = {
        "AvgRecency": float(kmeans["Recency"].mean()),
        "MedianRecency": float(kmeans["Recency"].median()),
        "AvgFrequency": float(kmeans["Frequency"].mean()),
        "MedianFrequency": float(kmeans["Frequency"].median()),
        "AvgMonetary": float(kmeans["Monetary"].mean()),
        "MedianMonetary": float(kmeans["Monetary"].median()),
        "AvgOrderValue": float(overall_aov),
    }

    for metric in (
        "AvgRecency",
        "AvgFrequency",
        "AvgMonetary",
        "AvgOrderValue",
    ):
        profile[f"{metric}VsOverall"] = safe_ratio(
            profile[metric],
            overall_values[metric],
        )

        profile[f"{metric}DiffPct"] = (
            profile[f"{metric}VsOverall"] - 1
        ) * 100

    # --------------------------------------------------------
    # Column Order
    # --------------------------------------------------------

    profile = profile[
        [
            "Cluster",
            "CustomerCount",
            "CustomerPercentage",
            "AvgRecency",
            "MedianRecency",
            "AvgFrequency",
            "MedianFrequency",
            "AvgMonetary",
            "MedianMonetary",
            "TotalOrders",
            "OrderShare",
            "TotalRevenue",
            "RevenueShare",
            "AvgOrderValue",
            "AvgRecencyVsOverall",
            "AvgRecencyDiffPct",
            "AvgFrequencyVsOverall",
            "AvgFrequencyDiffPct",
            "AvgMonetaryVsOverall",
            "AvgMonetaryDiffPct",
            "AvgOrderValueVsOverall",
            "AvgOrderValueDiffPct",
        ]
    ]

    profile = (
        profile
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    overall = pd.DataFrame(
        [
            {
                "Scope": "Overall",
                "CustomerCount": total_customers,
                "CustomerPercentage": 100.0,
                "AvgRecency": overall_values["AvgRecency"],
                "MedianRecency": overall_values["MedianRecency"],
                "AvgFrequency": overall_values["AvgFrequency"],
                "MedianFrequency": overall_values["MedianFrequency"],
                "AvgMonetary": overall_values["AvgMonetary"],
                "MedianMonetary": overall_values["MedianMonetary"],
                "TotalOrders": total_orders,
                "OrderShare": 100.0,
                "TotalRevenue": total_revenue,
                "RevenueShare": 100.0,
                "AvgOrderValue": overall_values["AvgOrderValue"],
            }
        ]
    )

    diagnostics = {
        "SalesRows": int(len(sales)),
        "MatchedSalesRows": int(len(matched_sales)),
        "UnmatchedSalesRows": unmatched_rows,
        "UnmatchedRevenue": round(unmatched_revenue, 2),
    }

    return profile, overall, diagnostics


# ============================================================
# 9. HUMAN-READABLE LEVELS FOR LLM
# ============================================================

def qualitative_level(
    diff_pct: float,
    recency: bool = False,
) -> str:
    if pd.isna(diff_pct):
        return "無法比較"

    # Recency 越低越好，所以方向反轉。
    value = -diff_pct if recency else diff_pct

    if value >= 25:
        return "明顯較活躍" if recency else "明顯偏高"

    if value >= 8:
        return "略為活躍" if recency else "略高"

    if value <= -25:
        return "明顯較不活躍" if recency else "明顯偏低"

    if value <= -8:
        return "略不活躍" if recency else "略低"

    return "接近整體"


def profile_records_for_llm(
    profile: pd.DataFrame,
    overall: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    將 Cluster KPI 轉成 LLM 可直接解讀的結構化資料。

    除了原始 KPI 與相對整體差異之外，
    額外加入排名資訊，避免小型 LLM 只看到絕對數值時
    誤判哪一群才是真正的高價值／核心客群。
    """
    overall_row = overall.iloc[0]
    records: list[dict[str, Any]] = []

    revenue_rank = (
        profile["RevenueShare"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    monetary_rank = (
        profile["AvgMonetary"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    frequency_rank = (
        profile["AvgFrequency"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    # Recency 越低越好，因此 ascending=True。
    activity_rank = (
        profile["AvgRecency"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    aov_rank = (
        profile["AvgOrderValue"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    for idx, row in profile.iterrows():
        records.append(
            {
                "cluster": int(row["Cluster"]),
                "customer_count": int(row["CustomerCount"]),
                "customer_share_pct": round(float(row["CustomerPercentage"]), 2),

                "avg_recency_days": round(float(row["AvgRecency"]), 2),
                "overall_avg_recency_days": round(float(overall_row["AvgRecency"]), 2),
                "recency_diff_pct": round(float(row["AvgRecencyDiffPct"]), 2),
                "activity_level": qualitative_level(
                    float(row["AvgRecencyDiffPct"]),
                    recency=True,
                ),
                "activity_rank": int(activity_rank.loc[idx]),

                "avg_frequency": round(float(row["AvgFrequency"]), 2),
                "overall_avg_frequency": round(float(overall_row["AvgFrequency"]), 2),
                "frequency_diff_pct": round(float(row["AvgFrequencyDiffPct"]), 2),
                "frequency_level": qualitative_level(
                    float(row["AvgFrequencyDiffPct"])
                ),
                "frequency_rank": int(frequency_rank.loc[idx]),

                "avg_monetary": round(float(row["AvgMonetary"]), 2),
                "overall_avg_monetary": round(float(overall_row["AvgMonetary"]), 2),
                "monetary_diff_pct": round(float(row["AvgMonetaryDiffPct"]), 2),
                "monetary_level": qualitative_level(
                    float(row["AvgMonetaryDiffPct"])
                ),
                "monetary_rank": int(monetary_rank.loc[idx]),

                "total_orders": int(row["TotalOrders"]),
                "order_share_pct": round(float(row["OrderShare"]), 2),

                "total_revenue": round(float(row["TotalRevenue"]), 2),
                "revenue_share_pct": round(float(row["RevenueShare"]), 2),
                "revenue_rank": int(revenue_rank.loc[idx]),

                "avg_order_value": round(float(row["AvgOrderValue"]), 2),
                "overall_avg_order_value": round(float(overall_row["AvgOrderValue"]), 2),
                "aov_diff_pct": round(float(row["AvgOrderValueDiffPct"]), 2),
                "aov_level": qualitative_level(
                    float(row["AvgOrderValueDiffPct"])
                ),
                "aov_rank": int(aov_rank.loc[idx]),

                "revenue_to_customer_share_ratio": round(
                    float(row["RevenueShare"]) / float(row["CustomerPercentage"])
                    if float(row["CustomerPercentage"]) != 0
                    else float("nan"),
                    3,
                ),
            }
        )

    return records


# ============================================================
# 10. LLM PROMPT
# ============================================================

def naming_terms(record: dict[str, Any]) -> list[str]:
    """Fact-level vocabulary; LLM chooses the combination, never a fixed cluster name."""
    r, f, m, a = (float(record[k]) for k in
                  ("recency_diff_pct", "frequency_diff_pct", "monetary_diff_pct", "aov_diff_pct"))
    return [
        "高活躍" if r <= -25 else "近期活躍" if r < -8 else "低活躍" if r > 8 else "一般活躍",
        "極低頻" if f <= -50 else "低頻" if f < -8 else "高頻" if f > 8 else "中頻",
        "低消費" if m < -8 else "高價值" if m > 8 else "中等消費",
        "低客單" if a < -8 else "高客單" if a > 8 else "中等客單",
    ]


def build_prompt(
    record: dict[str, Any], used_names: list[str], retry_feedback: str = "",
    all_records: list[dict[str, Any]] | None = None,
) -> str:
    metrics = [
        ("avg_recency_days", "recency_diff_pct", "最近消費距今天數", "低活躍", "近期活躍"),
        ("avg_frequency", "frequency_diff_pct", "購買頻率", "高頻", "低頻"),
        ("avg_monetary", "monetary_diff_pct", "平均累計消費金額", "高價值", "低消費"),
        ("avg_order_value", "aov_diff_pct", "平均客單價", "高客單", "低客單"),
    ]
    facts, basis, vocabulary = [], [], naming_terms(record)
    for metric, diff_key, label, higher_word, lower_word in metrics:
        diff = float(record[diff_key])
        direction = "higher" if diff > 8 else "lower" if diff < -8 else "similar"
        word = higher_word if direction == "higher" else lower_word if direction == "lower" else "接近整體"
        if metric == "avg_recency_days" and diff <= -25:
            word = "高活躍"
        if metric == "avg_frequency" and diff <= -50:
            word = "極低頻"
        facts.append(f"{label}：{word}")
        basis.append({"metric": metric, "comparison": "overall", "direction": direction})
    table = [{"cluster": r["cluster"], "R": r["avg_recency_days"], "F": r["avg_frequency"], "M": r["avg_monetary"]}
             for r in (all_records or [record])]
    return f"""只替 cluster {record['cluster']} 命名。你是繁體中文客群分析師。
Python 已確認本群事實，不得反轉：
{chr(10).join(facts)}
命名可用語意：{'、'.join(vocabulary)}。只挑選兩個最有區別的指標特徵組成自然名稱，不要把全部詞塞進名稱，以客群結尾，4 至 10 個中文字。
若所有指標接近整體，可以描述一般現況，不必強行湊兩個高低特徵。禁止用高價、低消等不完整縮寫，請用高價值、低消費。
不能使用與上述事實相反的字眼。低消費不能寫高價值。低活躍不能寫近期活躍。
不得使用消費率、活躍率、成長、流失、未來預測等未提供指標。
群間比較（R 越小越近期，F/M 越大越高）：{json.dumps(table, ensure_ascii=False)}
本群與相近群的區別：{cluster_distinction_hint(record)}
不得與這些名稱重複或只增減群、一般、型等字：{json.dumps(used_names, ensure_ascii=False)}
修正要求：{retry_feedback or '無'}
只回傳 JSON，包含以下欄位：
cluster：{record['cluster']}
cluster_name：依事實自行命名，兩個指標詞直接連接，再加客群。禁止群組、分群、編號、標點或多餘連接字。
naming_basis：從以下正確依據選取 1 至 3 個原樣複製，不要修改方向：{json.dumps(basis, ensure_ascii=False)}
此次只回答 cluster {record['cluster']}。
"""


# ============================================================
# 11. OLLAMA
# ============================================================

def ollama_generate(
    prompt: str,
    model: str,
    base_url: str,
    timeout: int,
    record: dict[str, Any] | None = None,
) -> str:
    endpoint = base_url.rstrip("/") + "/api/generate"

    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["cluster", "cluster_name", "naming_basis"],
        "properties": {
            "cluster": {"type": "integer", **({"enum": [int(record["cluster"])]} if record else {})},
            "cluster_name": {"type": "string", "minLength": 4, "maxLength": 10,
                             "pattern": ("^(?:" + "|".join(naming_terms(record)) + "){2}客群$") if record else "^[一-鿿]{2,8}客群$"},
            "naming_basis": {"type": "array", "minItems": 1, "maxItems": 3,
                             "items": {"type": "object", "required": ["metric", "comparison", "direction"],
                                "properties": {
                                    "metric": {"enum": ["avg_recency_days", "avg_frequency", "avg_monetary", "avg_order_value"]},
                                    "comparison": {"enum": ["overall"]},
                                    "direction": {"enum": ["higher", "lower", "similar"]},
                                }, "additionalProperties": False}},
        },
    }
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.35,
                "num_ctx": 8192,
                "num_predict": 768,
                "top_p": 0.9,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=timeout,
    ) as response:
        body = json.loads(
            response.read().decode("utf-8")
        )

    result = body.get("response")

    if not isinstance(result, str) or not result.strip():
        raise ValueError(
            "Ollama 回應缺少有效 response 文字。"
        )

    return result.strip()


def parse_json_response(
    text: str,
) -> dict[str, Any]:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                "LLM 回傳內容不含可解析的 JSON 物件。"
            )

        parsed = json.loads(
            cleaned[start:end + 1]
        )

    if not isinstance(parsed, dict):
        raise ValueError(
            "LLM JSON 最外層必須是物件。"
        )

    return parsed


# ============================================================
# 12. VALIDATE LLM OUTPUT
# ============================================================

def chinese_count(text: str) -> int:
    return len(
        re.findall(
            r"[\u4e00-\u9fff]",
            text,
        )
    )


def factual_narrative(row: pd.Series) -> tuple[list[str], str]:
    """Render current KPI facts, not cluster-specific labels or future predictions."""
    r, f, m, a = (float(row[k]) for k in
        ("AvgRecencyDiffPct", "AvgFrequencyDiffPct", "AvgMonetaryDiffPct", "AvgOrderValueDiffPct"))
    activity = "近期活躍度高於整體" if r < -8 else "近期活躍度低於整體" if r > 8 else "近期活躍度接近整體"
    def relative(label, diff):
        return label + ("高於整體" if diff > 8 else "低於整體" if diff < -8 else "接近整體")
    facts = [activity, relative("購買頻率", f), relative("平均累計消費金額", m), relative("平均客單價", a)]
    return facts, "此客群" + "，".join(facts) + "。"


def validate_single_llm_output(
    payload: dict[str, Any],
    expected_cluster: int,
    profile: pd.DataFrame,
    used_names: list[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON 最外層必須是物件。")

    try:
        cluster = int(payload.get("cluster"))
    except (TypeError, ValueError) as exc:
        raise ValueError("cluster 必須是整數。") from exc

    if cluster != expected_cluster:
        raise ValueError(
            f"LLM Cluster 編號不一致：預期 {expected_cluster}，實際 {cluster}"
        )

    name = str(payload.get("cluster_name", "")).strip()
    # 顯示完整指標名稱，避免「客單」縮寫讓使用者誤解。
    name = re.sub(r"客單(?!價)", "客單價", name)
    matching = profile.loc[profile["Cluster"] == cluster]
    if len(matching) != 1:
        raise ValueError(f"Cluster {cluster} 必須在 KPI 表中唯一存在")
    characteristics, description = factual_narrative(matching.iloc[0])
    # Legacy replies can still be checked by offline tests; production schema
    # only requests the name and evidence, and never uses model prose.
    characteristics = payload.get("key_characteristics", characteristics)
    description = str(payload.get("business_description", description)).strip()

    forbidden = re.compile(r"cluster|群組|分群|第\s*\d|\d", re.IGNORECASE)

    if not 4 <= chinese_count(name) <= 10:
        raise ValueError(f"Cluster {cluster} 名稱中文字數不符規則：{name!r}")
    if forbidden.search(name):
        raise ValueError(f"Cluster {cluster} 名稱包含禁止文字或數字：{name!r}")

    unsupported_future_terms = re.compile(
        r"成長型|成長潛力|潛力型|潛力客|流失型|流失風險|"
        r"高流失|低流失|未來高價值|未來價值|預測高價值|待成長"
    )
    if unsupported_future_terms.search(name):
        raise ValueError(
            f"Cluster {cluster} 名稱含有目前 KPI 無法支持的預測性詞彙：{name!r}"
        )

    if re.sub(r"一般|普通|型|消費者|客群|群|者", "", name) in [re.sub(r"一般|普通|型|消費者|客群|群|者", "", n) for n in used_names]:
        raise ValueError(
            f"Cluster {cluster} 名稱與已使用名稱重複：{name!r}；"
            f"已使用名稱={used_names}。請改用能反映此 Cluster 差異的不同名稱。"
        )

    if not isinstance(characteristics, list) or not 3 <= len(characteristics) <= 4:
        raise ValueError(f"Cluster {cluster} 核心特徵必須有 3 至 4 項。")

    characteristics = [str(value).strip() for value in characteristics]
    if any(not value for value in characteristics):
        raise ValueError(f"Cluster {cluster} 核心特徵不得為空。")

    if re.search(r"[0-9%]", description) or not description.endswith("。"):
        raise ValueError("描述需使用完整中文句子並以句號結尾，不要抄寫數字或百分比")
    if not 25 <= chinese_count(description) <= 80:
        raise ValueError(f"Cluster {cluster} 商業描述長度不符規則。")

    profile_lookup = profile.set_index("Cluster")
    if cluster not in profile_lookup.index:
        raise ValueError(f"LLM 回傳不存在的 Cluster：{cluster}")

    row = profile_lookup.loc[cluster]
    recency_diff = float(row["AvgRecencyDiffPct"])
    frequency_diff = float(row["AvgFrequencyDiffPct"])
    monetary_diff = float(row["AvgMonetaryDiffPct"])
    revenue_share = float(row["RevenueShare"])
    customer_share = float(row["CustomerPercentage"])

    top_revenue_cluster = int(
        profile.loc[profile["RevenueShare"].idxmax(), "Cluster"]
    )

    if cluster != top_revenue_cluster and re.search(r"核心|頂級|最高價值", name):
        raise ValueError(
            f"Cluster {cluster} 不是營收貢獻最高群，名稱不得使用核心/頂級/最高價值：{name!r}"
        )

    if recency_diff <= -8 and re.search(r"低活躍|不活躍|沉睡|待喚回", name):
        raise ValueError(
            f"Cluster {cluster} 的 Recency 顯示較活躍，名稱卻描述為低活躍/沉睡：{name!r}"
        )

    if recency_diff >= 8 and re.search(r"高活躍|(?<!低)(?<!不)活躍型|近期活躍", name):
        raise ValueError(
            f"Cluster {cluster} 的 Recency 顯示較不活躍，名稱卻描述為活躍：{name!r}"
        )

    if frequency_diff <= -8 and re.search(r"高頻|高頻率|頻繁", name):
        raise ValueError(
            f"Cluster {cluster} 的 Frequency 低於整體，名稱卻描述為高頻：{name!r}"
        )

    if frequency_diff >= 8 and re.search(r"低頻|低頻率", name):
        raise ValueError(
            f"Cluster {cluster} 的 Frequency 高於整體，名稱卻描述為低頻：{name!r}"
        )

    revenue_efficiency = revenue_share / customer_share if customer_share != 0 else 0.0

    if (
        monetary_diff <= -8
        and revenue_efficiency < 1.25
        and re.search(r"高價值|高貢獻", name)
    ):
        raise ValueError(
            f"Cluster {cluster} 的 Monetary 與營收效率不足以支持高價值/高貢獻命名：{name!r}"
        )

    if monetary_diff >= 8 and re.search(r"低價值|低消費", name):
        raise ValueError(
            f"Cluster {cluster} 的 Monetary 高於整體，名稱卻描述為低價值/低消費：{name!r}"
        )

    basis = payload.get("naming_basis")
    metric_diffs = {
        "avg_recency_days": recency_diff, "avg_frequency": frequency_diff,
        "avg_monetary": monetary_diff, "avg_order_value": float(row["AvgOrderValueDiffPct"]),
    }
    if not isinstance(basis, list) or not 1 <= len(basis) <= 3:
        raise ValueError("naming_basis 必須包含 1 至 3 個指標依據")
    for evidence in basis:
        if not isinstance(evidence, dict) or evidence.get("metric") not in metric_diffs or evidence.get("comparison") != "overall":
            raise ValueError("命名依據必須使用允許的 metric，comparison 必須是 overall")
        diff = metric_diffs[evidence["metric"]]
        direction = "higher" if diff > 8 else "lower" if diff < -8 else "similar"
        if evidence.get("direction") != direction:
            raise ValueError(f"{evidence['metric']} 相對整體差異 {diff:.2f}%，direction 應為 {direction}")
    if re.search(r"高價(?!值)|低消(?!費)", name):
        raise ValueError("請只選兩個指標，使用完整詞高價值或低消費，不要縮寫")
    if not name.endswith("客群"):
        raise ValueError("名稱必須以客群結尾，並表達指標特徵")
    dimensions = [r"活躍|近期|沉睡|久未", r"高頻|低頻|頻率", r"價值|高消費|低消費|消費金額", r"客單"]
    named_dimensions = sum(bool(re.search(pattern, name)) for pattern in dimensions)
    differing_dimensions = sum(abs(v) > 8 for v in (recency_diff, frequency_diff, monetary_diff))
    if differing_dimensions >= 2 and named_dimensions < 2:
        raise ValueError(f"名稱 {name!r} 太籠統，需選兩個指標詞，例如活躍度及購買頻率，且方向符合本群事實")
    if re.search(r"消費率|活躍率|成長率", "、".join([name, *characteristics, description])):
        raise ValueError("輸入沒有消費率、活躍率或成長率，請使用購買頻率、消費金額或近期活躍度")

    if len(set(re.findall(r"低活躍|高活躍|近期活躍|極低頻|低頻|高頻|低消費|高價值|低客單|高客單", name))) < 2 and differing_dimensions >= 2:
        raise ValueError("名稱必須使用兩個不同指標特徵，不可重複同一詞")
    # Check explicit directional claims in all prose, not only the name.
    prose = "、".join([name, *characteristics, description])
    aov_diff = float(row["AvgOrderValueDiffPct"])
    checks = [
        (aov_diff < -8, r"高客單|客單價偏高|客單價較高", "客單價低於整體"),
        (aov_diff > 8, r"低客單|客單價偏低|客單價較低", "客單價高於整體"),
        (recency_diff < -8, r"低活躍|不活躍|沉睡|待喚回|活躍度偏低|活躍度較低|距今較久", "Recency 較低，消費較近期"),
        (recency_diff > 8, r"高活躍|(?<!低)(?<!不)活躍型|活躍度偏高|活躍度較高|距今較短", "Recency 較高，消費較不近期"),
        (frequency_diff < -8, r"高頻|頻繁|頻率偏高|頻率較高", "購買頻率低於整體"),
        (frequency_diff > 8, r"低頻|頻率偏低|頻率較低", "購買頻率高於整體"),
        (monetary_diff < -8, r"高價值|高消費|消費金額偏高|消費金額較高", "平均消費低於整體"),
        (monetary_diff > 8, r"低價值|低消費|消費金額偏低|消費金額較低", "平均消費高於整體"),
    ]
    for applies, pattern, fact in checks:
        if applies and re.search(pattern, prose):
            raise ValueError(f"名稱、特徵或描述與 KPI 矛盾：{fact}。請使用明確的相對整體敘述")
    if len(set(characteristics)) != len(characteristics):
        raise ValueError("核心特徵不可重複")

    predictive_text = " ".join([name, *characteristics, description])
    unsupported_future_terms_full = re.compile(
        r"成長型|成長潛力|潛力型|潛力客|流失型|流失風險|"
        r"高流失|低流失|未來高價值|未來價值|預測高價值|"
        r"未來成長|有望成長|具成長|成長空間"
    )
    if unsupported_future_terms_full.search(predictive_text):
        raise ValueError(
            f"Cluster {cluster} 的 LLM 解讀含有目前描述性 KPI 無法支持的未來/預測性敘述。"
        )

    return {
        "Cluster": cluster,
        "ClusterName": name,
        "KeyCharacteristics": "、".join(characteristics),
        "BusinessDescription": description,
        "DescriptionSource": "Ollama LLM" if "business_description" in payload else "Python KPI",
    }


def add_cluster_comparisons(records: list[dict[str, Any]]) -> None:
    """Compare arbitrary clusters on standardized log RFM; ignore constant axes.

    A gap below 0.5 between-cluster standard deviations or below 8%
    of the larger original value is considered weak.
    This is a descriptive heuristic, not a statistical significance test.
    """
    metrics = ["avg_recency_days", "avg_frequency", "avg_monetary"]
    values = np.array([[r[m] for m in metrics] for r in records], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("群間比較需要有限且非負的 RFM")
    transformed = np.log1p(values)
    scale = transformed.std(axis=0)
    z = (transformed - transformed.mean(axis=0)) / np.where(scale > 1e-12, scale, 1)
    for i, record in enumerate(records):
        candidates = [j for j in range(len(records)) if j != i]
        if not candidates:
            record["nearest_cluster"] = None
            record["distinctions"] = []
            continue
        distances = {j: float(np.linalg.norm(z[i] - z[j])) for j in candidates}
        closest = min(distances.values())
        # Keep ties; cluster IDs must not determine the comparison.
        nearest = [j for j in candidates if np.isclose(distances[j], closest)]
        record["nearest_cluster"] = [records[j]["cluster"] for j in nearest]
        differences = []
        for j in nearest:
            for k, metric in enumerate(metrics):
                gap = float(z[i, k] - z[j, k])
                differences.append({
                    "metric": metric, "comparison_cluster": records[j]["cluster"],
                    "value": record[metric], "comparison_value": records[j][metric],
                    "direction": "higher" if gap > 1e-9 else "lower" if gap < -1e-9 else "similar",
                    "standardized_gap": round(abs(gap), 4),
                    "distinctive": bool(abs(gap) >= 0.5 and abs(values[i, k] - values[j, k]) / max(abs(values[i, k]), abs(values[j, k]), 1e-12) >= 0.08),
                })
        record["distinctions"] = sorted(differences, key=lambda d: -d["standardized_gap"])


def cluster_distinction_hint(record: dict[str, Any]) -> str:
    differences = record.get("distinctions", [])
    strong = [d for d in differences if d["distinctive"]][:3]
    if not strong:
        return "與最相近群的 RFM 差異不明顯，不可強行宣稱獨特、明顯或極端。依相對整體現況命名。"
    labels = {"avg_recency_days": "消費距今天數", "avg_frequency": "購買頻率", "avg_monetary": "平均消費"}
    return "；".join(f"相對群 {d['comparison_cluster']}，本群{labels[d['metric']]}{'較高' if d['direction'] == 'higher' else '較低'}（{d['value']} 對 {d['comparison_value']}）" for d in strong)



# ============================================================
# 13. LLM NAMING
# ============================================================

def name_clusters(
    profile: pd.DataFrame,
    overall: pd.DataFrame,
    model: str,
    base_url: str,
    timeout: int,
    retries: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    逐一對每個 Cluster 呼叫 Ollama。

    V3 一次要求 llama3.2:3b 回傳全部 Cluster，
    小模型可能少回一筆，造成 clusters 筆數不一致。
    V4 改為一次只命名一個 Cluster，降低格式遺漏。
    """
    records = profile_records_for_llm(profile, overall)
    add_cluster_comparisons(records)

    rows: list[dict[str, Any]] = []
    used_names: list[str] = []
    errors: list[str] = []
    llm_outputs: list[dict[str, Any]] = []
    attempts_by_cluster: dict[str, int] = {}

    for record in records:
        cluster = int(record["cluster"])
        success = False

        retry_feedback = ""

        for attempt in range(1, retries + 1):
            attempts_by_cluster[str(cluster)] = attempt

            try:
                prompt = build_prompt(
                    record,
                    used_names,
                    retry_feedback=retry_feedback,
                    all_records=records,
                )
                response_text = ollama_generate(
                    prompt,
                    model,
                    base_url,
                    timeout,
                    record=record,
                )
                parsed_response = parse_json_response(response_text)

                validated = validate_single_llm_output(
                    parsed_response,
                    cluster,
                    profile,
                    used_names,
                )

                rows.append(validated)
                used_names.append(validated["ClusterName"])
                llm_outputs.append(parsed_response)

                print(
                    f"Cluster {cluster} naming success "
                    f"(Attempt {attempt}): {validated['ClusterName']}"
                )

                success = True
                break

            except (
                HTTPError,
                URLError,
                TimeoutError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                message = (
                    f"Cluster {cluster} Attempt {attempt}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)
                print(f"LLM naming warning: {message}")

                # 將本次驗證失敗原因回饋給下一次 Prompt。
                # V4 每次重試使用完全相同的 Prompt，3B 模型很容易
                # 連續產生同一個重複名稱；V5 會要求下一次針對錯誤修正。
                retry_feedback = f"第 {attempt} 次失敗：{exc}。請改選兩個特徵組合，不要重複前次答案。"

                if attempt < retries:
                    time.sleep(1)

        if not success:
            cluster_errors = [
                error
                for error in errors
                if error.startswith(f"Cluster {cluster} ")
            ]
            raise ValueError(
                "Ollama Cluster 命名失敗，已達最大重試次數。\n"
                "本程式不使用規則式名稱替代 LLM 結果。\n"
                + "\n".join(cluster_errors)
            )

    naming = (
        pd.DataFrame(rows)
        .sort_values("Cluster")
        .reset_index(drop=True)
    )

    return naming, {
        "NamingSource": "Ollama LLM",
        "NamingStatus": "SUCCESS",
        "Model": model,
        "AttemptsByCluster": attempts_by_cluster,
        "Errors": errors,
        "LLMInput": records,
        "LLMOutput": llm_outputs,
    }


# ============================================================
# 14. OUTPUT FORMATTING
# ============================================================

def round_output(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = (
        result[numeric_columns]
        .round(4)
    )

    return result


# ============================================================
# 15. ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "建立 Cluster KPI Profile，並使用 Ollama 進行客群命名。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="專案根目錄；未指定時預設為 Cluster_Profile.py 上一層。",
    )

    parser.add_argument(
        "--kmeans",
        type=Path,
        help="K_Means.csv 完整路徑。",
    )

    parser.add_argument(
        "--fact-sales",
        type=Path,
        help="Fact_Sales.csv 完整路徑。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="輸出資料夾；預設為 K-means分群/Cluster。",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama 模型，預設：{DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama API URL。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="每次 Ollama 請求逾時秒數。",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="LLM 生成與驗證最多嘗試次數。",
    )

    return parser.parse_args()


# ============================================================
# 16. MAIN
# ============================================================

def main() -> int:
    args = parse_args()

    (
        project_root,
        dw_dir,
        kmeans_path,
        fact_sales_path,
        output_dir,
    ) = resolve_paths(args)

    banner(PROGRAM_TITLE)

    print(f"Project Root : {project_root}")
    print(f"DW Input     : {dw_dir}")
    print(f"K-Means Input: {kmeans_path}")
    print(f"Sales Input  : {fact_sales_path}")
    print(f"Output Dir   : {output_dir}")
    print(f"Ollama Model : {args.model}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    section("1. LOADING AND VALIDATING INPUT")

    raw_segments = segment.assignments(project_root, named=False)
    kmeans = prepare_kmeans(raw_segments.loc[raw_segments.SegmentOrigin.eq('Training')])

    sales = segment.baseline_profile_sales(project_root)

    print(
        f"K-Means customers : {len(kmeans):,}"
    )

    print(
        f"Cluster count     : {kmeans['Cluster'].nunique():,}"
    )

    print(
        f"Fact_Sales rows   : {len(sales):,}"
    )

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    section("2. CALCULATING CLUSTER KPI")

    profile, overall, diagnostics = calculate_profiles(
        kmeans,
        sales,
    )

    print(
        profile[
            [
                "Cluster",
                "CustomerCount",
                "CustomerPercentage",
                "RevenueShare",
                "AvgOrderValue",
            ]
        ].to_string(index=False)
    )

    if diagnostics["UnmatchedSalesRows"] > 0:
        print(
            "Warning: "
            f"{diagnostics['UnmatchedSalesRows']:,} 筆銷售資料未對應任何分群，"
            f"金額 = {diagnostics['UnmatchedRevenue']:,.2f}。"
        )

    # 先儲存純數值結果。
    # 即使 LLM 後續失敗，KPI 仍保留供檢查。

    profile_kpi_path = (
        output_dir
        / "Cluster_Profile_KPI.csv"
    )

    overall_path = (
        output_dir
        / "Cluster_Overall_KPI.csv"
    )

    round_output(profile).to_csv(
        profile_kpi_path,
        index=False,
        encoding="utf-8-sig",
    )

    round_output(overall).to_csv(
        overall_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(f"Saved KPI     : {profile_kpi_path}")
    print(f"Saved Overall : {overall_path}")

    # --------------------------------------------------------
    # LLM Naming
    # --------------------------------------------------------

    section("3. OLLAMA CLUSTER NAMING")

    naming, metadata = name_clusters(
        profile=profile,
        overall=overall,
        model=args.model,
        base_url=args.ollama_url,
        timeout=args.timeout,
        retries=max(1, args.retries),
    )

    print(
        f"Naming source : {metadata['NamingSource']}"
    )

    print(
        f"Naming status : {metadata['NamingStatus']}"
    )

    print(
        naming[
            [
                "Cluster",
                "ClusterName",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Final Profile
    # --------------------------------------------------------

    section("4. BUILDING FINAL PROFILE")

    final = profile.merge(
        naming,
        on="Cluster",
        how="left",
        validate="one_to_one",
    )

    final.insert(
        1,
        "ClusterName",
        final.pop("ClusterName"),
    )

    final.insert(
        2,
        "KeyCharacteristics",
        final.pop("KeyCharacteristics"),
    )

    final.insert(
        3,
        "BusinessDescription",
        final.pop("BusinessDescription"),
    )

    final.insert(
        4,
        "NamingSource",
        metadata["NamingSource"],
    )

    final_path = (
        output_dir
        / "Cluster_Profile_Final.csv"
    )

    json_path = (
        output_dir
        / "Cluster_Profile_LLM.json"
    )

    round_output(final).to_csv(
        final_path,
        index=False,
        encoding="utf-8-sig",
    )

    json_payload = {
        "metadata": {
            "naming_source": metadata["NamingSource"],
            "naming_status": metadata["NamingStatus"],
            "description_source": "Python KPI",
            "model": metadata["Model"],
            "attempts_by_cluster": metadata["AttemptsByCluster"],
            "errors": metadata["Errors"],
            "diagnostics": diagnostics,
        },
        "llm_input": metadata["LLMInput"],
        "llm_output": metadata["LLMOutput"],
        "validated_naming": naming.to_dict(
            orient="records"
        ),
    }

    json_path.write_text(
        json.dumps(
            json_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Output Summary
    # --------------------------------------------------------

    section("5. OUTPUT")

    print(
        f"Cluster KPI    : {profile_kpi_path}"
    )

    print(
        f"Overall KPI    : {overall_path}"
    )

    print(
        f"LLM Audit JSON : {json_path}"
    )

    print(
        f"Final Profile  : {final_path}"
    )

    print()
    print("Cluster Profile completed successfully.")

    if final.ClusterName.duplicated().any():
        raise ValueError('LLM客群名稱重複，請重新命名；不讓診斷混併客群。')
    ctx = segment.context(project_root, require_names=False)
    ctx['NamesHash'] = segment.digest(final_path)
    (output_dir / 'Naming_Manifest.json').write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding='utf-8')
    print('客群中心只使用季度合格訓練樣本；名稱只根據訓練客戶輪廓。')
    return 0


# ============================================================
# 17. RUN
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

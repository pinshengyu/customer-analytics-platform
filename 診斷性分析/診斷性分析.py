from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import warnings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment_contract as segment



PROJECT_ROOT = Path(__file__).resolve().parent.parent
ETL_DIR = PROJECT_ROOT / "ETL_Output"
AARRR_DIR = PROJECT_ROOT / "描述性分析" / "AARRR"
OUTPUT_DIR = Path(__file__).resolve().parent

# 百分比單位：-10 表示 -10%，採嚴格小於。
REVENUE_DROP_THRESHOLD = -10
ORDERS_DROP_THRESHOLD = -10
BUYERS_DROP_THRESHOLD = -10
AOV_DROP_THRESHOLD = -10
KPI_THRESHOLDS = {
    "Revenue": REVENUE_DROP_THRESHOLD, "Orders": ORDERS_DROP_THRESHOLD,
    "Buyers": BUYERS_DROP_THRESHOLD, "AOV": AOV_DROP_THRESHOLD,
}
EPSILON = 1e-9
# 請確認 Amount 是商品明細金額；若是重複於各列的訂單總額，必須先修正 ETL。
AMOUNT_GRAIN = "line"
# None 保留原程式所有日期；只診斷歷史資料時可改成 "2024-12-31"。
ANALYSIS_END_DATE = None
AUDIT_ROWS: list[dict] = []


def audit(check: str, count: int, detail: str) -> None:
    AUDIT_ROWS.append({"Check": check, "Count": count, "Detail": detail})


def require_columns(data: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要欄位：{missing}")


def unique_dimension(data: pd.DataFrame, key: str, source: str) -> pd.DataFrame:
    if data[key].isna().any():
        raise ValueError(f"{source}.{key} 存在空值")
    data = data.drop_duplicates().copy()
    if data[key].duplicated().any():
        raise ValueError(f"{source}.{key} 有衝突或多筆資料，無法安全進行 many-to-one merge")
    return data


def safe_pct(change: float, previous: float) -> float:
    if pd.isna(previous) or pd.isna(change) or previous <= EPSILON:
        return np.nan
    return float(change / previous * 100)


def contribution(change: float, total_change: float) -> float:
    if pd.isna(change) or pd.isna(total_change) or abs(total_change) <= EPSILON:
        return np.nan
    return float(change / total_change * 100)



def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False)


def write_output(folder: str, filename: str, data: pd.DataFrame) -> None:
    output_path = OUTPUT_DIR / folder / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")


def prepare_orders() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    orders = read_csv(ETL_DIR / "Order_Fact.csv")
    products = read_csv(ETL_DIR / "Product_Dim.csv")
    customers = read_csv(AARRR_DIR / "AARRR_Customer_Base.csv")
    campaigns = read_csv(ETL_DIR / "Marketing_Campaign_Table.csv")

    require_columns(orders, ["OrderID", "CustomerID", "ProductID", "OrderDate",
                            "Quantity", "GrossAmount", "DiscountAmount", "Amount", "CampaignID"], "Order_Fact")
    require_columns(products, ["ProductID", "Category", "Brand", "Cost"], "Product_Dim")
    require_columns(customers, ["CustomerID", "ClusterName", "AcquisitionChannel",
                               "IsActivated", "IsRepeatPurchaser", "IsReferrer"], "AARRR_Customer_Base")
    require_columns(campaigns, ["CampaignID", "Budget"], "Marketing_Campaign_Table")
    if AMOUNT_GRAIN != "line":
        raise ValueError("本程式沿用 Amount 為明細金額的既有定義，請先確認 ETL 金額粒度")
    if orders[["OrderID", "CustomerID", "ProductID"]].isna().any().any():
        raise ValueError("交易的 OrderID / CustomerID / ProductID 不可空白")
    audit("ExactDuplicateOrderRows", int(orders.duplicated().sum()),
          "無明細主鍵時不自行刪除；完全相同列可能是合法明細，也可能是重複匯入")
    if orders.duplicated().any():
        warnings.warn("發現完全相同交易列；未自行刪除，請確認明細主鍵與 ETL 去重結果")
    orders["OrderDate"] = pd.to_datetime(orders["OrderDate"], errors="coerce")
    audit("InvalidOrderDate", int(orders["OrderDate"].isna().sum()), "排除無法轉換的日期")
    orders = orders.loc[orders["OrderDate"].notna()].copy()
    if ANALYSIS_END_DATE is not None:
        orders = orders.loc[orders["OrderDate"] < pd.Timestamp(ANALYSIS_END_DATE) + pd.Timedelta(days=1)].copy()
    if orders.empty:
        raise ValueError("有效分析期間沒有交易資料")
    for column in ["Quantity", "GrossAmount", "DiscountAmount", "Amount"]:
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
        if (~np.isfinite(orders[column])).any():
            raise ValueError(f"Order_Fact.{column} 存在無效金額／數量，不能以 0 代替")
    orders["CampaignID"] = pd.to_numeric(orders["CampaignID"], errors="coerce").fillna(0)
    if ((orders["CampaignID"] % 1 != 0) | (orders["CampaignID"] < 0)).any():
        raise ValueError("Order_Fact.CampaignID 必須是非負整數")
    orders["CampaignID"] = orders["CampaignID"].astype("Int64")
    products = unique_dimension(products[["ProductID", "Category", "Brand", "Cost"]], "ProductID", "Product_Dim")
    customers = unique_dimension(customers, "CustomerID", "AARRR_Customer_Base")
    customers_for_orders = customers[["CustomerID", "ClusterName"]]
    campaigns = campaigns.copy()
    campaigns["CampaignID"] = pd.to_numeric(campaigns["CampaignID"], errors="coerce")
    if campaigns["CampaignID"].isna().any() or (campaigns["CampaignID"] % 1 != 0).any():
        raise ValueError("Marketing_Campaign_Table.CampaignID 必須是整數")
    campaigns["CampaignID"] = campaigns["CampaignID"].astype("Int64")
    campaigns = unique_dimension(campaigns, "CampaignID", "Marketing_Campaign_Table")
    before_rows, before_amount = len(orders), orders["Amount"].sum()
    # 衝突欄位不能默默形成 _x / _y；商品與客群標籤沿用原來源。
    overlap = set(orders.columns) & {"Category", "Brand", "Cost", "ClusterName"}
    if overlap:
        raise ValueError(f"交易表與維度欄位重疊：{sorted(overlap)}，請確認來源定義")
    orders = orders.merge(products, on="ProductID", how="left", validate="many_to_one")
    orders = orders.merge(customers_for_orders, on="CustomerID", how="left", validate="many_to_one")
    if len(orders) != before_rows or not np.isclose(orders["Amount"].sum(), before_amount):
        raise ValueError("merge 改變交易列數或總金額")
    orders["Cost"] = pd.to_numeric(orders["Cost"], errors="coerce")
    orders.loc[~np.isfinite(orders["Cost"]), "Cost"] = np.nan
    audit("MissingCostRows", int(orders["Cost"].isna().sum()), "成本未知時毛利留空，避免視為零成本")
    audit("MissingClusterRows", int(orders["ClusterName"].isna().sum()), "以未分類保留，確保營收可對帳")
    orders["EstimatedGrossProfit"] = orders["Amount"] - orders["Cost"] * orders["Quantity"]
    orders["Quarter"] = orders["OrderDate"].dt.to_period("Q").astype(str)
    if "Channel" not in orders:
        warnings.warn("交易表缺少 Channel，通路診斷僅能顯示未知通路")
        orders["Channel"] = "未知通路"
        audit("MissingChannelColumn", 1, "沒有虛構通路或以獲客來源代替交易通路")
    for column, default in [("Channel", "未知通路"), ("Category", "未分類"),
                            ("Brand", "未分類"), ("ClusterName", "未分類")]:
        orders[column] = orders[column].fillna(default).astype(str)
    # 每張訂單必須只屬於一位客戶及一季；商品、通路、活動可有多個。
    consistency = orders.groupby("OrderID")[["CustomerID", "Quarter"]].nunique()
    if consistency.gt(1).any().any():
        raise ValueError("同一 OrderID 跨客戶或跨季度，無法正確計算訂單數")
    multi = orders.loc[orders["OrderID"].duplicated(keep=False)]
    repeated = multi.groupby("OrderID")["Amount"].nunique().eq(1).sum()
    audit("SameAmountMultiLineOrders", int(repeated),
          "僅為檢查提示；相同明細金額可能合法，CSV 無法證明是否重複訂單總額")
    audit("AmountGrain", len(orders), "沿用明細 Amount 假設；未以 OrderID 去重金額")
    audit("MergeRowCount", len(orders), "兩次 many-to-one merge 列數與金額均不變")
    return orders, {"products": products, "customers": customers}, campaigns


def build_revenue(orders: pd.DataFrame) -> pd.DataFrame:
    summary = orders.groupby("Quarter", as_index=False).agg(
        Revenue=("Amount", "sum"),
        Orders=("OrderID", "nunique"),
        Buyers=("CustomerID", "nunique"),
        Units=("Quantity", "sum"),
        EstimatedGrossProfit=("EstimatedGrossProfit", lambda x: x.sum() if x.notna().all() else np.nan),
    )
    # 不把缺少交易的季度直接跳過；交易 CSV 本身無法區分零交易與資料漏載。
    quarters = pd.period_range(orders["Quarter"].min(), orders["Quarter"].max(), freq="Q").astype(str)
    summary = summary.set_index("Quarter").reindex(quarters).rename_axis("Quarter").reset_index()
    summary["DataAvailable"] = summary["Orders"].notna()
    summary["AOV"] = summary["Revenue"] / summary["Orders"].replace(0, np.nan)
    for metric in ["Revenue", "Orders", "Buyers", "AOV"]:
        previous = summary[metric].shift(1)
        summary[f"{metric}GrowthPct"] = (
            summary[metric] / previous.where(previous.ne(0)) - 1
        ) * 100
    summary["RevenueGrowth"] = summary["RevenueGrowthPct"]
    return summary.sort_values("Quarter").reset_index(drop=True)


def build_customer_trends(orders: pd.DataFrame) -> pd.DataFrame:
    return orders.groupby(["Quarter", "ClusterName"], as_index=False).agg(
        Revenue=("Amount", "sum"),
        Orders=("OrderID", "nunique"),
        Buyers=("CustomerID", "nunique"),
        EstimatedGrossProfit=("EstimatedGrossProfit", lambda x: x.sum() if x.notna().all() else np.nan),
    )


def build_product_summaries(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries = {}
    for dimension in ["Category", "Brand", "Channel"]:
        summary = orders.groupby(dimension, as_index=False).agg(
            Revenue=("Amount", "sum"),
            Units=("Quantity", "sum"),
            Orders=("OrderID", "nunique"),
            GrossAmount=("GrossAmount", "sum"),
            DiscountAmount=("DiscountAmount", "sum"),
            EstimatedGrossProfit=("EstimatedGrossProfit", lambda x: x.sum() if x.notna().all() else np.nan),
        )
        summary["DiscountRate"] = (
            summary["DiscountAmount"]
            / summary["GrossAmount"].replace(0, pd.NA)
            * 100
        )
        summaries[dimension] = summary.sort_values("Revenue", ascending=False)
    return summaries


def build_campaign_attribution(
    orders: pd.DataFrame, campaigns: pd.DataFrame
) -> pd.DataFrame:
    campaign_orders = orders.loc[orders["CampaignID"].gt(0)].copy()
    campaign_orders["CampaignID"] = campaign_orders["CampaignID"].astype("Int64")
    campaigns = campaigns.copy()
    campaigns["CampaignID"] = pd.to_numeric(
        campaigns["CampaignID"], errors="coerce"
    ).astype("Int64")
    activity = campaign_orders.groupby("CampaignID", as_index=False).agg(
        AttributedRevenue=("Amount", "sum"),
        AttributedOrders=("OrderID", "nunique"),
        AttributedDiscount=("DiscountAmount", "sum"),
        EstimatedGrossProfit=("EstimatedGrossProfit", lambda x: x.sum() if x.notna().all() else np.nan),
    ).merge(campaigns, on="CampaignID", how="left", validate="many_to_one")
    activity["Budget"] = pd.to_numeric(activity["Budget"], errors="coerce")
    activity["ProfitAfterBudgetEstimate"] = (
        activity["EstimatedGrossProfit"] - activity["Budget"]
    )
    return activity.sort_values("AttributedRevenue", ascending=False)


def build_acquisition_funnel(customers: pd.DataFrame) -> pd.DataFrame:
    customers = unique_dimension(customers, "CustomerID", "AARRR_Customer_Base")
    for column in ["IsActivated", "IsRepeatPurchaser", "IsReferrer"]:
        customers[column] = pd.to_numeric(customers[column], errors="coerce").fillna(0)
        if not customers[column].isin([0, 1]).all():
            raise ValueError(f"AARRR_Customer_Base.{column} 必須是 0/1 指標")
    funnel = customers.groupby("AcquisitionChannel", as_index=False).agg(
        Customers=("CustomerID", "nunique"),
        Activated=("IsActivated", "sum"),
        RepeatPurchasers=("IsRepeatPurchaser", "sum"),
        Referrers=("IsReferrer", "sum"),
    )
    funnel["ActivationRate"] = funnel["Activated"] / funnel["Customers"] * 100
    funnel["RepeatRate"] = (
        funnel["RepeatPurchasers"] / funnel["Activated"].replace(0, pd.NA) * 100
    )
    funnel["ReferralRate"] = (
        funnel["Referrers"] / funnel["Activated"].replace(0, pd.NA) * 100
    )
    return funnel


def build_quarterly_dimension(orders: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """實際去重 KPI + 可加總的訂單/買家分攤值；補齊消失與新出現的組合。"""
    keys = ["Quarter", *dimensions]
    table = orders.groupby(keys, as_index=False).agg(
        Revenue=("Amount", "sum"), Orders=("OrderID", "nunique"),
        Buyers=("CustomerID", "nunique"),
        EstimatedGrossProfit=("EstimatedGrossProfit", lambda x: x.sum() if x.notna().all() else np.nan),
    )
    for entity, output in [("OrderID", "AllocatedOrders"), ("CustomerID", "AllocatedBuyers")]:
        weight_column = "_OrderWeight" if entity == "OrderID" else "_BuyerWeight"
        # 依維度順序逐層分攤，使 Cluster→Category→Channel 與完整交叉表一致。
        membership = orders[[*keys, entity]].drop_duplicates().copy()
        membership[output] = 1.0
        if weight_column in orders:
            inherited = orders[["Quarter", entity, weight_column]].drop_duplicates()
            if inherited.duplicated(["Quarter", entity]).any():
                raise ValueError("同一父節點內分攤權重不一致")
            membership = membership.merge(inherited, on=["Quarter", entity], validate="many_to_one")
            membership[output] = membership[weight_column]
        for depth, dimension in enumerate(dimensions):
            parent_keys = ["Quarter", entity, *dimensions[:depth]]
            child_keys = [*parent_keys, dimension]
            counts = orders[child_keys].drop_duplicates().groupby(parent_keys, as_index=False).size()
            counts = counts.rename(columns={"size": "_ChildCount"})
            membership = membership.merge(counts, on=parent_keys, validate="many_to_one")
            membership[output] /= membership["_ChildCount"]
            membership = membership.drop(columns="_ChildCount")
        allocated = membership.groupby(keys, as_index=False)[output].sum()
        table = table.merge(allocated, on=keys, validate="one_to_one")
    quarters = pd.DataFrame({"Quarter": pd.period_range(
        orders["Quarter"].min(), orders["Quarter"].max(), freq="Q").astype(str)})
    combinations = orders[dimensions].drop_duplicates()
    grid = quarters.merge(combinations, how="cross")
    table["_Observed"] = True
    table = grid.merge(table, on=keys, how="left", validate="one_to_one")
    available = set(orders["Quarter"])
    table["DataAvailable"] = table["Quarter"].isin(available)
    absent = table["_Observed"].isna() & table["DataAvailable"]
    numeric = ["Revenue", "Orders", "Buyers", "EstimatedGrossProfit", "AllocatedOrders", "AllocatedBuyers"]
    table.loc[absent, numeric] = 0.0
    table = table.drop(columns="_Observed").sort_values([*dimensions, "Quarter"]).reset_index(drop=True)
    table["AOV"] = table["Revenue"] / table["Orders"].replace(0, np.nan)
    for metric in [*numeric, "AOV"]:
        previous = table.groupby(dimensions, dropna=False)[metric].shift(1)
        table[f"Previous{metric}"] = previous
        table[f"{metric}Change"] = table[metric] - previous
        table[f"{metric}GrowthPct"] = table[f"{metric}Change"] / previous.where(previous.gt(EPSILON)) * 100
    return table


def metric_effects(table: pd.DataFrame, kpi: str, root_previous: pd.Series,
                   root_current: pd.Series) -> pd.Series:
    """所有層級使用同一根節點 AOV 係數，讓下層效果可還原上層。"""
    if kpi == "Revenue":
        return table["RevenueChange"]
    if kpi in ("Orders", "Buyers"):
        return table[f"Allocated{kpi}Change"]
    o0, o1 = root_previous["Orders"], root_current["Orders"]
    if min(o0, o1) <= EPSILON:
        return pd.Series(np.nan, index=table.index)
    # 對稱分解 R1/O1 - R0/O0：營收效果 + 訂單分母效果，完全相等。
    revenue_coefficient = 0.5 * (1.0 / o0 + 1.0 / o1)
    order_coefficient = -0.5 * (root_previous["Revenue"] + root_current["Revenue"]) / (o0 * o1)
    return (table["RevenueChange"] * revenue_coefficient
            + table["AllocatedOrdersChange"] * order_coefficient)


def select_declining_driver(table: pd.DataFrame, kpi: str, previous: pd.Series,
                            current: pd.Series) -> tuple[pd.Series | None, pd.DataFrame]:
    table = table.copy()
    table["AttributedChange"] = metric_effects(table, kpi, previous, current)
    table = table.sort_values("AttributedChange", ascending=True)
    candidates = table.loc[table["AttributedChange"].lt(-EPSILON)]
    return (None if candidates.empty else candidates.iloc[0]), table


FINDING_COLUMNS = [
    "Period", "KPI", "CurrentValue", "PreviousValue", "Change", "ChangePct", "Anomaly",
    "ThresholdPct", "ComparisonStatus", "MainDriver", "MainDriverDimension", "MainDriverChange",
    "MainDriverRawChange", "MainDriverGrowthPct", "ContributionPct", "SecondaryDriver",
    "SecondaryDriverChange", "SecondaryContributionPct", "SecondaryShareOfMainPct",
    "TertiaryDriver", "TertiaryDriverChange", "TertiaryContributionPct", "TertiaryShareOfSecondaryPct",
    "DiagnosticLevel", "AttributionMethod", "DiagnosticChain", "Interpretation",
]
DRIVER_COLUMNS = ["Period", "KPI", "Dimension", "Driver", "CurrentValue", "PreviousValue",
                  "RawChange", "RawGrowthPct", "AttributedChange", "ContributionPct", "Direction", "AttributionMethod"]


def build_diagnostic_findings(orders: pd.DataFrame, summary: pd.DataFrame,
                              dimension_tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只對異常觸發因素診斷；每季每 KPI 保留檢查紀錄，空值不等於無異常。"""
    findings, drivers = [], []
    summary = summary.sort_values("Quarter").reset_index(drop=True)
    for i, current in summary.iterrows():
        previous = summary.iloc[i - 1] if i else None
        period = current["Quarter"]
        for kpi, threshold in KPI_THRESHOLDS.items():
            row = {key: np.nan for key in FINDING_COLUMNS}
            row.update(Period=period, KPI=kpi, CurrentValue=current[kpi], ThresholdPct=threshold,
                       DiagnosticLevel=0, DiagnosticChain="", Interpretation="僅表示資料中的變化來源，不代表因果驗證")
            method = ("RevenueSum" if kpi == "Revenue" else
                      "EqualMembershipAllocation" if kpi in ("Orders", "Buyers") else "SymmetricAOVDecomposition")
            row["AttributionMethod"] = method
            if previous is None:
                row.update(ComparisonStatus="NoPreviousPeriod")
                findings.append(row)
                continue
            row["PreviousValue"] = previous[kpi]
            row["Change"] = current[kpi] - previous[kpi]
            row["ChangePct"] = safe_pct(row["Change"], previous[kpi])
            if not current["DataAvailable"] or not previous["DataAvailable"]:
                row.update(ComparisonStatus="MissingQuarterData")
                findings.append(row)
                continue
            if pd.isna(row["ChangePct"]):
                row.update(ComparisonStatus="NonPositiveBaseline")
                findings.append(row)
                continue
            row.update(ComparisonStatus="Comparable", Anomaly=bool(row["ChangePct"] < threshold))
            if not row["Anomaly"]:
                findings.append(row)
                continue
            # 各維度分別對整體變化歸因；不同維度的貢獻不能相加。
            selected_cluster = None
            for dimension, full_table in dimension_tables.items():
                table = full_table.loc[full_table["Quarter"].eq(period)]
                selected, ranked = select_declining_driver(table, kpi, previous, current)
                for _, factor in ranked.iterrows():
                    effect = factor["AttributedChange"]
                    drivers.append(dict(
                        Period=period, KPI=kpi, Dimension=dimension, Driver=str(factor[dimension]),
                        CurrentValue=factor[kpi], PreviousValue=factor[f"Previous{kpi}"],
                        RawChange=factor[f"{kpi}Change"], RawGrowthPct=factor[f"{kpi}GrowthPct"],
                        AttributedChange=effect, ContributionPct=contribution(effect, row["Change"]),
                        Direction="Decline" if effect < -EPSILON else "Offset" if effect > EPSILON else "Stable",
                        AttributionMethod=method))
                if not np.isclose(ranked["AttributedChange"].sum(), row["Change"], atol=1e-7):
                    raise ValueError(f"{period} {kpi} {dimension} 貢獻無法還原整體變化")
                if dimension == "ClusterName":
                    selected_cluster = selected
            if selected_cluster is None:
                row["ComparisonStatus"] = "NoDecliningCluster"
                findings.append(row)
                continue
            main = selected_cluster
            cluster = main["ClusterName"]
            row.update(MainDriver=str(cluster), MainDriverDimension="ClusterName",
                       MainDriverChange=main["AttributedChange"], MainDriverRawChange=main[f"{kpi}Change"],
                       MainDriverGrowthPct=main[f"{kpi}GrowthPct"],
                       ContributionPct=contribution(main["AttributedChange"], row["Change"]), DiagnosticLevel=1)
            parts = [f"{kpi} {row['ChangePct']:.2f}%", f"ClusterName={cluster}: effect={main['AttributedChange']:.6g}, share={row['ContributionPct']:.2f}%"]
            # 只取本期與上一季，避免較早出現後消失的組合干擾下鑽。
            pair = orders.loc[orders["Quarter"].isin([previous["Quarter"], period]) & orders["ClusterName"].eq(cluster)]
            # 至少保留兩季範圍：父客群整季消失時 build 函式也必須比較至本期。
            secondary_table = build_pair_dimension(pair, ["Category"], previous["Quarter"], period)
            secondary, secondary_ranked = select_declining_driver(secondary_table, kpi, previous, current)
            if not np.isclose(secondary_ranked["AttributedChange"].sum(), main["AttributedChange"], atol=1e-7):
                raise ValueError(f"{period} {kpi} Category 下鑽無法還原客群變化")
            if secondary is not None:
                category = secondary["Category"]
                row.update(SecondaryDriver=f"Category={category}", SecondaryDriverChange=secondary["AttributedChange"],
                           SecondaryContributionPct=contribution(secondary["AttributedChange"], row["Change"]),
                           SecondaryShareOfMainPct=contribution(secondary["AttributedChange"], main["AttributedChange"]), DiagnosticLevel=2)
                parts.append(f"Category={category}: effect={secondary['AttributedChange']:.6g}, raw growth={secondary[f'{kpi}GrowthPct']:.2f}%")
                category_pair = pair.loc[pair["Category"].eq(category)].copy()
                # 繼承該 Category 的分攤份額，避免同一客戶／訂單在下鑽時再算一次。
                for entity, weight_column in [("OrderID", "_OrderWeight"), ("CustomerID", "_BuyerWeight")]:
                    category_counts = pair.groupby(["Quarter", entity])["Category"].nunique()
                    entity_index = pd.MultiIndex.from_frame(category_pair[["Quarter", entity]])
                    category_pair[weight_column] = 1.0 / category_counts.reindex(entity_index).to_numpy()
                tertiary_table = build_pair_dimension(category_pair, ["Channel"], previous["Quarter"], period)
                tertiary, tertiary_ranked = select_declining_driver(tertiary_table, kpi, previous, current)
                if not np.isclose(tertiary_ranked["AttributedChange"].sum(), secondary["AttributedChange"], atol=1e-7):
                    raise ValueError(f"{period} {kpi} Channel 下鑽無法還原 Category 變化")
                if tertiary is not None:
                    row.update(TertiaryDriver=f"Channel={tertiary['Channel']}", TertiaryDriverChange=tertiary["AttributedChange"],
                               TertiaryContributionPct=contribution(tertiary["AttributedChange"], row["Change"]),
                               TertiaryShareOfSecondaryPct=contribution(tertiary["AttributedChange"], secondary["AttributedChange"]), DiagnosticLevel=3)
                    parts.append(f"Channel={tertiary['Channel']}: effect={tertiary['AttributedChange']:.6g}, raw growth={tertiary[f'{kpi}GrowthPct']:.2f}%")
            row["DiagnosticChain"] = " → ".join(parts)
            findings.append(row)
    result = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    result["Anomaly"] = result["Anomaly"].astype("boolean")
    return result, pd.DataFrame(drivers, columns=DRIVER_COLUMNS)


def build_pair_dimension(orders: pd.DataFrame, dimensions: list[str], previous: str, current: str) -> pd.DataFrame:
    """父群在某季完全消失／新出現仍需補零；呼叫者已驗證整體兩季資料存在。"""
    # 用現有彙整函式，另將期末消失或期初新出現的群補成兩季。
    table = build_quarterly_dimension(orders, dimensions)
    combinations = orders[dimensions].drop_duplicates()
    grid = pd.DataFrame({"Quarter": [previous, current]}).merge(combinations, how="cross")
    base_columns = ["Revenue", "Orders", "Buyers", "EstimatedGrossProfit", "AllocatedOrders", "AllocatedBuyers"]
    values = table[["Quarter", *dimensions, *base_columns]].copy()
    values["_Present"] = True
    table = grid.merge(values, on=["Quarter", *dimensions], how="left", validate="one_to_one")
    table.loc[table["_Present"].isna(), base_columns] = 0.0
    table = table.drop(columns="_Present").sort_values([*dimensions, "Quarter"])
    table["AOV"] = table["Revenue"] / table["Orders"].replace(0, np.nan)
    for metric in [*base_columns, "AOV"]:
        table[f"Previous{metric}"] = table.groupby(dimensions)[metric].shift(1)
        table[f"{metric}Change"] = table[metric] - table[f"Previous{metric}"]
        table[f"{metric}GrowthPct"] = table[f"{metric}Change"] / table[f"Previous{metric}"].where(table[f"Previous{metric}"].gt(EPSILON)) * 100
    return table.loc[table["Quarter"].eq(current)].copy()


def run_diagnostics(orders: pd.DataFrame, campaigns: pd.DataFrame) -> None:
    summary = build_revenue(orders)
    dimensions = ["ClusterName", "Category", "Brand", "Channel", "CampaignID"]
    tables = {dimension: build_quarterly_dimension(orders, [dimension]) for dimension in dimensions}
    # CampaignID=0 必須保留，否則無活動營收的變化會漏算。
    for dimension, table in tables.items():
        write_output("", f"Diagnostic_Quarterly_{dimension}.csv", table)
    for dimension in ["Category", "Channel", "Brand", "CampaignID"]:
        table = build_quarterly_dimension(orders, ["ClusterName", dimension])
        suffix = "Campaign" if dimension == "CampaignID" else dimension
        if dimension == "CampaignID" and "CampaignName" in campaigns:
            table = table.merge(campaigns[["CampaignID", "CampaignName"]], on="CampaignID", how="left", validate="many_to_one")
            table.loc[table["CampaignID"].eq(0), "CampaignName"] = "無活動"
            table["CampaignName"] = table["CampaignName"].fillna("未知活動")
        write_output("", f"Diagnostic_Cluster_{suffix}.csv", table)
    # 證據鏈第三層需要聯合條件，不能由兩張邊際彙整表猜出 Category × Channel。
    write_output("", "Diagnostic_Cluster_Category_Channel.csv",
                 build_quarterly_dimension(orders, ["ClusterName", "Category", "Channel"]))
    findings, drivers = build_diagnostic_findings(orders, summary, tables)
    write_output("", "Diagnostic_Findings.csv", findings)
    write_output("", "Diagnostic_Driver_Contributions.csv", drivers)


def main() -> None:
    segment.begin_stage(PROJECT_ROOT, OUTPUT_DIR)
    segment.validate_stage(PROJECT_ROOT, AARRR_DIR)
    AUDIT_ROWS.clear()
    orders, source_data, campaigns = prepare_orders()
    write_output("營收驅動", "Diagnostic_Quarterly_Summary.csv", build_revenue(orders))
    write_output(
        "客群趨勢",
        "Diagnostic_Cluster_Quarterly.csv",
        build_customer_trends(orders),
    )
    for dimension, summary in build_product_summaries(orders).items():
        write_output(
            "商品與折扣",
            f"Diagnostic_Product_{dimension}.csv",
            summary,
        )
    write_output(
        "活動歸因",
        "Diagnostic_Campaign_Attribution.csv",
        build_campaign_attribution(orders, campaigns),
    )
    write_output(
        "來源漏斗",
        "Diagnostic_Acquisition_Funnel.csv",
        build_acquisition_funnel(source_data["customers"]),
    )
    run_diagnostics(orders, campaigns)
    write_output("", "Diagnostic_Data_Quality.csv", pd.DataFrame(AUDIT_ROWS))
    segment.finish_stage(PROJECT_ROOT, OUTPUT_DIR)
    print("診斷性分析資料已輸出至：", OUTPUT_DIR)


if __name__ == "__main__":
    main()

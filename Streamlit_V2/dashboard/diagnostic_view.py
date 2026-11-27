from __future__ import annotations

import pandas as pd
import streamlit as st

from .common import csv_download, money, number, percent
from .chart_helpers import rank_bar, line, detail
from .data_loader import DIAGNOSTIC_FILES, diagnostic_file_status, get_project_root, load_diagnostic

KPI_NAMES = {"Revenue": "營收", "Orders": "訂單數", "Buyers": "購買客戶數", "AOV": "平均訂單金額"}
DIMENSIONS = {"ClusterName": "客群", "Category": "商品類別", "Brand": "品牌", "Channel": "交易通路", "CampaignID": "活動"}


def optional_table(key: str) -> pd.DataFrame | None:
    try:
        return load_diagnostic(key)
    except FileNotFoundError:
        st.info(f"尚未產生 {DIAGNOSTIC_FILES[key]}。請先執行專案中的 診斷性分析/診斷性分析.py。")
    except Exception as exc:
        st.error(f"讀取 {DIAGNOSTIC_FILES[key]} 失敗：{exc}")
    return None


def require_fields(data: pd.DataFrame, columns: list[str]) -> bool:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        st.error(f"CSV 缺少欄位：{', '.join(missing)}。請重新執行本次提供的診斷程式。")
        return False
    return True


def anomaly_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def format_value(kpi: str, value) -> str:
    return money(value) if kpi in ("Revenue", "AOV") else number(value)


def render_overview(period: str, summary: pd.DataFrame) -> None:
    st.caption('圖表閱讀方式：先看本季 KPI，再看異常變化，最後下鑽變化來源；詳細資料保留於明細區。')
    row = summary.loc[summary["Quarter"].astype(str).eq(period)]
    if row.empty:
        st.info("這一季沒有摘要資料。")
        return
    row = row.iloc[0]
    st.subheader(f"{period} 季度 KPI")
    for column, (kpi, label) in zip(st.columns(4), KPI_NAMES.items()):
        growth = row.get(f"{kpi}GrowthPct")
        column.metric(label, format_value(kpi, row.get(kpi)),
                      None if pd.isna(growth) else percent(growth))
    st.caption("成長率與異常以相鄰日曆季度比較；缺少整季資料時，不跨季比較。")
    findings = optional_table("findings")
    if findings is not None and require_fields(findings, ["Period", "KPI", "Anomaly", "ChangePct", "ThresholdPct", "ComparisonStatus"]):
        selected = findings.loc[findings["Period"].astype(str).eq(period)].copy()
        anomalies = selected.loc[anomaly_mask(selected["Anomaly"])]
        if not anomalies.empty:
            st.warning("本季異常 KPI：" + "、".join(KPI_NAMES.get(x, x) for x in anomalies["KPI"]))
        elif not selected.empty and selected["ComparisonStatus"].eq("Comparable").all():
            st.info("本季沒有超過設定下降門檻的 KPI。")
        else:
            st.info("本季存在無法比較的 KPI；空白異常判斷不代表正常。")
        columns = ["KPI", "CurrentValue", "PreviousValue", "Change", "ChangePct", "ThresholdPct", "Anomaly", "ComparisonStatus"]
        chart = selected.copy(); chart["指標"] = chart.KPI.map(KPI_NAMES)
        rank_bar(chart, "指標", "ChangePct", "本季哪些指標上升／下降？", unit="季增率", percent=True, signed=True)
        detail(selected, "Diagnostic_Findings_Selected_Quarter")
    metric = st.selectbox("季度趨勢指標", list(KPI_NAMES), format_func=KPI_NAMES.get, key="diag_trend_metric")
    line(summary, "Quarter", metric, KPI_NAMES[metric]+"季度趨勢", unit="金額" if metric in ("Revenue", "AOV") else "數量")


def render_causes(period: str, kpi: str) -> None:
    findings = optional_table("findings")
    if findings is None or not require_fields(findings, ["Period", "KPI", "Anomaly", "ComparisonStatus"]):
        return
    selected = findings.loc[findings["Period"].astype(str).eq(period) & findings["KPI"].eq(kpi)]
    if selected.empty:
        st.info("沒有對應的 KPI 檢查紀錄。")
        return
    row = selected.iloc[0]
    if not anomaly_mask(pd.Series([row["Anomaly"]])).iloc[0]:
        if row["ComparisonStatus"] == "Comparable":
            st.info("此 KPI 未超過下降門檻，因此不觸發因素診斷。")
        else:
            st.info(f"無法進行異常診斷：{row['ComparisonStatus']}")
        return
    st.subheader("計算得到的診斷鏈")
    st.write(f"{KPI_NAMES[kpi]}：{percent(row.get('ChangePct'))}")
    for title, driver, effect, share in [
        ("主要客群", "MainDriver", "MainDriverChange", "ContributionPct"),
        ("客群內商品類別", "SecondaryDriver", "SecondaryDriverChange", "SecondaryContributionPct"),
        ("該類別內交易通路", "TertiaryDriver", "TertiaryDriverChange", "TertiaryContributionPct"),
    ]:
        if pd.notna(row.get(driver)):
            columns = st.columns([2, 1, 1])
            columns[0].write(f"**{title}**：{row[driver]}")
            columns[1].write(f"歸因變化：{row.get(effect):,.4f}")
            columns[2].write(f"占整體變化：{percent(row.get(share))}")
    st.caption("上層與下層是同一變化的下鑽，不可把三級貢獻加總。Python 找出變化來源，未經因果驗證。")
    if kpi in ("Orders", "Buyers"):
        st.info("歸因使用分攤後的訂單／買家變化，可含小數；實際 Orders／Buyers 仍為去重整數。")
    elif kpi == "AOV":
        st.info("AOV 歸因使用營收與訂單分母的對稱分解。歸因變化與各群原始 AOV 差值的意義不同。")
    drivers = optional_table("drivers")
    if drivers is not None and require_fields(drivers, ["Period", "KPI", "Dimension", "Driver", "AttributedChange", "ContributionPct"]):
        dimension = st.selectbox("查看變化來源", list(DIMENSIONS), format_func=DIMENSIONS.get, key="diag_driver_dimension")
        table = drivers.loc[drivers["Period"].astype(str).eq(period) & drivers["KPI"].eq(kpi) & drivers["Dimension"].eq(dimension)].copy()
        table = table.sort_values("AttributedChange")
        if not table.empty:
            table["Driver"] = table["Driver"].astype(str)
            rank_bar(table, "Driver", "AttributedChange", "變化來源：負值拉低，正值拉高", signed=True, unit="金額" if kpi in ("Revenue", "AOV") else "分攤數量")
            detail(table, "Diagnostic_Driver_Contributions_Filtered")
        else:
            st.info("此維度沒有診斷因素紀錄。")
        st.caption("整體下降時，正貢獻表示下降來源，負貢獻表示抵銷；存在抵銷時可超過 100%。各維度不可相加。")
    with st.expander("結構化診斷紀錄"):
        st.dataframe(selected, hide_index=True, width="stretch")
        csv_download(selected, "Diagnostic_Findings_Selected.csv", "diag_chain_download")


def render_cross(period: str) -> None:
    options = {
        "客群 × 商品類別": ("cluster_category", "Category"),
        "客群 × 交易通路": ("cluster_channel", "Channel"),
        "客群 × 品牌": ("cluster_brand", "Brand"),
        "客群 × 活動": ("cluster_campaign", "CampaignID"),
        "客群 × 商品類別 × 交易通路": ("cluster_category_channel", "Channel"),
    }
    choice = st.selectbox("交叉分析", list(options), key="diag_cross_dimension")
    key, dimension = options[choice]
    data = optional_table(key)
    if data is None or not require_fields(data, ["Quarter", "ClusterName", dimension, "Revenue", "Orders", "Buyers", "RevenueChange"]):
        return
    names = sorted(data["ClusterName"].dropna().astype(str).unique())
    if not names:
        st.info("沒有客群資料。")
        return
    cluster = st.selectbox("客群名稱", names, key="diag_cross_cluster")
    filtered = data.loc[data["ClusterName"].astype(str).eq(cluster)].copy()
    if key == "cluster_category_channel":
        categories = sorted(filtered["Category"].dropna().astype(str).unique())
        category = st.selectbox("商品類別", categories, key="diag_cross_category")
        filtered = filtered.loc[filtered["Category"].astype(str).eq(category)]
    metric = st.selectbox("圖表指標", ["RevenueChange", "Revenue", "Orders", "Buyers", "EstimatedGrossProfit", "RevenueGrowthPct"], key="diag_cross_metric")
    table = filtered.loc[filtered["Quarter"].astype(str).eq(period)].copy()
    if table.empty:
        st.info("此季度沒有交叉分析資料。")
        return
    label = dimension
    if dimension == "CampaignID":
        table["活動標籤"] = table["CampaignID"].astype(str)
        if "CampaignName" in table:
            table["活動標籤"] += " · " + table["CampaignName"].fillna("未知活動").astype(str)
        label = "活動標籤"
    if metric in table:
        rank_bar(table, label, metric, {"RevenueChange":"營收增減來源", "Revenue":"營收", "Orders":"訂單數", "Buyers":"買家數", "EstimatedGrossProfit":"估計毛利", "RevenueGrowthPct":"營收季增率"}[metric], signed=True, percent=metric=="RevenueGrowthPct", unit="百分比" if metric=="RevenueGrowthPct" else "數量" if metric in ("Orders","Buyers") else "金額")
    detail(table, "Diagnostic_Cross_Filtered")
    st.caption("Revenue、估計毛利可按完整明細分組加總；跨類別／通路／活動的實際 Orders、Buyers 可能重疊，請勿相加。Allocated 欄位為逐層分攤值。")
    st.caption("零基期的成長率留空；缺少整季資料的 KPI 留空。")


def render_campaign(data: pd.DataFrame) -> None:
    if not require_fields(data, ['CampaignID', 'AttributedRevenue']):
        return
    filtered = data.copy()
    if 'Channel' in filtered:
        channels = sorted(filtered.Channel.dropna().astype(str).unique())
        channel = st.selectbox('活動通路', ['全部通路'] + channels, key='diag_campaign_channel')
        if channel != '全部通路':
            filtered = filtered.loc[filtered.Channel.astype(str).eq(channel)].copy()
    scope = st.selectbox('顯示活動數', ['前 15 個', '前 30 個', '全部活動'], key='diag_campaign_limit')
    filtered['AttributedRevenue'] = pd.to_numeric(filtered.AttributedRevenue, errors='coerce')
    filtered = filtered.sort_values(['AttributedRevenue', 'CampaignID'], ascending=[False, True])
    labels = filtered.CampaignID.astype(str)
    if 'CampaignName' in filtered:
        labels = labels + ' · ' + filtered.CampaignName.fillna('未命名活動').astype(str)
    filtered['活動'] = labels
    limit = {'前 15 個': 15, '前 30 個': 30}.get(scope, len(filtered))
    shown = filtered.head(limit)
    st.caption(f'依歸因營收由高至低排列，顯示 {len(shown)}／{len(filtered)} 個活動；每條橫條代表一個活動。完整篩選明細可於下方下載。')
    if shown.empty:
        st.info('目前篩選條件沒有活動資料。')
    else:
        rank_bar(shown, '活動', 'AttributedRevenue', '各活動歸因營收', unit='歸因營收')
    detail(filtered.drop(columns=['活動']), 'Diagnostic_Campaign_Filtered')


def render_legacy(period: str) -> None:
    options = {"客群季度趨勢": "cluster_trends", "商品類別與折扣": "category", "品牌與折扣": "brand",
               "交易通路與折扣": "channel", "活動歸因": "campaign", "獲客來源漏斗": "funnel"}
    choice = st.selectbox("原有診斷資料", list(options), key="diag_legacy_table")
    data = optional_table(options[choice])
    if data is None:
        return
    if "Quarter" in data:
        data = data.loc[data["Quarter"].astype(str).eq(period)].copy()
        st.caption(f"顯示 {period} 資料。")
    else:
        st.caption("這份原有報表是全分析期間彙整，不受上方季度選擇影響。")
    if choice == '活動歸因':
        render_campaign(data)
        return
    if not data.empty:
        x = next((c for c in ["ClusterName", "Category", "Brand", "Channel", "CampaignID", "AcquisitionChannel"] if c in data), None)
        y = next((c for c in ["Revenue", "AttributedRevenue", "ActivationRate"] if c in data), None)
        if x and y:
            rank_bar(data, x, y, choice, percent=y=="ActivationRate", unit="百分比" if y=="ActivationRate" else "金額")
    detail(data, "Diagnostic_Legacy_Filtered")


def render_quality() -> None:
    st.subheader("診斷資料狀態")
    try:
        status = diagnostic_file_status()
    except FileNotFoundError as exc:
        st.info(str(exc))
        return
    st.caption("完整性狀態可在下方展開檢視；檢查紀錄不一定代表資料錯誤。")
    detail(status, "Diagnostic_File_Status")
    data = optional_table("quality")
    if data is not None:
        detail(data, "Diagnostic_Data_Quality")
    st.caption("Amount 必須是商品明細金額。成本缺失會使該群估計毛利留空；完全相同交易列需要依明細主鍵確認。")


def render() -> None:
    st.caption("異常偵測 → 變化來源 → 貢獻度 → 客群交叉下鑽 → 結構化診斷")
    summary = optional_table("summary")
    if summary is None:
        render_quality()
        return
    if summary.empty or not require_fields(summary, ["Quarter", "Revenue", "Orders", "Buyers", "AOV"]):
        st.info("尚無有效季度摘要。")
        return
    quarters = sorted(summary["Quarter"].dropna().astype(str).unique())
    if not quarters:
        st.info("尚無季度資料。")
        return
    period = st.selectbox("分析季度", quarters, index=len(quarters)-1, key="diag_period")
    tabs = st.tabs(["異常總覽", "原因鏈與貢獻度", "客群交叉分析", "原有診斷報表", "資料品質"], key='diagnostic_tabs', on_change='rerun')
    with tabs[0]:
        if tabs[0].open: render_overview(period, summary)
    with tabs[1]:
        if tabs[1].open:
            kpi = st.selectbox("診斷 KPI", list(KPI_NAMES), format_func=KPI_NAMES.get, key="diag_kpi")
            render_causes(period, kpi)
    with tabs[2]:
        if tabs[2].open: render_cross(period)
    with tabs[3]:
        if tabs[3].open: render_legacy(period)
    with tabs[4]:
        if tabs[4].open: render_quality()

from __future__ import annotations
"""季度描述性分析：沿用診斷客戶歷程的四季快照與固定群中心。
RFM 由已驗證的季度快照讀取；R 不加一天，F/M 使用觀察期內最多四季。
群中心、標準化參數、群編號及名稱均沿用正式訓練結果，不重新 fit 或配對名稱。
分群包含觀察窗口內有購買的客戶；營收、訂單與利潤仍僅統計當季交易。
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quarterly_data as quarterly
import segment_contract as segment


def root() -> Path:
    return Path(__file__).resolve().parent.parent


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(root()/"原始資料"/"Data_Warehouse"/name, encoding="utf-8-sig", low_memory=False, dtype={"CustomerID": "string"})


def qstr(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.to_period("Q").astype(str)


def _sales() -> pd.DataFrame:
    s=read("Fact_Sales.csv")[["CustomerKey","OrderID","DateKey","Amount","Profit"]].copy()
    t=read("Dim_Time.csv")[["DateKey","Date"]].drop_duplicates("DateKey")
    c=read("Dim_Customer.csv")[["CustomerKey","CustomerID"]]
    s=s.merge(t,on="DateKey",how="left",validate="many_to_one").merge(c,on="CustomerKey",how="left",validate="many_to_one")
    s["Date"]=pd.to_datetime(s["Date"],errors="coerce")
    for col in ["Amount","Profit"]: s[col]=pd.to_numeric(s[col],errors="coerce")
    if s[["CustomerID","OrderID","Date","Amount","Profit"]].isna().any().any():
        raise ValueError("季度分群所需交易欄位存在缺失或無法解析值。")
    s["Quarter"]=qstr(s["Date"])
    return s


def build_quarterly_clusters(sales: pd.DataFrame|None=None) -> tuple[pd.DataFrame,pd.DataFrame]:
    # 與診斷頁 load_history 使用相同資料與指派函式，不依 Train/Eligible 篩選歷史。
    project=root();segment.context(project)
    columns=['CustomerID','TargetQuarter','ObservationStart','ObservationEnd','Recency','Frequency','Monetary']
    frames=[quarterly.read_samples(project,name)[columns] for name in
            ['Quarterly_Samples','Quarterly_Holdout','Quarterly_Forecast']]
    data=pd.concat(frames,ignore_index=True)
    if data.duplicated(['CustomerID','TargetQuarter']).any():
        raise ValueError('客戶季度快照重複，請重新執行完整流程。')
    data=quarterly.attach_clusters(project,data)
    data['Quarter']=pd.to_datetime(data.ObservationEnd).dt.to_period('Q').astype(str)
    # 保留舊輸出欄位以相容既有資料介面；兩者現在均為正式群編號。
    data['QuarterCluster']=data.Cluster
    data['AlignedCluster']=data.Cluster
    assignments=data[['Quarter','CustomerID','QuarterCluster','AlignedCluster','ClusterName',
                      'Recency','Frequency','Monetary','ObservationStart','ObservationEnd']].copy()
    profiles=assignments.groupby(['Quarter','QuarterCluster','AlignedCluster','ClusterName'],as_index=False).agg(
        CustomerCount=('CustomerID','size'),Recency=('Recency','mean'),
        Frequency=('Frequency','mean'),Monetary=('Monetary','mean'))
    return assignments,profiles


def build_operational_cluster_quarterly(sales: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    x=sales.merge(assignment[["Quarter","CustomerID","QuarterCluster","AlignedCluster","ClusterName"]],on=["Quarter","CustomerID"],how="left",validate="many_to_one")
    if x.ClusterName.isna().any(): raise ValueError("季度營運 KPI 有交易無法對應當季快照的固定群中心分群。")
    out=x.groupby(["Quarter","AlignedCluster","ClusterName"],as_index=False).agg(TotalRevenue=("Amount","sum"),TotalOrders=("OrderID","nunique"),PurchasingCustomers=("CustomerID","nunique"))
    out=out.rename(columns={"AlignedCluster":"Cluster"});out["AOV"]=out.TotalRevenue/out.TotalOrders.replace(0,np.nan)
    out=out.sort_values(["Cluster","Quarter"]);out["RevenueGrowthRate"]=out.groupby("Cluster").TotalRevenue.pct_change().replace([np.inf,-np.inf],np.nan)*100
    return out


def build_aarrr_quarterly(assignment: pd.DataFrame|None=None) -> tuple[pd.DataFrame,pd.DataFrame]:
    customers=read("Dim_Customer.csv")[["CustomerKey","CustomerID","RegisterDate"]].copy();sales=_sales()
    referral=read("Fact_Referral.csv")[["ReferrerCustomerKey","ReferralDate"]].copy();key=customers[["CustomerKey","CustomerID"]]
    customers["RegisterDate"]=pd.to_datetime(customers.RegisterDate,errors="coerce");referral["ReferralDate"]=pd.to_datetime(referral.ReferralDate,errors="coerce")
    referral=referral.merge(key.rename(columns={"CustomerKey":"ReferrerCustomerKey"}),on="ReferrerCustomerKey",how="left",validate="many_to_one")
    orders=(sales.groupby(["CustomerID","OrderID"],as_index=False).Date.min().sort_values(["CustomerID","Date","OrderID"]));orders["OrderRank"]=orders.groupby("CustomerID").cumcount()+1
    first=orders.loc[orders.OrderRank.eq(1)].copy();second=orders.loc[orders.OrderRank.eq(2)].copy()
    def counts(df,date_col,name,keycol):
        x=df.dropna(subset=[date_col]).copy();x["Quarter"]=qstr(x[date_col]);return x.groupby("Quarter")[keycol].nunique().rename(name)
    frames=[counts(customers,"RegisterDate","NewCustomers","CustomerID"),counts(first,"Date","FirstPurchaseCustomers","CustomerID"),counts(second,"Date","RepeatMilestoneCustomers","CustomerID"),counts(referral,"ReferralDate","ReferrerCustomers","CustomerID"),sales.groupby("Quarter").Amount.sum().rename("Revenue")]
    overall=pd.concat(frames,axis=1).fillna(0).reset_index().sort_values("Quarter");overall["RevenueQoQ"]=overall.Revenue.pct_change().replace([np.inf,-np.inf],np.nan)*100
    for c in ["NewCustomers","FirstPurchaseCustomers","RepeatMilestoneCustomers","ReferrerCustomers"]:overall[c]=overall[c].astype(int)
    if assignment is None:return overall,pd.DataFrame()
    # AARRR 事件依當季末的四季快照歸群；當季未購買但窗口內有購買者仍可歸群。
    aq=assignment[["Quarter","CustomerID","AlignedCluster","ClusterName"]].copy()
    revenue=sales.groupby(["Quarter","CustomerID"],as_index=False).Amount.sum().merge(aq,on=["Quarter","CustomerID"],validate="one_to_one")
    def event_table(df,date_col,label):
        x=df[["CustomerID",date_col]].dropna().copy();x["Quarter"]=qstr(x[date_col]);x=x.merge(aq,on=["Quarter","CustomerID"],how="inner",validate="many_to_one")
        return x.groupby(["Quarter","AlignedCluster","ClusterName"])["CustomerID"].nunique().rename(label)
    parts=[event_table(first,"Date","FirstPurchaseCustomers"),event_table(second,"Date","RepeatMilestoneCustomers"),event_table(referral,"ReferralDate","ReferrerCustomers"),revenue.groupby(["Quarter","AlignedCluster","ClusterName"]).agg(Revenue=("Amount","sum"),PurchasingCustomers=("CustomerID","nunique"))]
    cluster=pd.concat(parts,axis=1).fillna(0).reset_index().rename(columns={"AlignedCluster":"Cluster"}).sort_values(["Cluster","Quarter"])
    return overall,cluster


def build_profit_quarterly(sales: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    x=sales.merge(assignment[["Quarter","CustomerID","AlignedCluster","ClusterName"]],on=["Quarter","CustomerID"],how="left",validate="many_to_one")
    if x.ClusterName.isna().any(): raise ValueError("季度利潤有交易無法對應當季快照的固定群中心分群。")
    cq=x.groupby(["Quarter","CustomerID","AlignedCluster","ClusterName"],as_index=False).agg(CustomerQuarterProfit=("Profit","sum"),CustomerQuarterRevenue=("Amount","sum"))
    def summarize(g,scope):
        keys=["Quarter"] if scope=="Overall" else ["Quarter","AlignedCluster","ClusterName"]
        out=g.groupby(keys,as_index=False).agg(QuarterlyProfit=("CustomerQuarterProfit","sum"),QuarterlyRevenue=("CustomerQuarterRevenue","sum"),PurchasingCustomers=("CustomerID","nunique"),AverageProfitPerPurchasingCustomer=("CustomerQuarterProfit","mean"),MedianCustomerQuarterProfit=("CustomerQuarterProfit","median"));out.insert(0,"Scope",scope)
        if scope=="Cluster":
            out=out.rename(columns={"AlignedCluster":"Cluster"}).sort_values(["Cluster","Quarter"]);out["ProfitQoQ"]=out.groupby("Cluster").QuarterlyProfit.pct_change().replace([np.inf,-np.inf],np.nan)*100;out["CumulativeHistoricalProfit"]=out.groupby("Cluster").QuarterlyProfit.cumsum()
        else:
            out=out.sort_values("Quarter");out["ProfitQoQ"]=out.QuarterlyProfit.pct_change().replace([np.inf,-np.inf],np.nan)*100;out["CumulativeHistoricalProfit"]=out.QuarterlyProfit.cumsum()
        return out
    return pd.concat([summarize(cq,"Overall"),summarize(cq,"Cluster")],ignore_index=True,sort=False)


def main(output_root: Path | None = None):
    sales=_sales();assignment,profiles=build_quarterly_clusters(sales)
    op=build_operational_cluster_quarterly(sales,assignment);a,ac=build_aarrr_quarterly(assignment);p=build_profit_quarterly(sales,assignment)
    dd=(Path(output_root) if output_root is not None else root())/"描述性分析";ad=dd/"AARRR";hd=dd/"Historical_CLV";od=dd/"Operational_KPI";qd=dd/"Quarterly_Clustering"
    for d in [ad,hd,od,qd]:d.mkdir(parents=True,exist_ok=True)
    customer_q=(sales.merge(
        assignment[["Quarter","CustomerID","QuarterCluster","AlignedCluster","ClusterName","Recency","Frequency","Monetary"]],
        on=["Quarter","CustomerID"],how="inner",validate="many_to_one"
    ).groupby(
        ["Quarter","CustomerID","QuarterCluster","AlignedCluster","ClusterName","Recency","Frequency","Monetary"],
        as_index=False
    ).agg(
        QuarterRevenue=("Amount","sum"),
        QuarterProfit=("Profit","sum"),
        QuarterOrders=("OrderID","nunique")
    ))
    customer_q.to_csv(qd/"Quarterly_Customer_KPI.csv",index=False,encoding="utf-8-sig")
    assignment.to_csv(qd/"Quarterly_Cluster_Assignments.csv",index=False,encoding="utf-8-sig");profiles.to_csv(qd/"Quarterly_Cluster_Profiles.csv",index=False,encoding="utf-8-sig")
    op.to_csv(od/"Operational_Cluster_Quarterly_Rolling.csv",index=False,encoding="utf-8-sig")
    a.to_csv(ad/"AARRR_Quarterly_Activity.csv",index=False,encoding="utf-8-sig");ac.to_csv(ad/"AARRR_Cluster_Quarterly_Rolling.csv",index=False,encoding="utf-8-sig")
    p.to_csv(hd/"Historical_CLV_Quarterly.csv",index=False,encoding="utf-8-sig")
    model=json.loads((segment.segment_dir(root())/'K_Means_Model.json').read_text(encoding='utf-8'))
    meta={"Rule":"Fixed training centers applied to the same quarterly snapshots as diagnostic customer history; no quarterly refit or name alignment.",
          "RFM":{"Recency":"observation end - last purchase date (no +1)","Frequency":"distinct OrderID in up to four observation quarters","Monetary":"Revenue sum in up to four observation quarters"},
          "WindowQuarters":quarterly.WINDOW_QUARTERS,"K":model['K'],"Context":segment.context(root()),
          "Quarters":sorted(assignment.Quarter.unique().tolist()),
          "MetricScope":"Quarter revenue, orders and profit use only transactions within that quarter."}
    (qd/"Quarterly_Clustering_Definition.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"季度固定群中心分析完成：{assignment.Quarter.nunique()} 季；{len(assignment):,} customer-quarter assignments")

if __name__=="__main__":main()

"""季度客戶快照、標籤與按筆數切分；邊界清除防止標籤時間洩漏。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from quarterly_plan import QuarterPlan, load_plan

WINDOW_QUARTERS = 4
CONTRACT = 'rolling-quarterly-70-15-15-v2'
FEATURES = ['Recency','Frequency','Monetary','AOV','TenureDays',
            'RecentQuarterOrders','RecentQuarterRevenue','PreviousQuarterRevenue','ActiveQuarters']
EXTRA = ['HistoricalProfit','ProfitPerOrder','ProfitMargin','RecentQuarterProfit',
         'PreviousQuarterOrders','PreviousQuarterProfit','OrdersChangeQuarter',
         'RevenueChangeQuarter','ProfitChangeQuarter','AveragePurchaseInterval','SinglePurchase']
ENHANCED = FEATURES + EXTRA

def save(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,encoding='utf-8-sig')

def load_sales(root):
    import segment_contract as segment
    plan=load_plan(root)
    s=segment.raw_sales(root)
    if 'Profit' not in s:raise ValueError('季度價值預測需要實際 Profit。')
    s['Profit']=pd.to_numeric(s.Profit,errors='coerce')
    if not np.isfinite(s.Profit).all():raise ValueError('Profit 必須為有效數值。')
    if (s.Date < pd.Timestamp(plan.start_date)).any() or (s.Date > pd.Timestamp(plan.score_end)).any():
        raise ValueError(f'交易必須在 {plan.history_start}～{plan.holdout} 內，預測 {plan.target}。')
    return s[['CustomerID','OrderID','Date','Revenue','Profit']].copy()

def snapshot(s,target,plan=None):
    plan=plan or QuarterPlan()
    target=pd.Period(target,freq='Q');end=target.start_time-pd.Timedelta(days=1)
    begin=max((target-WINDOW_QUARTERS).start_time,pd.Timestamp(plan.start_date))
    x=s.loc[s.Date.between(begin,end)]
    o=x.groupby(['CustomerID','OrderID'],as_index=False).agg(Date=('Date','min'),Revenue=('Revenue','sum'),Profit=('Profit','sum'))
    z=o.groupby('CustomerID').agg(Last=('Date','max'),First=('Date','min'),Frequency=('OrderID','nunique'),Monetary=('Revenue','sum'),HistoricalProfit=('Profit','sum'))
    z['Recency']=(end-z.Last).dt.days;z['TenureDays']=(end-z.First).dt.days+1
    z['AOV']=z.Monetary/z.Frequency;z['ProfitPerOrder']=z.HistoricalProfit/z.Frequency
    z['ProfitMargin']=z.HistoricalProfit/z.Monetary.replace(0,np.nan)
    for name,q in [('RecentQuarter',target-1),('PreviousQuarter',target-2)]:
        g=o.loc[o.Date.between(q.start_time,q.end_time)].groupby('CustomerID')
        z[name+'Orders']=g.OrderID.nunique();z[name+'Revenue']=g.Revenue.sum();z[name+'Profit']=g.Profit.sum()
    z['ActiveQuarters']=o.assign(Q=o.Date.dt.to_period('Q')).groupby('CustomerID').Q.nunique()
    days=o[['CustomerID','Date']].drop_duplicates().sort_values(['CustomerID','Date'])
    days['Gap']=days.groupby('CustomerID').Date.diff().dt.days
    z['AveragePurchaseInterval']=days.groupby('CustomerID').Gap.mean()
    z['SinglePurchase']=days.groupby('CustomerID').Date.nunique().eq(1).astype(int)
    z=z.fillna(0)
    for field in ['Orders','Revenue','Profit']:
        z[field+'ChangeQuarter']=z['RecentQuarter'+field]-z['PreviousQuarter'+field]
    z=z[ENHANCED].reset_index()
    z['ObservationStart']=str(begin.date());z['ObservationEnd']=str(end.date())
    z['TargetQuarter']=str(target);z['LabelEnd']=str(target.end_time.date())
    z['HistoryQuarterCount']=min(WINDOW_QUARTERS,target.ordinal-pd.Period(plan.history_start).ordinal)
    z['SampleID']=z.CustomerID.astype(str)+'@'+str(target)
    if target<=pd.Period(plan.holdout):
        future=s.loc[s.Date.between(target.start_time,target.end_time)]
        counts=future.groupby('CustomerID').OrderID.nunique()
        z['ActualChurn']=z.CustomerID.map(counts).fillna(0).eq(0).astype(int)
        z['ActualProfit']=z.CustomerID.map(future.groupby('CustomerID').Profit.sum()).fillna(0)
        z['Buyer']=1-z.ActualChurn
    if not np.isfinite(z[ENHANCED].to_numpy(float)).all():raise ValueError('季度特徵含無效數值。')
    return z

def split_samples(data):
    data=data.sort_values(['TargetQuarter','CustomerID'],kind='stable').reset_index(drop=True).copy()
    n=len(data);a=int(n*.7);b=int(n*.85)
    if min(a,b-a,n-b)<1:raise ValueError('樣本不足以切分70/15/15。')
    data['Split']=np.select([np.arange(n)<a,np.arange(n)<b],['Train','Validation'],default='Test')
    data['Eligible']=True;data['ExclusionReason']=''
    validation_asof=data.loc[data.Split.eq('Validation'),'ObservationEnd'].min()
    test_asof=data.loc[data.Split.eq('Test'),'ObservationEnd'].min()
    for split,cutoff in [('Train',validation_asof),('Validation',test_asof)]:
        mask=data.Split.eq(split)&data.LabelEnd.gt(cutoff)
        data.loc[mask,'Eligible']=False
        data.loc[mask,'ExclusionReason']='LabelNotAvailableBeforeNextSplit'
    for split in ['Train','Validation','Test']:
        if not (data.Split.eq(split)&data.Eligible).any():
            raise ValueError(f'{split} 邊界清除後沒有樣本；目前資料不適合這組時間切分。')
    return data

def build(root):
    plan=load_plan(root)
    s=load_sales(root)
    # 第一季沒有前期特徵，僅作暖身；歷史標籤最晚為獨立回測的前一季。
    history=pd.concat([snapshot(s,q,plan) for q in pd.period_range(plan.first_label,plan.history_end,freq='Q')],ignore_index=True)
    history=split_samples(history)
    holdout=snapshot(s,plan.holdout,plan);holdout['Split']='Holdout';holdout['Eligible']=True
    live=snapshot(s,plan.forecast,plan);live['Split']='Forecast';live['Eligible']=True
    out=Path(root)/'K-means分群'/'K_Means';out.mkdir(parents=True,exist_ok=True)
    for df,name in [(history,'Quarterly_Samples'),(holdout,'Quarterly_Holdout'),(live,'Quarterly_Forecast')]:save(df,out/(name+'.csv'))
    counts=history.groupby('Split',sort=False).agg(Assigned=('SampleID','size'),Eligible=('Eligible','sum')).reset_index()
    counts['Excluded']=counts.Assigned-counts.Eligible
    save(counts,out/'Split_Summary.csv')
    info={'Contract':CONTRACT,'HistoryStart':plan.history_start,'HistoryEnd':plan.history_end,'HoldoutQuarter':plan.holdout,
          'ForecastQuarter':plan.forecast,'WindowQuarters':WINDOW_QUARTERS,'WarmupQuarter':plan.history_start,
          'Sort':['TargetQuarter','CustomerID'],'NominalRatios':[.7,.15,.15],
          'BoundaryRule':'Purge Train labels after earliest Validation observation; purge Validation labels after earliest Test observation.',
          'Counts':counts.to_dict('records')}
    (out/'Quarterly_Design.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return s,history,holdout,live

def read_samples(root,name):
    import segment_contract as segment
    folder=segment.segment_dir(root);filename=name+'.csv'
    manifest=json.loads((folder/'Segmentation_Manifest.json').read_text(encoding='utf-8'))
    if manifest.get('Contract')!=CONTRACT or segment.digest(folder/filename)!=manifest['Files'].get(filename):
        raise ValueError('季度樣本版本不一致，請重新執行完整流程。')
    return pd.read_csv(folder/filename,encoding='utf-8-sig',dtype={'CustomerID':'string'},low_memory=False)

def attach_clusters(root,data):
    import segment_contract as segment
    model=json.loads((segment.segment_dir(root)/'K_Means_Model.json').read_text(encoding='utf-8'))
    x=data.copy();x['Cluster']=segment.assign_centers(x,model)
    names=pd.read_csv(Path(root)/'K-means分群'/'Cluster'/'Cluster_Profile_Final.csv',encoding='utf-8-sig')
    x=x.merge(names[['Cluster','ClusterName']],on='Cluster',how='left',validate='many_to_one')
    if x.ClusterName.isna().any():raise ValueError('季度樣本缺少客群名稱。')
    return x

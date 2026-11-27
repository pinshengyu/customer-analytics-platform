"""專案單一分群契約：固定基準期客群與分析輸出版本檢查。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def project_root(start=__file__):
    p=Path(start).resolve()
    for q in [p.parent,*p.parents]:
        if (q/'原始資料'/'Data_Warehouse').is_dir():return q
    raise FileNotFoundError('找不到原始資料/Data_Warehouse；請將完整增補包放在專案根目錄。')

def identifiers(s):
    return s.astype('string').str.strip().str.replace(r'\.0$','',regex=True)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def segment_dir(r):return Path(r)/'K-means分群'/'K_Means'

def manifest(r):
    d=segment_dir(r);m=json.loads((d/'Segmentation_Manifest.json').read_text(encoding='utf-8'))
    from quarterly_data import CONTRACT
    from quarterly_plan import load_plan
    if m.get('Contract')!=CONTRACT:raise ValueError('尚未建立季度70/15/15版分群，請執行完整流程。')
    if m.get('TargetQuarter')!=load_plan(r).target:raise ValueError('分群與目標季度不一致，請執行完整流程。')
    for n in ['K_Means.csv','K_Means_Model.json']:
        if digest(d/n)!=m['Files'][n]:raise ValueError(f'{n} 與分群版本不一致，請重跑K-Means及後續模組。')
    return m

def context(r,require_names=True):
    m=manifest(r);result={'Contract':m['Contract'],'TargetQuarter':m['TargetQuarter'],'SegmentVersion':m['SegmentVersion'],'AssignmentHash':m['Files']['K_Means.csv'],'BaselineStart':m['BaselineStart'],'BaselineEnd':m['BaselineEnd'],'TrainingFeatureEnd':m['TrainingFeatureEnd']}
    if require_names:
        folder=Path(r)/'K-means分群'/'Cluster'
        n=json.loads((folder/'Naming_Manifest.json').read_text(encoding='utf-8'))
        if n['SegmentVersion']!=m['SegmentVersion'] or n['AssignmentHash']!=result['AssignmentHash'] or digest(folder/'Cluster_Profile_Final.csv')!=n['NamesHash']:raise ValueError('客群名稱不是目前分群版本，請重新執行Cluster_Profile.py。')
        result['NamesHash']=n['NamesHash']
    return result

def assignments(r,as_of=None,named=True):
    ctx=context(r,require_names=named);d=segment_dir(r)
    m=pd.read_csv(d/'K_Means.csv',encoding='utf-8-sig',dtype={'CustomerID':'string'})
    if m.CustomerID.isna().any() or m.CustomerID.duplicated().any():raise ValueError('分群CustomerID無效或重複。')
    if as_of is not None:
        m=m.loc[pd.to_datetime(m.AssignedAt).le(pd.Timestamp(as_of))].copy()
    if named:
        n=pd.read_csv(Path(r)/'K-means分群'/'Cluster'/'Cluster_Profile_Final.csv',encoding='utf-8-sig')
        if n.Cluster.duplicated().any() or n.ClusterName.duplicated().any():raise ValueError('客群編號及客群名稱必須唯一，避免診斷依名稱合併不同群。')
        m=m.merge(n[['Cluster','ClusterName']],on='Cluster',how='left',validate='many_to_one')
        if m.ClusterName.isna().any():raise ValueError('客群命名不完整。')
    return m

def attach(r,df,as_of=None):
    a=assignments(r,as_of=as_of)[['CustomerID','Cluster','ClusterName']]
    x=df.drop(columns=['Cluster','ClusterName'],errors='ignore').copy();x.CustomerID=identifiers(x.CustomerID)
    x=x.merge(a,on='CustomerID',how='left',validate='many_to_one')
    if x.Cluster.isna().any():raise ValueError('購買客戶缺少單一基準期分群，請重新執行K-Means及命名。')
    return x

def begin_stage(r,folder):
    c=context(r);p=Path(folder)/'Analysis_Context.json'
    if p.exists():p.unlink()
    return c

def finish_stage(r,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'Analysis_Context.json').write_text(json.dumps(context(r),ensure_ascii=False,indent=2),encoding='utf-8')

def validate_stage(r,folder):
    p=Path(folder)/'Analysis_Context.json'
    if not p.exists():raise ValueError(f'{Path(folder).name} 尚未重跑統一分群版，請先執行對應分析程式。')
    if json.loads(p.read_text(encoding='utf-8'))!=context(r):raise ValueError(f'{Path(folder).name} 使用舊分群或名稱，請重新執行對應分析。')

def ensure_paths(r,kmeans,names=None):
    if Path(kmeans).resolve()!=(segment_dir(r)/'K_Means.csv').resolve():raise ValueError('全專案只能使用原K_Means/K_Means.csv，不能另外指定分支分群。')
    if names is not None and Path(names).resolve()!=(Path(r)/'K-means分群'/'Cluster'/'Cluster_Profile_Final.csv').resolve():raise ValueError('全專案只能使用原Cluster/Cluster_Profile_Final.csv。')

def raw_sales(r):
    d=Path(r)/'原始資料'/'Data_Warehouse'
    s=pd.read_csv(d/'Fact_Sales.csv',encoding='utf-8-sig',low_memory=False)
    c=pd.read_csv(d/'Dim_Customer.csv',encoding='utf-8-sig',low_memory=False)
    t=pd.read_csv(d/'Dim_Time.csv',encoding='utf-8-sig',low_memory=False)
    for df,key in [(c,'CustomerKey'),(c,'CustomerID'),(t,'DateKey')]:
        if df[key].isna().any() or df[key].duplicated().any():raise ValueError(f'{key}必須唯一且非空。')
    date=next((n for n in ['Date','FullDate','CalendarDate'] if n in t),None)
    if date is None:raise ValueError('Dim_Time缺少Date。')
    s=s.drop(columns=['CustomerID','Date'],errors='ignore').merge(c[['CustomerKey','CustomerID']],on='CustomerKey',how='left',validate='many_to_one')
    s=s.merge(t[['DateKey',date]].rename(columns={date:'Date'}),on='DateKey',how='left',validate='many_to_one')
    s['Date']=pd.to_datetime(s.Date,errors='coerce').dt.normalize();s.CustomerID=identifiers(s.CustomerID)
    amount=next((n for n in ['Amount','Revenue','NetAmount'] if n in s),None)
    if amount is None:raise ValueError('Fact_Sales缺少Amount。')
    s['Revenue']=pd.to_numeric(s[amount],errors='coerce')
    if s[['CustomerID','Date','OrderID','Revenue']].isna().any().any() or not np.isfinite(s.Revenue).all():raise ValueError('交易存在無法對應維度的鍵或無效金額。')
    if 'SalesKey' in s and s.SalesKey.duplicated().any():raise ValueError('SalesKey重複。')
    if (s.Revenue<0).any():raise ValueError('原RFM口徑不含負營收，請先確認退款口徑。')
    return s

def baseline_profile_sales(r):
    manifest(r)
    return pd.read_csv(segment_dir(r)/'Naming_Training_Sales.csv',encoding='utf-8-sig',dtype={'CustomerID':'string'})

def assign_centers(rfm,model):
    z=(np.log1p(rfm[['Recency','Frequency','Monetary']].to_numpy(float))-np.array(model['ScalerMean']))/np.array(model['ScalerScale'])
    distance=((z[:,None,:]-np.array(model['Centers'])[None,:,:])**2).sum(axis=2)
    return distance.argmin(axis=1)+1

"""季度訓練資料分群與供既有報表使用的客戶對照。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import segment_contract as segment
import quarterly_data as quarterly

def main():
    parser=argparse.ArgumentParser(description='僅用季度訓練集建立共用分群')
    parser.add_argument('--project-root',type=Path);parser.add_argument('--k',type=int)
    a=parser.parse_args();root=a.project_root.resolve() if a.project_root else segment.project_root(__file__)
    plan=quarterly.load_plan(root)
    sales,history,holdout,live=quarterly.build(root)
    train=history.loc[history.Split.eq('Train')&history.Eligible].copy()
    cols=['Recency','Frequency','Monetary']
    scaler=StandardScaler();z=scaler.fit_transform(np.log1p(train[cols].to_numpy(float)))
    if a.k is not None and a.k<2:raise ValueError('K至少2。')
    scores=[]
    for k in ([a.k] if a.k is not None else range(3,8)):
        if k>=len(z) or k>len(np.unique(z,axis=0)):continue
        fit=KMeans(n_clusters=k,n_init=50,random_state=42).fit(z)
        if len(np.unique(fit.labels_))!=k:continue
        score=silhouette_score(z,fit.labels_,sample_size=min(5000,len(z)),random_state=42)
        scores.append({'K':k,'SilhouetteScore':score,'Inertia':fit.inertia_,'SilhouetteSampleSize':min(5000,len(z))})
        print(f'K={k} Silhouette={score:.6f}',flush=True)
    if not scores:raise ValueError('有效訓練樣本不足以分群。')
    k=int(max(scores,key=lambda row:row['SilhouetteScore'])['K'])
    fit=KMeans(n_clusters=k,n_init=100,random_state=42).fit(z)
    out=segment.segment_dir(root)
    model={'Contract':quarterly.CONTRACT,'TargetQuarter':plan.target,'BaselineStart':plan.start_date,'BaselineEnd':plan.development_end,
           'TrainingFeatureEnd':train.ObservationEnd.max(),'TrainingLabelEnd':train.LabelEnd.max(),
           'TrainingSampleCount':len(train),'TrainingSampleHash':hashlib.sha256('\n'.join(train.SampleID).encode()).hexdigest(),
           'Features':cols,'Transform':'log1p','ScalerMean':scaler.mean_.tolist(),'ScalerScale':scaler.scale_.tolist(),
           'Centers':fit.cluster_centers_.tolist(),'K':k,'RandomState':42,
           'AssignmentRule':'Quarterly forecasts use snapshots; historical dashboards use latest training snapshot or first purchase day.'}
    version=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()[:20]
    train['Cluster']=segment.assign_centers(train,model)
    quarterly.save(train,out/'Clustering_Training_Samples.csv')
    naming=train.sort_values(['ObservationEnd','CustomerID']).drop_duplicates('CustomerID',keep='last').copy()
    quarterly.save(naming,out/'Naming_Training_Customers.csv')
    name_sales=sales.merge(naming[['CustomerID','ObservationStart','ObservationEnd']],on='CustomerID',validate='many_to_one')
    name_sales=name_sales.loc[name_sales.Date.ge(pd.to_datetime(name_sales.ObservationStart))&name_sales.Date.le(pd.to_datetime(name_sales.ObservationEnd))]
    quarterly.save(name_sales[['CustomerID','OrderID','Revenue']].rename(columns={'Revenue':'Amount'}),out/'Naming_Training_Sales.csv')
    canonical=naming[['CustomerID',*cols,'Cluster']].copy();canonical['SegmentOrigin']='Training'
    other=sales.loc[~sales.CustomerID.isin(canonical.CustomerID)]
    if not other.empty:
        first=other.groupby('CustomerID').Date.transform('min');firstday=other.loc[other.Date.eq(first)]
        q=firstday.groupby('CustomerID',as_index=False).agg(Frequency=('OrderID','nunique'),Monetary=('Revenue','sum'))
        q['Recency']=1;q['Cluster']=segment.assign_centers(q,model);q['SegmentOrigin']='FirstPurchaseDay'
        canonical=pd.concat([canonical,q],ignore_index=True)
    canonical['AssignedAt']=canonical.CustomerID.map(sales.groupby('CustomerID').Date.min()).dt.strftime('%Y-%m-%d')
    canonical['SegmentVersion']=version;canonical=canonical.sort_values('CustomerID')
    for name in ['K_Means.csv','Customer_RFM.csv']:quarterly.save(canonical,out/name)
    quarterly.save(pd.DataFrame(scores),out/'K_Silhouette_Score.csv')
    (out/'K_Means_Model.json').write_text(json.dumps(model,ensure_ascii=False,indent=2),encoding='utf-8')
    profile=naming.groupby('Cluster',as_index=False).agg(CustomerCount=('CustomerID','size'),AvgRecency=('Recency','mean'),AvgFrequency=('Frequency','mean'),AvgMonetary=('Monetary','mean'))
    if len(profile)!=k:raise ValueError('命名用訓練客戶快照未涵蓋所有群。')
    profile['CustomerPercentage']=100*profile.CustomerCount/profile.CustomerCount.sum()
    for name in ['Cluster_Profile.csv','Cluster_Summary.csv']:quarterly.save(profile,out/name)
    quarterly.save(canonical.groupby(['Cluster','SegmentOrigin'],as_index=False).size(),out/'Cluster_Assignment_Summary.csv')
    quarterly.save(pd.DataFrame(fit.cluster_centers_,columns=['Recency_Scaled','Frequency_Scaled','Monetary_Scaled']).assign(Cluster=np.arange(1,k+1)),out/'Cluster_Centers.csv')
    files=['K_Means.csv','K_Means_Model.json','Quarterly_Samples.csv','Quarterly_Holdout.csv','Quarterly_Forecast.csv','Clustering_Training_Samples.csv','Naming_Training_Customers.csv','Naming_Training_Sales.csv','Quarterly_Design.json']
    manifest={'Contract':quarterly.CONTRACT,'TargetQuarter':plan.target,'SegmentVersion':version,'BaselineStart':plan.start_date,'BaselineEnd':plan.development_end,
              'TrainingFeatureEnd':model['TrainingFeatureEnd'],'TrainingLabelEnd':model['TrainingLabelEnd'],
              'K':k,'TrainingSampleCount':len(train),'BaselineCustomerCount':len(naming),'AssignedCustomerCount':len(canonical),
              'Files':{name:segment.digest(out/name) for name in files}}
    (out/'Segmentation_Manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'季度訓練集分群完成：K={k}；合格訓練樣本={len(train)}；特徵截止={model["TrainingFeatureEnd"]}',flush=True)

if __name__=='__main__':main()

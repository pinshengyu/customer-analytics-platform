"""季度70/15/15選模、滾動獨立回測與下一季部署預測。"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,
    average_precision_score,log_loss,mean_absolute_error,mean_squared_error,r2_score)
from xgboost import XGBClassifier,XGBRegressor
import quarterly_data as qd
import segment_contract as segment

CONFIGS={
    'original':dict(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.8,colsample_bytree=.8),
    'regularized':dict(n_estimators=150,max_depth=2,learning_rate=.05,min_child_weight=8,reg_lambda=10,reg_alpha=1,subsample=.8,colsample_bytree=.8),
    'balanced':dict(n_estimators=250,max_depth=3,learning_rate=.03,min_child_weight=5,reg_lambda=5,reg_alpha=.2,subsample=.8,colsample_bytree=.8),
}

class Predictor:
    def __init__(self,kind,variant,config):
        self.kind=kind;self.variant=variant;self.config=config
        self.cols=qd.FEATURES if variant=='Original' else qd.ENHANCED
        self.reg=None;self.clf=None;self.value=0.;self.probability=1.
    def fit(self,d):
        if d.empty:raise ValueError('模型訓練樣本不得為空。')
        params=dict(CONFIGS[self.config],random_state=42,n_jobs=2,tree_method='hist')
        if self.kind=='churn':
            if d.Actual.nunique()==2:
                self.clf=XGBClassifier(**params,eval_metric='logloss');self.clf.fit(d[self.cols],d.Actual)
            else:self.probability=float(d.Actual.mean())
            return self
        values=d
        if self.variant=='TwoStage':
            if d.Buyer.nunique()==2:
                self.clf=XGBClassifier(**params,eval_metric='logloss');self.clf.fit(d[self.cols],d.Buyer)
            else:self.probability=float(d.Buyer.mean())
            values=d.loc[d.Buyer.eq(1)]
        if self.variant=='Mean' or len(values)<10 or values.Actual.nunique()<2:
            self.value=float(values.Actual.mean()) if len(values) else 0.
        else:
            self.reg=XGBRegressor(**params,objective='reg:squarederror');self.reg.fit(values[self.cols],values.Actual)
        return self
    def components(self,d):
        if d.empty:return np.array([]),np.array([])
        p=self.clf.predict_proba(d[self.cols])[:,1] if self.clf is not None else np.full(len(d),self.probability)
        v=self.reg.predict(d[self.cols]) if self.reg is not None else np.full(len(d),self.value)
        return p,v
    def predict(self,d):
        p,v=self.components(d)
        return p if self.kind=='churn' else p*v
    def export(self,folder,prefix):
        record={'Kind':self.kind,'Variant':self.variant,'Config':self.config,'Features':self.cols,
                'ConstantValue':self.value,'ConstantProbability':self.probability,'Contract':qd.CONTRACT}
        for stage,model in [('profit',self.reg),('classification',self.clf)]:
            path=folder/(prefix+'_'+stage+'.json')
            if model is not None:model.save_model(path)
            else:path.unlink(missing_ok=True)
        (folder/(prefix+'_spec.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')

def evaluate(kind,y,p):
    if len(y)==0:raise ValueError('評估集合為空。')
    if kind=='clv':
        return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),
                'R2':float(r2_score(y,p)) if len(y)>1 else np.nan}
    hard=(p>=.5).astype(int)
    return {'Accuracy':accuracy_score(y,hard),'Precision':precision_score(y,hard,zero_division=0),
            'Recall':recall_score(y,hard,zero_division=0),'F1':f1_score(y,hard,zero_division=0),
            'ROC_AUC':roc_auc_score(y,p) if y.nunique()==2 else np.nan,
            'PR_AUC':average_precision_score(y,p) if y.nunique()==2 else np.nan,
            'LogLoss':log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1]),'ActualChurnRate':float(y.mean())}

def main(kind):
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path)
    a=parser.parse_args();root=a.project_root.resolve() if a.project_root else segment.project_root(__file__)
    plan=qd.load_plan(root)
    segment.context(root)
    historical=qd.attach_clusters(root,qd.read_samples(root,'Quarterly_Samples'))
    holdout=qd.attach_clusters(root,qd.read_samples(root,'Quarterly_Holdout'))
    live=qd.attach_clusters(root,qd.read_samples(root,'Quarterly_Forecast'))
    target='ActualChurn' if kind=='churn' else 'ActualProfit'
    for d in [historical,holdout]:d['Actual']=d[target]
    train=historical.loc[historical.Split.eq('Train')&historical.Eligible]
    validation=historical.loc[historical.Split.eq('Validation')&historical.Eligible]
    test=historical.loc[historical.Split.eq('Test')&historical.Eligible]
    assert train.LabelEnd.max()<=validation.ObservationEnd.min()
    assert validation.LabelEnd.max()<=test.ObservationEnd.min()
    assert historical.LabelEnd.max()<=holdout.ObservationEnd.min()
    assert holdout.LabelEnd.max()<=live.ObservationEnd.min()
    folder=root/'預測性分析'/'預測結果'/('流失預測' if kind=='churn' else 'CLV價值預測')
    folder.mkdir(parents=True,exist_ok=True);segment.begin_stage(root,folder)
    metrics=[];comparisons=[];selection=[];tests=[];valids=[];backtests=[];backmetrics=[];preds=[];importance=[];chosen=[]
    candidates=([('Original',c) for c in ['regularized','balanced']] if kind=='churn' else
                [('Original','original'),('Mean','original')]+[(v,c) for v in ['Enhanced','TwoStage'] for c in ['regularized','balanced']])
    keys=['SampleID','CustomerID','TargetQuarter','ObservationEnd','LabelEnd','Cluster','ClusterName','Actual','Buyer']
    for cluster in sorted(historical.Cluster.unique()):
        tr=train.loc[train.Cluster.eq(cluster)];va=validation.loc[validation.Cluster.eq(cluster)];te=test.loc[test.Cluster.eq(cluster)]
        if min(len(tr),len(va),len(te))==0:raise ValueError(f'客群 {cluster} 缺少合格訓練／驗證／測試樣本。')
        name=tr.ClusterName.iloc[0];fits=[]
        for variant,config in candidates:
            predictor=Predictor(kind,variant,config).fit(tr)
            score=evaluate(kind,va.Actual,predictor.predict(va))
            loss=score['LogLoss' if kind=='churn' else 'MAE']
            row={'Cluster':int(cluster),'ClusterName':name,'Variant':variant,'Config':config,
                 'TrainCount':len(tr),'ValidationCount':len(va),'ValidationLoss':float(loss),**{'Validation'+k:v for k,v in score.items()}}
            selection.append(row);fits.append((loss,predictor,row))
        _,best,bestrow=min((x for x in fits if x[1].variant!='Mean'),key=lambda x:x[0])
        picked={'Cluster':int(cluster),'Variant':best.variant,'Config':best.config};chosen.append(picked)
        # 所有選擇只看驗證集；在這之後才計算測試與獨立回測成績。
        tp=te[keys].copy()
        for _,model,row in fits:
            prediction=model.predict(te)
            result={'Cluster':int(cluster),'ClusterName':name,'Variant':model.variant,'Config':model.config,
                    'Selected':model is best,'TrainCount':len(tr),'ValidationCount':len(va),'TestCount':len(te),
                    'ValidationLoss':row['ValidationLoss'],**evaluate(kind,te.Actual,prediction)}
            comparisons.append(result)
            tp[model.variant+'_'+model.config+'_Prediction']=prediction
            if model is best:
                result={**result,'ModelStatus':'Constant' if kind=='churn' and best.clf is None else best.variant,
                        'EvaluationSet':'Test','BaselineMAE':float(mean_absolute_error(te.Actual,np.full(len(te),tr.Actual.mean())))}
                metrics.append(result)
        tp['Predicted']=best.predict(te);tests.append(tp)
        val=va[keys].copy();val['Predicted']=best.predict(va);valids.append(val)
        best.export(folder,f'evaluation_cluster_{cluster}')
        for stage,model in [('profit',best.reg),('classification',best.clf)]:
            if model is not None:
                importance.extend({'Cluster':int(cluster),'ClusterName':name,'Stage':stage,'Feature':f,'Importance':float(v)} for f,v in zip(best.cols,model.feature_importances_))
        past=historical.loc[historical.Cluster.eq(cluster)]
        h=holdout.loc[holdout.Cluster.eq(cluster)]
        if h.empty:raise ValueError(f'客群 {cluster} 沒有獨立回測樣本。')
        frozen=Predictor(kind,best.variant,best.config).fit(past)
        hp=h[keys].copy();hp['Predicted']=frozen.predict(h);hp['TrainedThrough']=plan.development_end;backtests.append(hp)
        backmetrics.append({'Cluster':int(cluster),'ClusterName':name,'EvaluationSet':'Holdout','TargetQuarter':plan.holdout,
                            'TrainCount':len(past),'TestCount':len(h),**evaluate(kind,h.Actual,hp.Predicted)})
        frozen.export(folder,f'holdout_cluster_{cluster}')
        deployed=Predictor(kind,best.variant,best.config).fit(pd.concat([past,h],ignore_index=True))
        forecast=live.loc[live.Cluster.eq(cluster)].copy()
        forecast['Predicted']=deployed.predict(forecast);forecast['SelectedVariant']=best.variant;forecast['TrainedThrough']=plan.score_end
        if kind=='churn':forecast['RiskLevel']=np.select([forecast.Predicted>=.7,forecast.Predicted>=.4],['高風險','中風險'],default='低風險')
        else:
            probability,value=deployed.components(forecast)
            forecast['PurchaseProbability']=probability if best.variant=='TwoStage' else np.nan
            forecast['ConditionalProfit']=value if best.variant=='TwoStage' else np.nan
        preds.append(forecast);deployed.export(folder,f'deployment_cluster_{cluster}')
        print(f'{kind} 群{cluster}：Train={len(tr)} Val={len(va)} Test={len(te)}；選定{best.variant}/{best.config}',flush=True)
    pred=pd.concat(preds,ignore_index=True)
    if len(pred)!=len(live) or pred.SampleID.duplicated().any():raise ValueError('預測客戶覆蓋不完整。')
    splits=historical[['SampleID','CustomerID','TargetQuarter','ObservationEnd','LabelEnd','Split','Eligible','ExclusionReason','Cluster']]
    outputs=[(pred,'Customer_Predictions'),(historical,'Training_Features'),(splits,'Train_Test_Split'),
             (pd.concat(tests),'Test_Predictions'),(pd.concat(valids),'Validation_Predictions'),(pd.DataFrame(selection),'Validation_Results'),
             (pd.DataFrame(metrics),'Model_Metrics'),(pd.DataFrame(comparisons),'Model_Comparison'),
             (pd.concat(backtests),'Holdout_Predictions'),(pd.DataFrame(backmetrics),'Holdout_Metrics'),
             (pd.DataFrame(importance,columns=['Cluster','ClusterName','Stage','Feature','Importance']),'Feature_Importance')]
    for df,name in outputs:qd.save(df,folder/(name+'.csv'))
    qd.save(pred.groupby(['Cluster','ClusterName'],as_index=False).agg(CustomerCount=('CustomerID','size'),MeanPrediction=('Predicted','mean'),TotalPrediction=('Predicted','sum')),folder/'Cluster_Summary.csv')
    design=json.loads((segment.segment_dir(root)/'Quarterly_Design.json').read_text(encoding='utf-8'))
    metadata={'Model':'quarterly_'+kind,'Contract':qd.CONTRACT,'ObservationEnd':train.ObservationEnd.max(),
              'LabelEnd':plan.development_end,'ScoreEnd':plan.score_end,'ForecastEnd':plan.forecast_end,'TargetQuarter':plan.forecast,
              'HoldoutQuarter':plan.holdout,'DataThrough':plan.score_end,'WindowQuarters':qd.WINDOW_QUARTERS,
              'Target':'下一季度未購買' if kind=='churn' else '下一季度利潤合計（非終身價值、未折現）',
              'Features':qd.ENHANCED,'SelectedModels':chosen,'SplitDesign':design,'SegmentContext':segment.context(root),
              'ClusterWarning':'分群中心僅以合格Train特徵建立，名稱僅以訓練客戶輪廓產生；預測快照依固定中心逐季指派，描述與診斷使用固定客戶對照。',
              'Population':'特徵截止日前4季曾購買的客戶；初期可用歷史較短，不涵蓋當時尚未購買的新客。',
              'Selection':'樣本按目標季度、CustomerID排序切70/15/15；邊界清除後僅Train訓練、Validation選模、Test評估。',
              'EvaluationCaution':f'70/15/15是名義筆數比例；清除跨界未到期標籤後實際使用數較少。同一客戶可跨季出現。{plan.holdout}另作時間回測，不用於選模。',
              'Deployment':f'{plan.holdout}回測模型重訓至{plan.history_end}；完成回測後固定參數、加入回測季標籤重訓以預測{plan.target}。',
              'ProbabilityCaution':'機率未另校準；季度未購買不等於永久流失。'}
    (folder/'Metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    segment.finish_stage(root,folder)

if __name__=='__main__':raise SystemExit('請使用流失預測.py 或 CLV價值預測.py 執行。')

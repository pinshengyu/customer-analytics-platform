"""單一營收序列：按季度觀測筆數切分，獨立回測與下一季預測分開。"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import quarterly_data as qd
import segment_contract as segment

def predict(values,steps,model):
    if model=='Mean':return np.full(steps,float(np.mean(values)))
    return np.polyval(np.polyfit(np.arange(len(values)),values,1),np.arange(len(values),len(values)+steps)).clip(0)

def comparison(quarters,actual,predicted,split):
    d=pd.DataFrame({'Quarter':quarters,'ActualRevenue':actual,'PredictedRevenue':predicted,'Split':split})
    d['AbsoluteError']=(d.ActualRevenue-d.PredictedRevenue).abs()
    d['AbsolutePercentageError']=d.AbsoluteError/d.ActualRevenue.abs().replace(0,np.nan)
    return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path)
    parser.add_argument('--horizon',type=int,default=1)
    a=parser.parse_args()
    if a.horizon!=1:raise ValueError('本版僅預測設定的下一個完整季度。')
    root=a.project_root.resolve() if a.project_root else segment.project_root(__file__)
    plan=qd.load_plan(root)
    segment.validate_stage(root,root/'描述性分析'/'Operational_KPI')
    h=pd.read_csv(root/'描述性分析'/'Operational_KPI'/'Operational_Quarterly_Trend.csv',encoding='utf-8-sig').rename(columns={'TotalRevenue':'Revenue'})[['Quarter','Revenue']]
    h.Quarter=h.Quarter.astype(str).str.replace(' ','',regex=False);h=h.sort_values('Quarter').reset_index(drop=True)
    expected=list(map(str,pd.period_range(plan.history_start,plan.holdout,freq='Q')))
    if h.Quarter.tolist()!=expected or not np.isfinite(h.Revenue).all() or (h.Revenue<0).any():raise ValueError('需要目標季度前完整12季的非負季度營收。')
    history=h.loc[h.Quarter.le(plan.history_end)].copy();n=len(history);aidx=int(n*.7);bidx=int(n*.85)
    tr=history.iloc[:aidx];va=history.iloc[aidx:bidx];te=history.iloc[bidx:]
    scores=[]
    for model in ['LinearTrend','Mean']:
        p=predict(tr.Revenue.to_numpy(),len(va),model)
        scores.append({'Model':model,'ValidationMAE':float(np.mean(np.abs(va.Revenue.to_numpy()-p))),'TrainCount':len(tr),'ValidationCount':len(va)})
    chosen=min(scores,key=lambda x:x['ValidationMAE'])['Model']
    vp=predict(tr.Revenue.to_numpy(),len(va),chosen)
    tp=predict(tr.Revenue.to_numpy(),n-len(tr),chosen)[len(va):]
    validation=comparison(va.Quarter.to_numpy(),va.Revenue.to_numpy(),vp,'Validation')
    test=comparison(te.Quarter.to_numpy(),te.Revenue.to_numpy(),tp,'Test')
    actual=h.loc[h.Quarter.eq(plan.holdout)]
    bp=predict(history.Revenue.to_numpy(),1,chosen)
    back=comparison(actual.Quarter.to_numpy(),actual.Revenue.to_numpy(),bp,'Holdout')
    back['TrainedThroughQuarter']=plan.history_end
    p=predict(h.Revenue.to_numpy(),1,chosen)
    band=float(test.AbsoluteError.quantile(.9))
    forecast=pd.DataFrame({'Quarter':[plan.forecast],'PredictedRevenue':p,'LowerEstimate':np.maximum(0,p-band),'UpperEstimate':p+band})
    summary=pd.DataFrame([{'Model':chosen,'TrainingStartQuarter':h.Quarter.iloc[0],'TrainingEndQuarter':plan.holdout,
        'TrainingQuarterCount':len(h),'LatestActualRevenue':h.Revenue.iloc[-1],'NextForecastQuarter':plan.forecast,
        'NextQuarterPredictedRevenue':p[0],'NextQuarterChangePct':(p[0]/h.Revenue.iloc[-1]-1)*100 if h.Revenue.iloc[-1] else np.nan,
        'BacktestMAPE':back.AbsolutePercentageError.mean()*100,'BacktestMAE':back.AbsoluteError.mean(),'BacktestCount':len(back),
        'MAPEValidCount':int(back.AbsolutePercentageError.notna().sum()),'ErrorBand90thPercentile':band,
        'ValidationMAE':validation.AbsoluteError.mean(),'TestMAE':test.AbsoluteError.mean(),'TestMAPE':test.AbsolutePercentageError.mean()*100,
        'TrainCount':len(tr),'ValidationCount':len(va),'TestCount':len(te),'HoldoutQuarter':plan.holdout}])
    split=history.copy();split['Split']=['Train']*aidx+['Validation']*(bidx-aidx)+['Test']*(n-bidx)
    out=root/'預測性分析'/'預測結果'/'整體季度營收趨勢預測';out.mkdir(parents=True,exist_ok=True)
    segment.begin_stage(root,out)
    for df,name in [(h,'Revenue_History'),(forecast,'Revenue_Forecast'),(back,'Revenue_Backtest'),(summary,'Forecast_Summary'),
                    (validation,'Revenue_Validation'),(test,'Revenue_Test'),(split,'Revenue_Split'),(pd.DataFrame(scores),'Revenue_Model_Selection')]:
        qd.save(df,out/(name+'.csv'))
    segment.finish_stage(root,out)
    print(f'營收序列：{len(tr)}/{len(va)}/{len(te)}季；{chosen}；獨立回測與下一季預測已分開。',flush=True)

if __name__=='__main__':main()

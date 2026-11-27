"""發布前驗證時間隔離、訓練來源、保存模型和評估指標。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import quarterly_data as qd
import segment_contract as segment
from quarterly_models import Predictor, evaluate
from xgboost import XGBClassifier, XGBRegressor


def validate(root):
    root=Path(root);folder=segment.segment_dir(root)
    plan=qd.load_plan(root)
    manifest=segment.manifest(root)
    for name,digest in manifest['Files'].items():
        assert segment.digest(folder/name)==digest, name+' hash mismatch'
    h=qd.read_samples(root,'Quarterly_Samples')
    rebuilt=qd.split_samples(h)
    pd.testing.assert_frame_equal(h[['SampleID','Split','Eligible']],rebuilt[['SampleID','Split','Eligible']],check_dtype=False)
    assert h.SampleID.is_unique
    tr=h.loc[h.Split.eq('Train')&h.Eligible]
    va=h.loc[h.Split.eq('Validation')&h.Eligible]
    te=h.loc[h.Split.eq('Test')&h.Eligible]
    assert tr.LabelEnd.max()<=va.ObservationEnd.min()
    assert va.LabelEnd.max()<=te.ObservationEnd.min()
    model=json.loads((folder/'K_Means_Model.json').read_text(encoding='utf-8'))
    assert model['TrainingSampleHash']==hashlib.sha256('\n'.join(tr.SampleID).encode()).hexdigest()
    assert model['TrainingSampleCount']==len(tr)
    z=np.log1p(tr[model['Features']].to_numpy(float))
    np.testing.assert_allclose(z.mean(axis=0),model['ScalerMean'])
    training=pd.read_csv(folder/'Clustering_Training_Samples.csv')
    assert set(training.SampleID)==set(tr.SampleID)
    naming=pd.read_csv(folder/'Naming_Training_Customers.csv',dtype={'CustomerID':'string'})
    assert set(naming.SampleID).issubset(set(tr.SampleID)) and naming.CustomerID.is_unique
    hold=qd.read_samples(root,'Quarterly_Holdout');live=qd.read_samples(root,'Quarterly_Forecast')
    assert hold.TargetQuarter.eq(plan.holdout).all() and live.TargetQuarter.eq(plan.forecast).all()
    assert h.LabelEnd.max()<=hold.ObservationEnd.min()
    assert hold.LabelEnd.max()<=live.ObservationEnd.min()
    assert not any(c.startswith('Actual') for c in live.columns)
    sales=qd.load_sales(root)
    # 從原始交易重新計算各季度快照與標籤，檢查匯出檔未混用年度定義。
    for data in [h,hold,live]:
        for quarter,group in data.groupby('TargetQuarter'):
            expected=qd.snapshot(sales,quarter,plan).sort_values('SampleID').reset_index(drop=True)
            actual=group.sort_values('SampleID').reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=1e-8)
    for kind,dirname in [('churn','流失預測'),('clv','CLV價值預測')]:
        out=root/'預測性分析'/'預測結果'/dirname
        segment.validate_stage(root,out)
        meta=json.loads((out/'Metadata.json').read_text(encoding='utf-8'))
        assert meta['Contract']==qd.CONTRACT and meta['TargetQuarter']==plan.forecast
        selection=pd.read_csv(out/'Validation_Results.csv')
        for cluster,rows in selection.groupby('Cluster'):
            best=rows.loc[rows.Variant.ne('Mean')].sort_values('ValidationLoss',kind='stable').iloc[0]
            picked=next(x for x in meta['SelectedModels'] if x['Cluster']==cluster)
            assert picked['Variant']==best.Variant and picked['Config']==best.Config
        for filename,source,prefix in [('Test_Predictions',te,'evaluation'),('Holdout_Predictions',hold,'holdout'),('Customer_Predictions',live,'deployment')]:
            data=pd.read_csv(out/(filename+'.csv'),dtype={'CustomerID':'string'})
            assert data.SampleID.is_unique and set(data.SampleID)==set(source.SampleID)
            assert np.isfinite(data.Predicted).all()
            if kind=='churn':assert data.Predicted.between(0,1).all()
            if prefix=='holdout':assert data.TrainedThrough.eq(plan.development_end).all()
            if prefix=='deployment':assert data.TrainedThrough.eq(plan.score_end).all()
            features=qd.attach_clusters(root,source)
            for cluster,rows in data.groupby('Cluster'):
                spec=json.loads((out/f'{prefix}_cluster_{cluster}_spec.json').read_text(encoding='utf-8'))
                predictor=Predictor(kind,spec['Variant'],spec['Config'])
                predictor.value=spec['ConstantValue'];predictor.probability=spec['ConstantProbability']
                for stage,attribute,cls in [('profit','reg',XGBRegressor),('classification','clf',XGBClassifier)]:
                    path=out/f'{prefix}_cluster_{cluster}_{stage}.json'
                    if path.exists():
                        loaded=cls();loaded.load_model(path);setattr(predictor,attribute,loaded)
                ordered=features.set_index('SampleID').loc[rows.SampleID]
                assert ordered.Cluster.eq(cluster).all()
                np.testing.assert_allclose(predictor.predict(ordered),rows.Predicted,rtol=1e-6,atol=1e-5)
                if prefix!='deployment':
                    metricfile='Model_Metrics.csv' if prefix=='evaluation' else 'Holdout_Metrics.csv'
                    metric=pd.read_csv(out/metricfile).set_index('Cluster').loc[cluster]
                    target='ActualChurn' if kind=='churn' else 'ActualProfit'
                    np.testing.assert_allclose(rows.Actual,ordered[target],rtol=1e-8)
                    for key,value in evaluate(kind,rows.Actual,rows.Predicted.to_numpy()).items():
                        np.testing.assert_allclose(value,metric[key],rtol=1e-5,atol=1e-5,equal_nan=True)
    revenue=root/'預測性分析'/'預測結果'/'整體季度營收趨勢預測'
    assert pd.read_csv(revenue/'Revenue_Forecast.csv').Quarter.tolist()==[plan.forecast]
    assert pd.read_csv(revenue/'Revenue_Backtest.csv').Quarter.tolist()==[plan.holdout]
    report=json.loads((root/'報告結果'/'報告摘要.json').read_text(encoding='utf-8'))
    assert report['Context']==segment.context(root)
    result={'Contract':qd.CONTRACT,'TargetQuarter':plan.target,'HistoryStart':plan.history_start,'HoldoutQuarter':plan.holdout,'Train':len(tr),'Validation':len(va),'Test':len(te),'Holdout':len(hold),'Forecast':len(live),'Status':'passed'}
    (root/'報告結果'/'季度驗證.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    validate(parser.parse_args().project_root)

import json
import logging
import pandas as pd
import altair as alt
import streamlit as st
from .data_loader import get_project_root,read_csv,segment
from .chart_helpers import show,rank_bar,histogram,detail,brief,BLUE,TEAL,GRAY,RISK,RISK_COLORS,AMOUNT_AXIS

def revenue(base):
    folder=base/'整體季度營收趨勢預測';segment.validate_stage(get_project_root(),folder)
    h=read_csv(str(folder/'Revenue_History.csv')).copy();f=read_csv(str(folder/'Revenue_Forecast.csv')).copy();s=read_csv(str(folder/'Forecast_Summary.csv')).iloc[0]
    h['Quarter']=h.Quarter.astype(str).str.replace(' ','',regex=False);f['Quarter']=f.Quarter.astype(str).str.replace(' ','',regex=False);h=h.sort_values('Quarter');f=f.sort_values('Quarter')
    a,b=st.columns(2);a.metric('下一季預測營收',f'{s.NextQuarterPredictedRevenue:,.0f}');b.metric('預測季增率',f'{s.NextQuarterChangePct:.1f}%')
    brief(f"下一季 {f.iloc[0].Quarter} 預測營收 {s.NextQuarterPredictedRevenue:,.0f}，較上一季變化 {s.NextQuarterChangePct:+.1f}%。")
    order=sorted(set(h.Quarter)|set(f.Quarter));x=alt.X('Quarter:O',sort=order,title=None,axis=alt.Axis(labelAngle=0,labelOverlap=True));y=alt.Y('Value:Q',title='營收',axis=alt.Axis(labelExpr=AMOUNT_AXIS))
    actual=h.rename(columns={'Revenue':'Value'});actual['Series']='實際營收'
    forecast=f.rename(columns={'PredictedRevenue':'Value'});forecast['Series']='預測營收'
    anchor=pd.DataFrame({'Quarter':[h.iloc[-1].Quarter],'Value':[h.iloc[-1].Revenue],'Series':['預測營收']});joined=pd.concat([anchor,forecast],ignore_index=True)
    color=alt.Color('Series:N',scale=alt.Scale(domain=['實際營收','預測營收'],range=[BLUE,TEAL]),title=None)
    area=alt.Chart(f).mark_rule(color=TEAL,opacity=.3,strokeWidth=10).encode(x=x,y=alt.Y('LowerEstimate:Q',title='營收（元）'),y2='UpperEstimate:Q',tooltip=['Quarter:O',alt.Tooltip('LowerEstimate:Q',title='估計下界',format=',.0f'),alt.Tooltip('UpperEstimate:Q',title='估計上界',format=',.0f')])
    aplot=alt.Chart(actual).mark_line(point=True,strokeWidth=3).encode(x=x,y=y,color=color,tooltip=['Quarter:O',alt.Tooltip('Value:Q',title='實際營收',format=',.0f')])
    fplot=alt.Chart(joined).mark_line(point=True,strokeWidth=3,strokeDash=[6,4]).encode(x=x,y=y,color=color,tooltip=['Quarter:O',alt.Tooltip('Value:Q',title='預測／銜接值',format=',.0f')])
    show((area+aplot+fplot).properties(title='實線為歷史，虛線為預測，淡色區為估計範圍',height=420))
    st.caption('估計範圍依內部測試集誤差推估，未校準覆蓋率；不代表 90% 信賴區間。')
    detail(f,'Revenue_Forecast')

def risk_chart(q):
    d=q.RiskLevel.value_counts().reindex(RISK,fill_value=0).rename_axis('風險').reset_index(name='客戶數');d['占比']=d['客戶數']/max(1,len(q))*100
    base=alt.Chart(d).encode(theta='客戶數:Q',color=alt.Color('風險:N',scale=alt.Scale(domain=RISK,range=RISK_COLORS)),tooltip=['風險:N','客戶數:Q',alt.Tooltip('占比:Q',format='.1f')])
    show(base.mark_arc(innerRadius=75).properties(title='高、中、低風險占比',height=340))

def customer(base,kind):
    folder=base/('CLV價值預測' if kind=='下一季客戶價值預測' else kind);segment.validate_stage(get_project_root(),folder);meta=json.loads((folder/'Metadata.json').read_text(encoding='utf-8'))
    p=read_csv(str(folder/'Customer_Predictions.csv'));names=p[['Cluster','ClusterName']].drop_duplicates().sort_values('Cluster');labels=dict(zip(names.Cluster,names.ClusterName))
    selected=st.selectbox('客群',['整體']+names.Cluster.tolist(),format_func=lambda x:'整體' if x=='整體' else labels[x],key=kind);q=p if selected=='整體' else p.loc[p.Cluster.eq(selected)]
    st.caption(f"目標季度 {meta['TargetQuarter']}｜以截至 {meta['ScoreEnd']} 的資料，預測至 {meta['ForecastEnd']}。{meta['Target']}")
    st.caption(meta['Population'])
    a,b,c=st.columns(3);a.metric('評分客戶數',f'{len(q):,}')
    if kind=='流失預測':
        b.metric('平均模型流失機率',f'{q.Predicted.mean():.1%}');c.metric('高風險客戶數',f'{q.Predicted.ge(.7).sum():,}')
        brief(f"高風險客戶占 {q.Predicted.ge(.7).mean():.1%}；可先檢視這些客戶的近期購買情況與維繫需求。")
        left,right=st.columns(2)
        with left:risk_chart(q)
        with right:
            if selected=='整體':
                risk=st.selectbox('分布圖風險等級',RISK,key='churn_distribution_risk')
                members=q.loc[q.RiskLevel.eq(risk)]
                counts=members.groupby('ClusterName').size().reindex(names.ClusterName.tolist(),fill_value=0)
                distribution=counts.rename('CustomerCount').rename_axis('ClusterName').reset_index()
                st.caption(f'{risk}共 {len(members):,} 人；顯示這些客戶在各客群的人數分布。')
                if members.empty:
                    st.info(f'目前沒有{risk}客戶。')
                else:
                    rank_bar(distribution,'ClusterName','CustomerCount',f'{risk}客戶的客群分布',unit='客戶數（人）',cluster=True)
            else:histogram(q.Predicted*100,'模型流失機率分布',unit='機率 (%)')
        st.caption('高風險 ≥70%；中風險 40%～未滿70%；低風險 <40%。模型機率尚未另做校準。')
    else:
        b.metric('下一季度預測利潤合計',f'{q.Predicted.sum():,.0f}');c.metric('平均每人預測利潤',f'{q.Predicted.mean():,.0f}')
        st.caption('固定下一季度利潤預測，允許負值；未折現。')
        if selected=='整體':
            d=q.groupby('ClusterName').Predicted.agg(Total='sum',Average='mean').reset_index()
            st.markdown('### 客群下一季總預測利潤')
            rank_bar(d,'ClusterName','Total','各客群下一季總預測利潤',signed=True)
            st.markdown('### 客群下一季平均預測利潤')
            rank_bar(d,'ClusterName','Average','各客群下一季平均預測利潤',signed=True)
        else:histogram(q.Predicted,'未來客戶價值分布')
    with st.expander('查詢客戶與下載預測明細'):
        filtered=q.copy()
        if kind=='流失預測':
            risks=st.multiselect('風險等級',RISK,default=RISK,key='churn_risks');filtered=filtered.loc[filtered.RiskLevel.isin(risks)]
        cid=st.text_input('客戶 ID（空白顯示全部）',key=kind+'_cid')
        if cid:filtered=filtered.loc[filtered.CustomerID.astype(str).eq(cid.strip())]
        cols=[x for x in ['CustomerID','ClusterName','Predicted','RiskLevel','SelectedVariant'] if x in filtered]
        display=filtered.sort_values('Predicted',ascending=False)[cols].rename(columns={'CustomerID':'客戶 ID','ClusterName':'客群','Predicted':'流失機率' if kind=='流失預測' else '預測利潤','RiskLevel':'風險等級','SelectedVariant':'部署版本'})
        if kind=='流失預測':display['流失機率']=display['流失機率'].map(lambda x:f'{x:.1%}')
        st.dataframe(display,hide_index=True,width='stretch');st.download_button('下載完整預測 CSV',filtered.to_csv(index=False).encode('utf-8-sig'),kind+'_Customer_Predictions.csv','text/csv',key=kind+'_pred_download')

def revenue_evaluation(base):
    folder=base/'整體季度營收趨勢預測'
    segment.validate_stage(get_project_root(),folder)
    s=read_csv(str(folder/'Forecast_Summary.csv')).iloc[0]
    st.subheader(f'{s.HoldoutQuarter} 獨立時間回測')
    st.metric(f'{s.HoldoutQuarter} 營收預測百分比誤差',f'{s.BacktestMAPE:.1f}%')
    st.caption(f'營收是季度彙總序列，共11筆開發資料；按筆數近似70/15/15，分為 {int(s.TrainCount)}/{int(s.ValidationCount)}/{int(s.TestCount)} 季。驗證集選定 {s.Model}，{s.HoldoutQuarter} 不參與選模。')
    bt=read_csv(str(folder/'Revenue_Backtest.csv'))
    latest=bt.iloc[-1];gap=float(latest.PredictedRevenue-latest.ActualRevenue)
    direction='低估' if gap<0 else '高估' if gap>0 else '無差距'
    actual_col,predicted_col,gap_col=st.columns(3)
    actual_col.metric('實際營收',f'{latest.ActualRevenue:,.0f} 元')
    predicted_col.metric('預測營收',f'{latest.PredictedRevenue:,.0f} 元')
    gap_col.metric('預測差距',f'{direction} {abs(gap)/10000:,.1f} 萬元' if gap else '0 元')
    d=bt.melt(id_vars='Quarter',value_vars=['ActualRevenue','PredictedRevenue'],var_name='Series',value_name='Value')
    d['Series']=d.Series.map({'ActualRevenue':'實際營收','PredictedRevenue':'預測營收'})
    d['Display']=d.Value.map(lambda v:f'{v/10000:,.1f} 萬元')
    high=max(float(d.Value.max()),1.)
    axis=alt.X('Value:Q',title='營收（萬元）',scale=alt.Scale(domain=[0,high*1.25],zero=True),axis=alt.Axis(labelExpr="format(datum.value / 10000, ',.0f')",tickCount=6))
    y=alt.Y('Series:N',sort=['預測營收','實際營收'],title=None,axis=alt.Axis(labelAngle=0))
    base=alt.Chart(d).encode(x=axis,y=y,tooltip=[alt.Tooltip('Quarter:N',title='季度'),alt.Tooltip('Series:N',title='項目'),alt.Tooltip('Value:Q',title='營收（元）',format=',.0f')])
    bars=base.mark_bar(size=36,cornerRadiusEnd=4).encode(color=alt.Color('Series:N',scale=alt.Scale(domain=['實際營收','預測營收'],range=[BLUE,TEAL]),legend=None))
    values=base.mark_text(align='left',dx=8,fontSize=14).encode(text='Display:N')
    show((bars+values).properties(title=f'{latest.Quarter} 預測與實際營收比較',height=260))
    st.caption('獨立回測只有一季，不能代表長期平均準確度。模型未處理季節性；輸入須為完整季度。')
    detail(bt,'Revenue_Backtest')
    st.subheader('內部驗證與測試')
    st.metric('內部測試平均絕對誤差（元）',f'{s.TestMAE:,.0f}')
    for name in ['Revenue_Split','Revenue_Model_Selection','Revenue_Validation','Revenue_Test']:
        detail(read_csv(str(folder/(name+'.csv'))),name)


def customer_evaluation(base,kind):
    folder=base/('CLV價值預測' if kind=='下一季客戶價值預測' else kind)
    segment.validate_stage(get_project_root(),folder)
    meta=json.loads((folder/'Metadata.json').read_text(encoding='utf-8'))
    holdout_label=meta['HoldoutQuarter']+' 獨立回測'
    mode=st.radio('評估資料',[holdout_label,'內部測試集（15%）'],key=kind+'_evaluation_set',horizontal=True)
    holdout=mode==holdout_label
    metric_file='Holdout_Metrics' if holdout else 'Model_Metrics'
    prediction_file='Holdout_Predictions' if holdout else 'Test_Predictions'
    metrics=read_csv(str(folder/(metric_file+'.csv')))
    names=metrics[['Cluster','ClusterName']].drop_duplicates().sort_values('Cluster')
    labels=dict(zip(names.Cluster,names.ClusterName))
    selected=st.selectbox('客群',['整體']+names.Cluster.tolist(),format_func=lambda x:'整體' if x=='整體' else labels[x],key=kind)
    st.caption(f"{meta['SplitDesign']['HistoryStart']}～{meta['SplitDesign']['HistoryEnd']} → 按時間與筆數切70/15/15 → 合格Train分群與訓練 → Validation選模 → Test評估 → {meta['HoldoutQuarter']}獨立回測 → {meta['TargetQuarter']}預測")
    st.subheader(mode)
    m=metrics if selected=='整體' else metrics.loc[metrics.Cluster.eq(selected)]
    if kind=='流失預測':
        metric=st.selectbox('評估指標',['Recall','Precision','F1'],format_func={'Recall':'召回率','Precision':'精確率','F1':'F1'}.get,key='risk_eval')
        d=m.copy();d[metric]=pd.to_numeric(d[metric],errors='coerce')*100
        rank_bar(d,'ClusterName',metric,mode+'分類表現',unit='百分比',percent=True)
        st.caption('評估分類門檻0.5；部署風險門檻0.4／0.7。季度未購買不等於永久流失；機率未另校準。')
    else:rank_bar(m,'ClusterName','MAE',mode+'平均絕對誤差（元）')
    st.caption('內部測試使用僅Train訓練的選定模型。'+meta['Deployment'])
    st.write(meta.get('ClusterWarning',''));st.write(meta.get('EvaluationCaution',''));st.caption(meta.get('Population',''))
    st.dataframe(pd.DataFrame(meta['SplitDesign']['Counts']).rename(columns={'Split':'集合','Assigned':'名義切分筆數','Eligible':'實際使用筆數','Excluded':'邊界排除筆數'}),hide_index=True,width='stretch')
    st.caption('每筆為一位客戶的一季快照；歷史窗第一季作為起始觀察期。特徵使用最近4季（初期使用已有歷史），排除標籤在下一集合觀察日尚未出爐的邊界樣本。')
    detail(m,kind+'_'+metric_file)
    predictions=read_csv(str(folder/(prediction_file+'.csv')))
    if selected!='整體':predictions=predictions.loc[predictions.Cluster.eq(selected)]
    detail(predictions,kind+'_'+prediction_file)
    validation=read_csv(str(folder/'Validation_Results.csv'))
    if selected!='整體':validation=validation.loc[validation.Cluster.eq(selected)]
    detail(validation,kind+'_Validation_Results')
    if kind=='下一季客戶價值預測' and not holdout:
        comp=read_csv(str(folder/'Model_Comparison.csv'))
        if selected!='整體':comp=comp.loc[comp.Cluster.eq(selected)]
        original=comp.loc[comp.Variant.eq('Original'),['Cluster','ClusterName','MAE']].rename(columns={'MAE':'OriginalMAE'})
        chosen=comp.loc[comp.Selected.astype(str).str.lower().isin(['true','1','1.0']),['Cluster','MAE']].rename(columns={'MAE':'SelectedMAE'})
        compare=original.merge(chosen,on='Cluster')
        d=compare.melt(id_vars='ClusterName',value_vars=['OriginalMAE','SelectedMAE'],var_name='版本',value_name='MAE')
        d['版本']=d['版本'].map({'OriginalMAE':'原版','SelectedMAE':'驗證集選定版本'})
        show(alt.Chart(d).mark_bar().encode(y=alt.Y('ClusterName:N',title=None),yOffset='版本:N',x=alt.X('MAE:Q',title='測試平均絕對誤差（元，越低越好）'),color='版本:N',tooltip=['ClusterName:N','版本:N',alt.Tooltip('MAE:Q',format=',.1f')]).properties(height=max(180,65*len(compare))))
        st.caption('版本僅依驗證集選定；這裡呈現內部測試成績，可能改善，也可能退步。')
        detail(comp,'CLV_Model_Comparison')


def render(evaluation=False):
    base=get_project_root()/'預測性分析'/'預測結果'
    from .common import developer_mode
    if developer_mode() and st.button('重新讀取模型結果' if evaluation else '重新讀取預測結果'):
        st.cache_data.clear(); st.rerun()
    names=['整體季度營收趨勢預測','流失預測','下一季客戶價值預測']
    for tab,name in zip(st.tabs(names,key='evaluation_tabs' if evaluation else 'predictive_tabs',on_change='rerun'),names):
        with tab:
            if not tab.open:continue
            try:
                if name==names[0]:
                    (revenue_evaluation if evaluation else revenue)(base)
                else:
                    (customer_evaluation if evaluation else customer)(base,name)
            except (FileNotFoundError,KeyError,ValueError) as exc:
                logging.getLogger(__name__).exception('預測結果載入失敗：%s', base/name)
                st.error(f'{name} 結果暫時無法載入：{exc}')
                st.caption('請先按本頁的重新讀取按鈕。若仍失敗，回首頁查看資料處理狀態。')
                if developer_mode():
                    with st.expander('錯誤詳細資料'):
                        st.code(str(base/name),language='text')
                        st.write(str(exc))

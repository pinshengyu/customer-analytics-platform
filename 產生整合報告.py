"""由本次經驗證的分析結果建立可離線閱讀的 HTML、Markdown 及摘要 JSON。"""
from pathlib import Path
import argparse
import html
import json
from urllib.request import Request,urlopen
import pandas as pd
import segment_contract as segment

E=html.escape

def scalar(v):
    if pd.isna(v):return None
    if hasattr(v,'item'):return v.item()
    return v

def records(d):return [{k:scalar(v) for k,v in row.items()} for row in d.to_dict('records')]
def fmt(v,decimals=0):return '—' if v is None or pd.isna(v) else f'{float(v):,.{decimals}f}'
def read(root,relative):return pd.read_csv(root/relative,encoding='utf-8-sig',low_memory=False)
def table(d,columns):
    return d[columns].rename(columns={'ClusterName':'客群','Mean':'平均流失機率 (%)','HighRisk':'高風險客戶','Count':'客戶數','Total':'總預測利潤','Average':'平均預測利潤','Quarter':'季度','Revenue':'實際營收','PredictedRevenue':'預測營收'}).to_html(index=False,border=0,escape=True,float_format=lambda v:f'{v:,.2f}')

def bars(d,label,value,title,percentage=False,unit='元'):
    unit='%' if percentage else unit
    d=d.sort_values(value,ascending=False);width=900;height=max(180,len(d)*52+70);left=240;right=160;plot=width-left-right
    lo=min(0,float(d[value].min()));hi=max(0,float(d[value].max()));span=hi-lo or 1
    scale=lambda v:left+(float(v)-lo)/span*plot
    bits=[f'<h3>{E(title)}</h3><p class="chart-unit">單位：{E(unit)}</p><svg role="img" aria-label="{E(title)}（{E(unit)}）" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Microsoft JhengHei, Noto Sans CJK TC, sans-serif"><title>{E(title)}（{E(unit)}）</title>']
    bits.append(f'<line x1="{scale(0):.1f}" x2="{scale(0):.1f}" y1="30" y2="{height-30}" stroke="#94a3b8"/>')
    for i,(_,r) in enumerate(d.iterrows()):
        y=40+i*52;v=float(r[value]);x=min(scale(v),scale(0));w=abs(scale(v)-scale(0));color='#dc2626' if v<0 else '#0d9488'
        text=fmt(v,1 if percentage else 0)+('%' if percentage else ' '+unit)
        bits.append(f'<text x="{left-14}" y="{y+18}" text-anchor="end" font-size="14">{E(str(r[label]))}</text><rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="27" rx="3" fill="{color}"/><text x="{scale(v)+7:.1f}" y="{y+19}" font-size="13">{E(text)}</text>')
    bits.append('</svg>');return ''.join(bits)

def revenue_svg(h,f):
    h=h.copy();f=f.copy();h.Quarter=h.Quarter.astype(str).str.replace(' ','',regex=False);f.Quarter=f.Quarter.astype(str).str.replace(' ','',regex=False);h=h.sort_values('Quarter');f=f.sort_values('Quarter')
    quarters=list(h.Quarter)+list(f.Quarter);w=900;height=365;left=90;right=30;top=25;bottom=85
    maximum=max(float(h.Revenue.max()),float(f.UpperEstimate.max()))*1.1 or 1
    x=lambda q:left+quarters.index(q)/max(1,len(quarters)-1)*(w-left-right)
    y=lambda v:height-bottom-float(v)/maximum*(height-top-bottom)
    bits=[f'<p class="chart-unit">縱軸：營收（元）｜橫軸：季度</p><svg role="img" aria-label="實際與預測營收（元）" viewBox="0 0 {w} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Microsoft JhengHei, Noto Sans CJK TC, sans-serif"><title>實際與預測季度營收（元）</title>']
    for tick in range(5):
        value=maximum*tick/4;yy=y(value);bits.append(f'<line x1="{left}" x2="{w-right}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#e2e8f0"/><text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-size="12">{fmt(value)}</text>')
    polygon=[(x(r.Quarter),y(r.UpperEstimate)) for r in f.itertuples()]+[(x(r.Quarter),y(r.LowerEstimate)) for r in reversed(list(f.itertuples()))]
    bits.append('<polygon points="'+' '.join(f'{a:.1f},{b:.1f}' for a,b in polygon)+'" fill="#0d9488" opacity=".15"/>')
    for data,color,dash in [(list(zip(h.Quarter,h.Revenue)),'#2563eb',''),([(h.iloc[-1].Quarter,h.iloc[-1].Revenue)]+list(zip(f.Quarter,f.PredictedRevenue)),'#0d9488','stroke-dasharray="7 5"')]:
        bits.append('<polyline points="'+' '.join(f'{x(q):.1f},{y(v):.1f}' for q,v in data)+f'" fill="none" stroke="{color}" stroke-width="3" {dash}/>')
        for q,v in data:bits.append(f'<circle cx="{x(q):.1f}" cy="{y(v):.1f}" r="3" fill="{color}"><title>{E(q)}：{fmt(v)} 元</title></circle>')
    for q in quarters:bits.append(f'<text transform="translate({x(q):.1f},{height-bottom+23}) rotate(-35)" text-anchor="end" font-size="12">{E(q)}</text>')
    bits.append('</svg>');return ''.join(bits)

def cohort_svg(m):
    qs=[f'Q{i}' for i in range(5) if f'Q{i}' in m];height=len(m)*34+65;width=800;left=130;cell=(width-left)/len(qs)
    bits=[f'<p class="chart-unit">單位：留存率（%）｜Q0～Q4：距首次購買的季度數</p><svg role="img" aria-label="季度留存熱圖（%）" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Microsoft JhengHei, Noto Sans CJK TC, sans-serif"><title>季度留存熱圖（%）</title>']
    for j,q in enumerate(qs):bits.append(f'<text x="{left+j*cell+cell/2}" y="22" text-anchor="middle" font-size="14">{q}</text>')
    for i,r in enumerate(m.itertuples()):
        yy=35+i*34;bits.append(f'<text x="{left-12}" y="{yy+22}" text-anchor="end" font-size="13">{E(str(r.CohortQuarter))}</text>')
        for j,q in enumerate(qs):
            v=getattr(r,q);a=0 if pd.isna(v) else max(0,min(1,float(v)/100));color='#e5e7eb' if pd.isna(v) else f'rgb({round(235-205*a)},{round(245-181*a)},{round(255-80*a)})';text='—' if pd.isna(v) else fmt(v)+'%'
            bits.append(f'<rect x="{left+j*cell:.1f}" y="{yy}" width="{cell-2:.1f}" height="32" fill="{color}"/><text x="{left+j*cell+cell/2:.1f}" y="{yy+22}" text-anchor="middle" font-size="13" fill="'+('white' if a>.55 else '#334155')+f'">{text}</text>')
    return ''.join(bits)+'</svg>'

def build(root,job_id='',source_manifest=None,llm=False,model='llama3.2:3b',ollama_url='http://localhost:11434'):
    from quarterly_plan import load_plan
    root=Path(root);context=segment.context(root);plan=load_plan(root)
    for rel in ['描述性分析/Operational_KPI','描述性分析/AARRR','描述性分析/Historical_CLV','診斷性分析','預測性分析/預測結果/整體季度營收趨勢預測','預測性分析/預測結果/流失預測','預測性分析/預測結果/CLV價值預測']:segment.validate_stage(root,root/rel)
    op=read(root,'描述性分析/Operational_KPI/Operational_Overall_KPI.csv').iloc[0]
    oc=read(root,'描述性分析/Operational_KPI/Operational_Cluster_KPI.csv')
    a=read(root,'描述性分析/AARRR/AARRR_Overall_KPI.csv').iloc[0]
    clv=read(root,'描述性分析/Historical_CLV/Historical_CLV_Overall.csv').iloc[0]
    cohort=read(root,'描述性分析/Cohort/Cohort_Retention_Matrix.csv')
    diagnostic=read(root,'診斷性分析/營收驅動/Diagnostic_Quarterly_Summary.csv').sort_values('Quarter');latest=diagnostic.iloc[-1]
    findings=read(root,'診斷性分析/Diagnostic_Findings.csv');findings=findings.loc[findings.Period.astype(str).eq(str(latest.Quarter))];anomalies=findings.loc[findings.Anomaly.astype(str).str.lower().isin(['true','1','1.0'])]
    pred='預測性分析/預測結果/'
    h=read(root,pred+'整體季度營收趨勢預測/Revenue_History.csv');f=read(root,pred+'整體季度營收趨勢預測/Revenue_Forecast.csv');forecast=read(root,pred+'整體季度營收趨勢預測/Forecast_Summary.csv').iloc[0]
    churn=read(root,pred+'流失預測/Customer_Predictions.csv');future=read(root,pred+'CLV價值預測/Customer_Predictions.csv')
    cs=churn.assign(HighRisk=churn.Predicted.ge(.7)).groupby('ClusterName').agg(Count=('CustomerID','size'),Mean=('Predicted','mean'),HighRisk=('HighRisk','sum')).reset_index();cs['RiskPct']=cs.HighRisk/cs.Count*100
    fs=future.groupby('ClusterName').Predicted.agg(Total='sum',Average='mean',Count='size').reset_index()
    metrics=read(root,pred+'CLV價值預測/Model_Metrics.csv');comparison=read(root,pred+'CLV價值預測/Model_Comparison.csv')
    clv_holdout=read(root,pred+'CLV價值預測/Holdout_Metrics.csv')
    churn_holdout=read(root,pred+'流失預測/Holdout_Metrics.csv')
    design=json.loads((segment.segment_dir(root)/'Quarterly_Design.json').read_text(encoding='utf-8'))
    top=oc.loc[oc.TotalRevenue.idxmax()];risk=cs.loc[cs.RiskPct.idxmax()];value=fs.loc[fs.Total.idxmax()]
    facts=[f"歷史總營收 {fmt(op.TotalRevenue)}，訂單 {fmt(op.TotalOrders)}，購買客戶 {fmt(op.PurchasingCustomers)}；累積已實現利潤 {fmt(clv.TotalHistoricalProfit)}。",
           f"營收最高客群為 {top.ClusterName}，營收占比 {fmt(top.RevenueShare,1)}%。",
           f"下一季 {forecast.NextForecastQuarter} 預測營收 {fmt(forecast.NextQuarterPredictedRevenue)}，預測季增率 {fmt(forecast.NextQuarterChangePct,1)}%；{plan.holdout}單季回測百分比誤差 {fmt(forecast.BacktestMAPE,1)}%。",
           f"模型高風險客戶 {fmt(churn.Predicted.ge(.7).sum())} 位，占評分客戶 {fmt(churn.Predicted.ge(.7).mean()*100,1)}%；高風險比例最高的客群為 {risk.ClusterName}（{fmt(risk.RiskPct,1)}%）。",
           f"下一季度（{plan.target}）預測利潤合計 {fmt(future.Predicted.sum())}；總預測利潤最高客群為 {value.ClusterName}。"]
    if not anomalies.empty:facts.append('最新季度 '+str(latest.Quarter)+' 超過下降門檻的指標：'+'、'.join(anomalies.KPI.astype(str)))
    elif len(findings) and findings.ComparisonStatus.eq('Comparable').all():facts.append('最新季度 '+str(latest.Quarter)+' 未出現超過設定下降門檻的 KPI。')
    else:facts.append('最新季度存在無法比較的 KPI；不可把空白判斷當成正常。')
    for _,finding in anomalies.iterrows():
        if pd.notna(finding.get('MainDriver')):facts.append(f"{finding.KPI} 的主要客群歸因來源：{finding.MainDriver}；請於診斷頁下鑽，不能視為已證明的因果關係。")
    actions=['先檢視高風險比例與人數較高的客群，結合近期購買紀錄規劃維繫方案。','比較客群總預測利潤與每人預測利潤，分別評估規模與客戶價值。','若最新季度觸發異常，依診斷來源下鑽檢查商品、交易通路與活動；歸因結果不等於因果證明。']
    notes=['所有分群分析使用僅由合格Train建立的分群中心及 LLM 客群名稱；預測快照逐季指派，歷史報表使用固定客戶對照；Cohort 不使用分群。','流失機率未另做校準；高風險門檻70%，測試分類門檻50%。',f'CLV 是下一季度（{plan.target}）利潤預測，允許負值，未折現；部署版本由15%驗證集決定。','營收淡色範圍來自內部測試誤差，未校準覆蓋率，不是90%信賴區間。','建議是待驗證的行動方向；目前專案未包含策略模擬，報告不宣稱已驗證ROI。']
    source=json.loads(Path(source_manifest).read_text(encoding='utf-8')) if source_manifest else {}
    notes+=[f'{plan.history_start}為觀察暖身；{plan.first_label}～{plan.history_end}客戶季度快照按目標季度、CustomerID排序切70/15/15。跨界未到期標籤排除，同一客戶可跨季出現。',
            '切分筆數（名義／實際使用／排除）：'+ '；'.join(f"{r['Split']} {r['Assigned']}/{r['Eligible']}/{r['Excluded']}" for r in design['Counts']),
            f'{plan.holdout}獨立回測不參與選模；回測模型訓練至{plan.history_end}，{plan.target}部署模型加入回測季已知標籤重訓。',
            '營收是季度彙總序列，11筆開發資料近似切為7/2/2季；獨立回測只有一筆，不能代表長期平均表現。',
            '季度流失指下一季未購買，不等於永久流失。預測對象限最近4季曾購買客戶；早期快照使用已有歷史。']
    summary={'JobID':job_id,'Context':context,'SourceValidation':source.get('Validation',[]),'InputHashes':source.get('InputHashes',{}),'Facts':facts,'Actions':actions,'Limitations':notes,'Forecast':{k:scalar(v) for k,v in forecast.items()},'ChurnClusters':records(cs),'CLVClusters':records(fs),'CLVModelMetrics':records(metrics),'CLVComparison':records(comparison),'LLMEnabled':llm}
    interpretation=''
    summary.update(SplitDesign=design,CLVHoldoutMetrics=records(clv_holdout),ChurnHoldoutMetrics=records(churn_holdout))
    if llm:
        from pipeline_runtime import ollama_url as valid_url
        url=valid_url(ollama_url)
        prompt='請以繁體中文撰寫簡短的客戶分析報告解讀與三項待驗證行動建議。只能引用以下已計算的事實，不能虛構數字、因果關係、ROI或策略成效。清楚區分歷史結果與未來預測。不要產生HTML。\n'+json.dumps({'Facts':facts,'Limitations':notes},ensure_ascii=False)
        request=Request(url+'/api/generate',data=json.dumps({'model':model,'prompt':prompt,'stream':False,'options':{'temperature':0.1,'num_predict':1200}}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as response:interpretation=json.load(response).get('response','').strip()
        if not interpretation:raise ValueError('LLM 報告沒有回傳文字；請檢查 Ollama。')
        notes.append('LLM 解讀為生成文字，可能有錯誤；以本報告已計算指標為準。')
        summary['LLMText']=interpretation
    out=root/'報告結果';out.mkdir(parents=True,exist_ok=True)
    markdown='# 客戶分析整合報告\n\n批次：'+job_id+'\n\n## 重點結果\n\n'+'\n'.join('- '+x for x in facts)+'\n\n## 待驗證行動\n\n'+'\n'.join('- '+x for x in actions)+'\n\n## 定義與限制\n\n'+'\n'.join('- '+x for x in notes)
    if interpretation:markdown+='\n\n## LLM 解讀（生成文字，需人工確認）\n\n'+interpretation
    (out/'客戶分析整合報告.md').write_text(markdown,encoding='utf-8')
    sections='<section><h2>重點結果</h2><ul>'+''.join('<li>'+E(x)+'</li>' for x in facts)+'</ul></section>'
    sections+='<section><h2>季度營收：實際與預測</h2>'+revenue_svg(h,f)+'<p>藍色實線：歷史；綠色虛線：預測；淡色區：經驗估計範圍。</p></section>'
    sections+='<section><h2>流失風險：先檢視哪些客群？</h2>'+bars(cs,'ClusterName','RiskPct','各客群高風險占比',True)+table(cs.assign(Mean=cs.Mean*100),['ClusterName','Count','HighRisk','Mean'])+'</section>'
    sections+=f'<section><h2>下一季度（{plan.target}）價值</h2>'+bars(fs,'ClusterName','Total','各客群總預測利潤')+bars(fs,'ClusterName','Average','各客群平均每人預測利潤',unit='元／人')+'</section>'
    sections+='<section><h2>保留測試集：CLV 預測誤差</h2>'+bars(metrics,'ClusterName','MAE','CLV平均絕對誤差')+'<p>平均絕對誤差越低越好。此處是各群驗證集選定版本的保留測試集成績，不是完整資料重訓成績。</p></section>'
    sections+='<section><h2>季度 Cohort 留存</h2>'+cohort_svg(cohort)+'<p>灰色「—」代表尚未到達的季度；0% 代表已觀察但沒有回購。</p></section>'
    sections+=f'<section><h2>{plan.holdout} 獨立回測</h2>'+bars(clv_holdout,'ClusterName','MAE','獨立回測利潤預測平均絕對誤差')+bars(churn_holdout.assign(RecallPct=churn_holdout.Recall*100),'ClusterName','RecallPct','獨立回測流失召回率',True)+f'<p>使用訓練至{plan.history_end}的模型；此處與內部測試成績分開，{plan.holdout}不參與版本選擇。</p></section>'
    sections+='<section><h2>AARRR 活動率</h2><p>啟用率 '+fmt(a.ActivationRate,1)+'%；再購率 '+fmt(a.RepeatPurchaseRate,1)+'%；推薦率 '+fmt(a.ReferralRate,1)+'%。各項率依原分析分母定義，不能當成同一漏斗逐階流失。</p></section>'
    sections+='<section><h2>待驗證行動</h2><ol>'+''.join('<li>'+E(x)+'</li>' for x in actions)+'</ol><h3>定義與限制</h3><ul>'+''.join('<li>'+E(x)+'</li>' for x in notes)+'</ul></section>'
    if interpretation:sections+='<section><h2>LLM 解讀（生成文字，需人工確認）</h2><div class="llm">'+E(interpretation)+'</div></section>'
    page='<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>客戶分析整合報告</title><style>body{font-family:system-ui,"Microsoft JhengHei",sans-serif;background:#f1f5f9;color:#0f172a;margin:0;padding:28px}main{max-width:1000px;margin:auto}section{background:white;border-radius:12px;padding:24px;margin:20px 0;box-shadow:0 2px 8px #0f172a08}h1{font-size:28px}h2{font-size:21px}p,li{line-height:1.8}li{margin:8px 0}svg{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child{text-align:left}.llm{white-space:pre-wrap;line-height:1.8}@media print{body{background:white;padding:0}section{box-shadow:none;break-inside:avoid}}</style><main><h1>客戶分析整合報告</h1><p>批次 '+E(job_id)+' · 共用分群版本 '+E(context['SegmentVersion'])+'</p>'+sections+'</main></html>'
    (out/'客戶分析整合報告.html').write_text(page,encoding='utf-8');(out/'報告摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('報告完成：'+str(out),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path);parser.add_argument('--job-id',default='手動產生');parser.add_argument('--source-manifest',type=Path);parser.add_argument('--llm',action='store_true');parser.add_argument('--model',default='llama3.2:3b');parser.add_argument('--ollama-url',default='http://localhost:11434');a=parser.parse_args();build(a.project_root or Path(__file__).resolve().parent,a.job_id,a.source_manifest,a.llm,a.model,a.ollama_url)

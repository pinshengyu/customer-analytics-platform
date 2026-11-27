"""圖表共用樣式與彙總；圖上呈現資訊，原始明細放進展開區。"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

BLUE='#2563eb';TEAL='#0d9488';RED='#dc2626';AMBER='#d97706';GRAY='#94a3b8'
PALETTE=['#2563eb','#0d9488','#7c3aed','#ea580c','#db2777','#0891b2','#65a30d']
RISK=['高風險','中風險','低風險'];RISK_COLORS=[RED,AMBER,TEAL]
AMOUNT_AXIS="abs(datum.value) >= 100000000 ? format(datum.value / 100000000, '.1f') + '億' : abs(datum.value) >= 10000 ? format(datum.value / 10000, '.0f') + '萬' : format(datum.value, ',.0f')"

def show(chart):
    st.altair_chart(
        chart.configure_view(stroke=None)
        .configure_title(fontSize=17, anchor='start', offset=14)
        .configure_axis(labelFontSize=13, titleFontSize=13, labelLimit=260, gridOpacity=.10, titlePadding=10)
        .configure_legend(title=None, labelFontSize=12, orient='bottom', columns=4, symbolLimit=20),
        width='stretch'
    )

def brief(text):
    st.info(text)

def detail(df,name,key=None,columns=None):
    with st.expander('查看明細／下載 '+name):
        st.dataframe(df if columns is None else df[[c for c in columns if c in df]],hide_index=True,width='stretch')
        st.download_button('下載完整 CSV',df.to_csv(index=False).encode('utf-8-sig'),name+'.csv','text/csv',key=key or name+'_download')

def cluster_color(names):
    domain=sorted(set(map(str,names)))
    return alt.Color('ClusterName:N',scale=alt.Scale(domain=domain,range=PALETTE[:len(domain)] if len(domain)<=len(PALETTE) else PALETTE),legend=None)

def rank_bar(df,label,value,title,unit='金額',percent=False,signed=False,cluster=False):
    d=df[[label,value]].copy();d[value]=pd.to_numeric(d[value],errors='coerce');d=d.dropna()
    if d.empty:st.caption('尚無可繪製資料。');return
    d[label]=d[label].astype(str);d['Display']=d[value].map(lambda v:f'{v:,.1f}%' if percent else f'{v:,.0f}')
    color=cluster_color(d[label]) if cluster and label=='ClusterName' else (alt.condition(alt.datum[value]<0,alt.value(RED),alt.value(TEAL)) if signed else alt.value(BLUE))
    axis=alt.Axis(labelExpr="datum.label + '%'" if percent else AMOUNT_AXIS if unit=='金額' else "format(datum.value, ',.0f')")
    base=alt.Chart(d).encode(y=alt.Y(label+':N',sort='ascending' if label in ('Quarter','CohortQuarter') else alt.SortField(field=value,order='descending'),title=None),x=alt.X(value+':Q',title=unit,axis=axis),tooltip=[alt.Tooltip(label+':N',title='項目'),alt.Tooltip(value+':Q',title=title,format=',.2f' if percent else ',.0f')])
    bars=base.mark_bar(cornerRadiusEnd=4).encode(color=color)
    labels=base.mark_text(align='left',dx=6,fontSize=12).encode(text='Display:N')
    # 多留文字空間；資料為負也保持零基準線。
    lo=min(0.,float(d[value].min())*1.18);hi=max(0.,float(d[value].max())*1.25)
    if lo==hi:hi=1.
    chart=(bars+labels).encode(x=alt.X(value+':Q',title=unit,axis=axis,scale=alt.Scale(domain=[lo,hi])))
    if signed:
        zero=alt.Chart(pd.DataFrame({'Zero':[0]})).mark_rule(color=GRAY).encode(x='Zero:Q');chart=chart+zero
    show(chart.properties(title=title,height=max(240,min(680,46*len(d)))))

def line(df,x,y,title,series=None,unit='金額'):
    d=df.copy()
    if x=='Quarter':d[x]=d[x].astype(str).str.replace(' ','',regex=False)
    d=d.dropna(subset=[y]);d=d.sort_values(x)
    if d.empty:st.caption('尚無可繪製資料。');return
    kwargs={'x':alt.X(x+':O',title=None,axis=alt.Axis(labelAngle=0,labelOverlap=True)), 'y':alt.Y(y+':Q',title=unit,axis=alt.Axis(labelExpr=AMOUNT_AXIS) if unit=='金額' else alt.Axis()),'tooltip':[alt.Tooltip(x+':O',title='期間'),alt.Tooltip(y+':Q',title=title,format=',.1f')]}
    if series:kwargs['color']=alt.Color(series+':N',scale=alt.Scale(range=PALETTE));kwargs['tooltip'].append(alt.Tooltip(series+':N',title='客群'))
    else:kwargs['color']=alt.value(BLUE)
    show(alt.Chart(d).mark_line(point=True,strokeWidth=3).encode(**kwargs).properties(title=title,height=380))

def histogram(values,title,unit='金額'):
    a=pd.to_numeric(values,errors='coerce').dropna().to_numpy(float);a=a[np.isfinite(a)]
    if not len(a):st.caption('尚無分布資料。');return
    counts,edges=np.histogram(a,bins=min(24,max(6,int(np.sqrt(len(a))))))
    d=pd.DataFrame({'Lower':edges[:-1],'Upper':edges[1:],'Count':counts})
    show(alt.Chart(d).mark_bar(color=BLUE,opacity=.85).encode(x=alt.X('Lower:Q',bin='binned',title=unit,axis=alt.Axis(labelExpr=AMOUNT_AXIS) if unit=='金額' else alt.Axis()),x2='Upper:Q',y=alt.Y('Count:Q',title='客戶數'),tooltip=[alt.Tooltip('Lower:Q',title='下界',format=',.0f'),alt.Tooltip('Upper:Q',title='上界',format=',.0f'),alt.Tooltip('Count:Q',title='客戶數',format=',')]).properties(title=title,height=340))

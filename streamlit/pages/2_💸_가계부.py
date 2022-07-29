import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
ROOT_DIR_MODEL = os.path.join(ROOT_DIR, 'model')
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR_MODEL)
import streamlit as st
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
#from model.inference import inference
import plotly.express as px
import plotly.graph_objects as go

from model.inference_streamlit import load_model, inference
from model.modules.preprocess import preprocess_infer

st.set_page_config(layout= 'wide')

@st.experimental_singleton(show_spinner = False)
def loading_model_sidebar(select_model, maxlen, lossfn):
    return load_model(select_model, maxlen, lossfn)


@st.experimental_memo(show_spinner = False)
def loading_csv_data(upload):
    # loading
    # if len(upload) == 1:
    #     load_data_ = pd.read_csv(upload[0], index_col = 0)

    # elif len(upload) >= 2:
    load_data_ = pd.DataFrame(columns = ['거래일자','업체명','금액'])
    for i in upload:
        try:
            read_data = pd.read_csv(i, index_col = 0)
        except:
            read_data = pd.read_csv(i, index_col = 0, encoding = 'euc-kr')
        load_data_ = pd.concat([load_data_, read_data])
    load_data_ = load_data_.reset_index(drop = True)

    return load_data_

# @st.experimental_memo(show_spinner = False)
def inference_csv_data(load_data, maxlen, _model, _tokenizer):
    inferdata, _ = inference(preprocess_infer(load_data), maxlen, _model, _tokenizer)
    weekday_list = ['월','화','수','목','금','토','일']
    day = []
    for i in range(len(inferdata)):
        day.append(weekday_list[datetime.datetime.strptime(inferdata['거래일자'][i], "%Y-%m-%d").weekday()])
    inferdata['요일'] = day
    return inferdata

sidebar = st.sidebar
with sidebar:
        with st.container():
            st.subheader('모델 설정')
            select_model = st.selectbox('Select Model', ['KoBERT', 'KoELECTRA', 'RoBERTa'])
            maxlen = st.radio('Select Max Sequence Length', [26, 32])
            lossfn = st.radio('Select Loss Function', ['Cross Entropy', 'Weighted Cross Entropy', 'Focal Loss'])
            model_confirm = st.button('확인', key = 'model_confirm')
            if model_confirm:
                st.session_state['model_config_2'] = f'{select_model}_{maxlen}_{"".join(lossfn.split())}'
                if maxlen == 32 and lossfn == 'Weighted Cross Entropy':
                    st.warning('Wrong Hyperparameter!!!')
                else:
                    with st.spinner(f'Loading Model... \n <{select_model}_{maxlen}_{"".join(lossfn.split())}>'):
                        model, tokenizer = loading_model_sidebar(select_model, maxlen, lossfn)
                        st.session_state['model_2'] = model
                        st.session_state['tokenizer_2'] = tokenizer
                    st.success(f'Successfully load model!')
            try:
                if st.session_state.model_config_2:
                    st.write(f'현재 사용 모델 : {st.session_state.model_config_2}')
            except AttributeError:
                pass
        st.markdown('***')
        with st.container():
            st.subheader('CSV 파일 업로드')
            upload = st.file_uploader(
                'The names of columns should be: \'거래일자\', \'업체명\', \'금액\', encoding should be \'utf-8\'', 
                type = 'csv',
                accept_multiple_files = True)
            csv_upload = st.button('업로드', key = 'csv_upload')
            if upload and csv_upload:
                with st.spinner('Loading & Inferencing CSV file'):
                    rrdata = loading_csv_data(upload)
                    rrdata = inference_csv_data(rrdata, maxlen, st.session_state.model_2, st.session_state.tokenizer_2)
                    st.session_state['old'] = st.session_state.model_2
                    rdata = rrdata.sort_values(by = '거래일자').reset_index(drop = True)
                    st.session_state['data'] = rdata
                st.success(f'Successfully load and inference CSV file!')
            if upload:
                st.write('현재 사용 데이터 :')
                st.write(', '.join([upload[i].name for i, _ in enumerate(upload)]))

st.header('💸가계부🧾')

try:
    if st.session_state.model_2 and st.session_state.model_config_2 == f'{select_model}_{maxlen}_{"".join(lossfn.split())}':
        if st.session_state.model_2 != st.session_state.old:
            st.warning('Model is changed. You have to re-upload your CSV file.')
        data = st.session_state.data
        with st.container():
            date = st.date_input('조회할 기간을 선택하세요', [ datetime.datetime.strptime(data['거래일자'].min(), '%Y-%m-%d').date(), 
                                    datetime.datetime.strptime(data['거래일자'].max(), '%Y-%m-%d').date()  ])

        with st.container():
            st.subheader('소비 리포트')
            total_pie, weekly, previous, budget_manage = st.tabs(['전체', '업종 별', '전월 대비', '예산 관리'])

            with total_pie: 
                if len(date) == 2:  
                    with st.container():    
                        col1, col2 = st.columns(2)
                                
                        with col1:
                            df = data[(data['거래일자'] >=date[0].strftime('%Y-%m-%d')) & (data['거래일자'] <= date[1].strftime('%Y-%m-%d'))].groupby(by = '업종')
                            # df = data[ data['거래일자'].isin( pd.date_range( date[0], date[1] ) ) ].groupby(data['업종'])

                            fig = px.pie(
                                        df['금액'].sum(), 
                                        title = '업종 분포', 
                                        values = df['금액'].sum().tolist(), 
                                        names = df['금액'].sum().index,             # names = '업종'으로 바꿔야 함
                                        hole = 0.3
                                        )   
                            fig.update_yaxes(tickformat=',')            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:

                            if date is not None: 
                                df_date = data[(data['거래일자'] >=date[0].strftime('%Y-%m-%d')) & (data['거래일자'] <= date[1].strftime('%Y-%m-%d'))]

                                # 전체 시계열
                                timeseries_df = df_date['금액'].groupby(df_date['거래일자']).sum()
                                fig = px.line( timeseries_df, title = '전체 소비 추이', x=timeseries_df.index, y='금액')
                                fig.update_yaxes(tickformat=',')
                                st.plotly_chart(fig)

                    with st.container():
                        # 카테고리 별 시계열
                        category = st.multiselect('업종을 선택하세요', df_date['업종'].value_counts().index, key= 2)

                        if category:
                            df_cat = df_date[df_date['업종'].str.contains('|'.join(category))][['거래일자','요일','업체명','업종','금액']]

                            if len(df_cat) == 0:
                                st.write('거래 내역이 없습니다.')

                            else:
                                fig = px.line(df_cat, title = '업종 별 소비 추이', x = '거래일자', y = '금액', color = '업종', hover_data = ['업체명', '금액'])
                                fig.update_yaxes(tickformat=',')
                                st.plotly_chart(fig)


            #요일별
            with weekly:
                col1, col2 = st.columns(2)
                if len(date) == 2:
                    data_ = data[(data['거래일자'] >=date[0].strftime('%Y-%m-%d')) & (data['거래일자'] <= date[1].strftime('%Y-%m-%d'))].fillna(0)
                    if len(data_) == 0:
                        st.write('거래 내역이 없습니다')
                    else:
                        
                        with col1:
                            st.header('모든 업종')
                            data_['요일'] = data_['요일'].astype(str)
                            chartdata = data_.groupby(by = '요일').sum().reindex(['월','화','수','목','금','토','일']).fillna(0).astype(int)
                            chartdata1 = data_.groupby(by = '업종').sum()
                            chartdata2 = data_.groupby(by = ['요일','업종']).sum().reset_index()
                            chartdata_ = chartdata.transpose()
                            chartdata_['합계'] = chartdata_.sum(axis = 1)
                            #st.write(chartdata_)
                            st.write(date[0].strftime('%Y/%m/%d'), '~', date[1].strftime('%Y/%m/%d') , ' 소비한 금액은 총 ', chartdata_['합계']['금액'], '원입니다.')

                            chartdata = chartdata.reset_index()
                            check = st.checkbox('업종 별로도 확인할래요?')
                            if check:
                                fig2 = px.bar(chartdata2, x="요일", y="금액",color = '업종',
                                            height=400,
                                            category_orders = {'요일':['월','화','수','목','금','토','일']})
                                fig2.update_yaxes(tickformat=',')
                                st.plotly_chart(fig2)
                            else:
                                trace1 = go.Bar(x = chartdata['요일'], y = chartdata['금액'], text = chartdata['금액'], textposition = 'outside')
                                bar = [trace1]
                                fig = go.Figure(data = bar)
                                fig.update_yaxes(tickformat=',')
                                st.plotly_chart(fig)
                            

                        with col2:    
                            st.write( date[0].strftime('%Y/%m/%d'), '~', date[1].strftime('%Y/%m/%d') + ' 업종 + 요일 별 소비한 금액')
                            chartdata4 = pd.pivot_table(data_, index = '업종', columns = '요일', values = '금액', aggfunc = 'sum').fillna(0).astype(int)
                            chartdata4 = chartdata4[['월','화','수','목','금','토','일']]
                            fig1 = px.imshow(chartdata4, x = ['월','화','수','목','금','토','일'], y = chartdata4.index,color_continuous_scale='Greens', aspect='auto')
                            fig1.update_xaxes(tickformat=',')
                            st.plotly_chart(fig1)
                            chartdata5 = chartdata4.copy()
                            chartdata5['합계'] = chartdata5.sum(axis = 1)
                            chartdata5.loc['합계'] = chartdata5.sum(axis = 0)
                            with st.expander('상세 내역 보기'):
                                st.write(chartdata5)

                        st.markdown('***')
                        
                    
                        st.header('선택한 업종')
                        category = st.multiselect('업종을 선택하세요', data['업종'].value_counts().index)  ### list
                        
                        if category:
                            data__ = data_[data_['업종'].str.contains('|'.join(category))][['거래일자','요일','업체명','업종','금액']]

                            if len(data__) == 0:
                                st.write('거래 내역이 없습니다.')
                            else:
                                st.write(date[0].strftime('%Y/%m/%d'), '~', date[1].strftime('%Y/%m/%d') + ' 요일 별', ', '.join(category) + '에 소비한 금액')
                                chartdata = data__.groupby(by = '요일').sum().reindex(['월','화','수','목','금','토','일']).fillna(0).astype(int)
                                chartdata1 = data__.groupby(by = '업종').sum()
                                chartdata2 = data__.groupby(by = ['요일','업종']).sum().reset_index()
                                chartdata3 = data__.pivot_table(index = '요일', 
                                                            columns = '업종', 
                                                            values = '금액', 
                                                            aggfunc = 'sum',).reindex(['월','화','수','목','금','토','일']).fillna(0).astype(int).transpose()
                                
                                chartdata3['합계'] = chartdata3.sum(axis = 1)
                                chartdata3.loc['합계'] = chartdata3.sum(axis = 0)
                                chartdata = chartdata.reset_index()
                                if len(category) >= 2:
                                    check = st.checkbox('업종 별로도 확인할래요?', key = 1)
                                    if check:
                                        fig2 = px.bar(chartdata2, x="요일", y="금액",color = '업종',
                                                    height=400,
                                                    category_orders = {'요일':['월','화','수','목','금','토','일']})
                                        fig2.update_layout(barmode = 'group')
                                        fig2.update_yaxes(tickformat=',')
                                        st.plotly_chart(fig2)
                                    else:
                                        trace1 = go.Bar(x = chartdata['요일'], y = chartdata['금액'])
                                        bar = [trace1]
                                        fig = go.Figure(data = bar)
                                        fig.update_yaxes(tickformat=',')
                                        st.plotly_chart(fig)
                                else:
                                    trace1 = go.Bar(x = chartdata['요일'], y = chartdata['금액'])
                                    bar = [trace1]
                                    fig = go.Figure(data = bar)
                                    fig.update_yaxes(tickformat=',')
                                    st.plotly_chart(fig)

                                with st.expander('펼쳐보기'):
                                    st.write(data__.set_index('거래일자'))
                                    data__['요일'] = data__['요일'].astype(str)
                                    st.write(chartdata3)
                            


            # 전월 대비
            with previous:
                if len(date) == 2:
                    prev_date = date[1] - relativedelta(months = 1)
                    prev_m = prev_date.strftime('%Y-%m')
                    pres_m = date[1].strftime('%Y-%m')

                    prev = data[data['거래일자'].str.contains(prev_m)]
                    pres = data[data['거래일자'].str.contains(pres_m)]

                    
                    prev_ = prev.groupby(by = '업종').sum()
                    prev_['월'] = [prev_m] * len(prev_)

                    pres_ = pres.groupby(by = '업종').sum()
                    pres_['월'] = [pres_m] * len(pres_)

                    conc = pd.concat([prev_, pres_])
                    

                    if len(prev_) == 0 or len(pres_) == 0:
                        st.warning('비교할 수 없습니다! 기간을 다시 설정해주세요!')

                    else:
                        col1, col2 = st.columns(2)            
                        
                        with col1:
                            st.header(prev_m + ' 거래 내역 요약')
                            with st.expander('상세 거래 내역 보기 - ' + prev_m):
                                st.write(prev.sort_values(by = '거래일자').set_index('거래일자')[['업체명','업종','금액','요일']])
                            pie1 = px.pie(prev_.reset_index(), values= '금액', names = '업종', title = '업종 별 사용 금액 비율 - ' + prev_m, color = '업종')
                            st.plotly_chart(pie1)

                        with col2:
                            st.header(pres_m + ' 거래 내역 요약')
                            with st.expander('상세 거래 내역 보기 - ' + pres_m):
                                st.write(pres.sort_values(by = '거래일자').set_index('거래일자')[['업체명','업종','금액','요일']])
                            pie2 = px.pie(pres_.reset_index(), values= '금액', names = '업종', title = '업종 별 사용 금액 비율 - ' + pres_m, color = '업종')
                            st.plotly_chart(pie2)

                        col1_, col2_ = st.columns(2)
                        
                        with col1_:
                            st.header('전월 대비 비교')

                            with st.expander('상세 보기'):
                                conc_ = conc.reset_index().pivot_table(index = '월', values = '금액', columns = '업종', aggfunc = 'sum').fillna(0).astype(int)
                                conc_.loc['전월 대비'] = conc_.diff().iloc[1,:].fillna(0).astype(int)
                                st.dataframe(conc_)

                            comparison = conc_.loc['전월 대비']
                            fig = px.bar(comparison, x=comparison.index, y = '전월 대비', height=600)
                            fig.update_yaxes(tickformat=',')
                            st.plotly_chart(fig)
                        

                        with col2_:
                            with st.expander('세부 거래 내역 펼쳐보기'):
                                st.header('세부 거래 내역')             
                                category = st.selectbox('choose one', sorted(conc.index.unique()))
                                if category:
                                    when = st.radio(prev_m + ' or ' + pres_m, (prev_m, pres_m), horizontal = True)
                                    if when == prev_m:
                                        datashow = prev[prev['업종'] == category].sort_values(by = '거래일자').set_index('거래일자')[['업체명','업종','금액','요일']]
                                    elif when == pres_m:
                                        datashow = pres[pres['업종'] == category].sort_values(by = '거래일자').set_index('거래일자')[['업체명','업종','금액','요일']]
                                    if len(datashow) == 0:
                                        st.write('거래 내역이 없습니다')
                                    else:
                                        st.write(datashow)





            with budget_manage:
                st.header('서비스 준비 중..!')
            #     total = data['금액'][data['거래일자'].isin(pd.date_range( datetime.datetime(datetime.date.today().year, datetime.date.today().month, 1) + relativedelta(months = -1), 
            #                                                             datetime.datetime(datetime.date.today().year, datetime.date.today().month, 1) + relativedelta(seconds = -1) 
            #                                                             )
            #                                                             )
            #                                                             ].sum()
            #     st.write(f'이번 달 지출은 총 {total}원입니다.')                  #전월 합산
                
            #     # 전체 & 업종 별 예산
            #     budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')
            #     # budget = st.number_input('전체 예산')

            #     if st.button('저장'):
            #         if budget < total:
            #             st.info('절약 칭찬해')
            #             st.balloons()

            #         else:
            #             st.warning('아껴쓰세요')

    else: 
        st.warning('사이드바에서 \'확인\' 또는 \'업로드\'를 클릭해주세요.')
except AttributeError:
    st.warning('사이드바에서 모델 설정과 CSV 파일 업로드를 완료해주세요.')
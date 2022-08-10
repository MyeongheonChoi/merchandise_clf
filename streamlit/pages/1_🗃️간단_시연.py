import streamlit as st
import pandas as pd
import plotly.express as px
import requests

BACKEND_INFERENCE_SINGLE = 'http://localhost:8000/inference_single'

st.set_page_config(layout = 'centered')
st.title("가맹점 분류")
st.header('체험판')

name = st.text_input('가게 이름을 넣어주세요.', '스타벅스 신촌오거리점')

possible = ['', 'KoBERT_26_CrossEntropy', 'KoELECTRA_26_CrossEntropy', 'RoBERTa_32_CrossEntropy']

### 사이드바
sidebar = st.sidebar
with sidebar:
    with st.container():
        st.subheader('Select a model')
        select_model = sidebar.selectbox('Select Model', ['KoBERT', 'KoELECTRA', 'RoBERTa'])
        maxlen = sidebar.radio('Select Max Sequence Length', [26, 32])
        lossfn = sidebar.radio('Select Loss Function', ['Cross Entropy', 'Focal Loss', 'Weighted Cross Entropy'])
    
    if maxlen == 32 and lossfn == 'Weighted Cross Entropy':
        st.warning('Wrong Hyperparameter!!!')
    elif f'{select_model}_{maxlen}_{"".join(lossfn.split())}' not in possible:
        st.warning('Wrong Hyperparameter!!!')
    else:
        if st.button('확인'):
            st.session_state['model_config'] = f'{select_model}_{maxlen}_{"".join(lossfn.split())}'
            st.success(f'Succesfully load model!')

### 본문
try:
    if st.session_state.model_config == f'{select_model}_{maxlen}_{"".join(lossfn.split())}':
        if maxlen == 32 and lossfn == 'Weighted Cross Entropy':
            st.warning('설정하신 Hyperparameter와 일치하는 모델이 없습니다!')
        elif f'{select_model}_{maxlen}_{"".join(lossfn.split())}' not in possible:
            st.warning('설정하신 Hyperparameter와 일치하는 모델이 없습니다!')
        else:
            try:
                input_data = {
                    'model' : st.session_state.model_config,
                    # 'model' : 'KoBERT_26_CrossEntropy',
                    'name' : name}
                    
                r = requests.post(
                    BACKEND_INFERENCE_SINGLE, json = input_data
                )

                result = pd.read_json(r.text, orient = 'records')
                cate = result['업종']
                prob = result['예측확률']

                st.info(f'[{name}]의 업종은 <{cate[0]}> 입니다.')

                df = result.set_index('업종')
                df['예측확률'] *= 100
                df.columns = ['예측확률 TOP5 (%)']

                st.subheader('예측확률')
                st.dataframe(df.T)

                fig = px.bar(df)
                fig.update_layout(showlegend = False)
                st.plotly_chart(fig)

            except ValueError:
                st.warning('가게 이름을 정확히 입력해주세요')

except AttributeError:
    st.warning('사이드바에서 모델 설정을 하거나 또는 가게 이름을 입력해주세요.')
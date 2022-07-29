import sys, os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
ROOT_DIR_MODEL = os.path.join(ROOT_DIR, 'model')
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR_MODEL)

import streamlit as st
import pandas as pd
import numpy as np
from model.inference_streamlit import load_model, inference
from model.modules.preprocess import preprocess_infer
import streamlit.components.v1 as components


st.set_page_config(layout = 'wide')

st.title("가맹점 분류")

st.header('체험판')
model = None
name = st.text_input('가게 이름을 넣어주세요.', '스타벅스 신촌오거리점')

@st.experimental_singleton(show_spinner = False)
def loading_model_sidebar(select_model, maxlen, lossfn):
    return load_model(select_model, maxlen, lossfn)

sidebar = st.sidebar
with sidebar:
    with st.container():
        st.subheader('Select a model')
        select_model = sidebar.selectbox('Select Model', ['KoBERT', 'KoELECTRA', 'RoBERTa'])
        maxlen = sidebar.radio('Select Max Sequence Length', [26, 32])
        lossfn = sidebar.radio('Select Loss Function', ['Cross Entropy', 'Focal Loss', 'Weighted Cross Entropy'])
    if st.button('확인'):
        st.session_state['model_config'] = f'{select_model}_{maxlen}_{"".join(lossfn.split())}'
        if maxlen == 32 and lossfn == 'Weighted Cross Entropy':
            st.warning('Wrong Hyperparameter!!!')
        else:
            with st.spinner(f'Loading Model... \n <{select_model}_{maxlen}_{"".join(lossfn.split())}>'):
                model, tokenizer = loading_model_sidebar(select_model, maxlen, lossfn)
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
            st.success(f'Succesfully load model!')
    
    try:
        if st.session_state.model_config:
            st.write(f'현재 사용 모델 : {st.session_state.model_config}')

    except AttributeError:
        pass
# st.write(f'{select_model}_{maxlen}_{"".join(lossfn.split())}')
# st.write(st.session_state.model_config)

try:
    if st.session_state.model and st.session_state.model_config == f'{select_model}_{maxlen}_{"".join(lossfn.split())}':
        if maxlen == 32 and lossfn == 'Weighted Cross Entropy':
            st.warning('설정하신 Hyperparameter와 일치하는 모델이 없습니다!')
        else:
            cate, prob = inference(preprocess_infer(name), maxlen, st.session_state.model, st.session_state.tokenizer, topk = True)
            st.info(f'[{name}]의 업종은 <{cate[0][0]}> 입니다.')
            np.set_printoptions(formatter = {'float_kind' : lambda x:'{0:0.2f}'.format(x)})
            st.write(pd.DataFrame(prob[0] * 100, index = cate[0], columns = ['예측 확률 (%, TOP 5)']))

except AttributeError:
    st.warning('사이드바에서 모델 설정을 하거나 또는 가게 이름을 입력해주세요.')\

# # %%
# import numpy as np
# np.set_printoptions(formatter = {'float_kind' : lambda x:'{0:0.3f}'.format(x)})
# np.round(np.array([1.23, 1.234235, 45345.34123]), 2)
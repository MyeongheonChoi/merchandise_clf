import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title = '가맹점 분류기', page_icon = '🙌')

wandb_url = {
    'KoBERT_26_CrossEntropy': 'https://wandb.ai/emseoyk/huggingface/reports/KoBERT_26_CrosssEntropy--VmlldzoyMzg3Mjg2?accessToken=1h36dl3ch6git1697g218ga7ikroxkw8a0ge2yksapr4rijmj1mh1rlywich1fon', 
    'KoBERT_26_FocalLoss': 'https://wandb.ai/emseoyk/huggingface/reports/KoBERT_26_FocalLoss--VmlldzoyMzg3MDQ4?accessToken=4y4muonzofygw2p8374xmq5s3nendv7mt5usn9usrh6odmo7pp0td3n0cvy51u25',
    'KoBERT_26_WeightedCrossEntropy': 'https://wandb.ai/emseoyk/huggingface/reports/KoBERT_26_WeightedCE--VmlldzoyMzg2NzQ0?accessToken=58jfl1xrcc5uvopqvdv6z3cgh9kcy3jzt44x9fo8g2memtijtemcdx0odhb758g0', 
    'KoBERT_32_CrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoBERT_32_CrossEntropyLoss--VmlldzoyMzg3MzM0?accessToken=3aljme5knf3uxxtuogbuo6lluc7ogm06smytsdogi34ki7xel8kwqdti7ycl0hel',
    'KoBERT_32_FocalLoss': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoBERT_32_FocalLoss--VmlldzoyMzg3MzI2?accessToken=vbtaan70d2d2i3dm1igcjt73nm8hy30zhg56vchmb8bwzv0bew9vt599641tspgl',
    
    'KoELECTRA_26_CrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoELECTRA_26_CrossEntropyLoss--VmlldzoyMzg3MzEw?accessToken=gyzlz4962r3x1n4ov80jwc1ug31rzhestjn912vloksvim0idp516e1kx2ksoofj', 
    'KoELECTRA_26_FocalLoss': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoELECTRA_26_FocalLoss--VmlldzoyMzg3MzAy?accessToken=9d0063le3669wzhmq6awjkyzoamjk2io9mmxwpl2yz6l2sbzvmik7r39r2ntl44v',
    'KoELECTRA_26_WeightedCrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoELECTRA_26_WeightedCE--VmlldzoyMzg3Mjg1?accessToken=u7rcgvfj4baruc0kntuowouo2tay9gjtolpgdr2dvm72p3hxv8e6m1zdlhht6gv9', 
    'KoELECTRA_32_CrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoELECTRA_32_CrossEntropyLoss--VmlldzoyMzg3Mjc4?accessToken=lewugy2zk7lv32dh9zbf3fkd8nk306hfwvk9xbfeimm83z5il3thodx9ss53a1yc',
    'KoELECTRA_32_FocalLoss': 'https://wandb.ai/myeongheonchoi/huggingface/reports/KoELECTRA_32_FocalLoss--VmlldzoyMzg3MjYy?accessToken=oeo3n5ajo5d7byvq8n0dyo4p8z5dtrqwi3vorrjp0s7be33s003c21qusfiqe624',
    
    'RoBERTa_26_CrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/RoBERTa_26_CrossEntropyLoss--VmlldzoyMzg3MjUx?accessToken=rqfxsj0dk49tv1cy6w2u5szrwrzwaeiv2584fg8lvbn2kdc3ohiyl1t4ndmq9wo9', 
    'RoBERTa_26_FocalLoss': 'https://wandb.ai/myeongheonchoi/huggingface/reports/RoBERTa_26_FocalLoss--VmlldzoyMzg3MjM5?accessToken=m9we00k04lo226vm0gskxc6xl64ikip62n4o7txwld4u8dqkq6vfu4z2xovpdph2',
    'RoBERTa_26_WeightedCrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/RoBERTa_26_WeightedCE--VmlldzoyMzg3MjI0?accessToken=c5zni9z39b4x4u8f7yu0y0dn3y5u3t2pitw17hzt4fpj44nm30o3fpi5ua6sh32p', 
    'RoBERTa_32_CrossEntropy': 'https://wandb.ai/myeongheonchoi/huggingface/reports/RoBERTa_32_CrossEntropyLoss--VmlldzoyMzg3MjAz?accessToken=8sjyu1sq3dkfbrfba5umrnyvtdwrjuo5flf982rrr1ngb7n9ok4i0ujqogt7uj35',
    'RoBERTa_32_FocalLoss': 'https://wandb.ai/myeongheonchoi/huggingface/reports/RoBERTa_32_FocalLoss--VmlldzoyMzg3MTc1?accessToken=3pux2ovnblnfu8e2njd4d4gamrrzmw21z39ou6ywmsap2j2w80sapudbm1ga5ckp'
}

st.header('가맹점 분류기')
# st.subheader('README')
# st.markdown('dkssuddkssud')
st.subheader('모델 성능')

select_model = st.selectbox('Select Model', ['KoBERT', 'KoELECTRA', 'RoBERTa'])
maxlen = st.radio('Select Max Sequence Length', [26, 32])
lossfn = st.radio('Select Loss Function', ['Cross Entropy', 'Focal Loss', 'Weighted Cross Entropy'])

components.iframe(
wandb_url[f'{select_model}_{maxlen}_{"".join(lossfn.split())}'], 
width = 1000, height = 600, scrolling = True)
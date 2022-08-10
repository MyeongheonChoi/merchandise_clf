import json, sys, os, datetime, torch
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR_MODEL = os.path.join(ROOT_DIR, 'model')
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR_MODEL)
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel

from model.inference_streamlit import load_model, inference
from model.modules.preprocess import preprocess_infer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(
    title='merch_clf',
    description='This is a demo of merch_clf',
)

# MODEL, TOKENIZER = load_model('KoBERT', 26, 'Cross Entropy', 'cpu')
MODEL1, TOKENIZER1 = load_model('KoBERT', 26, 'Cross Entropy', 'cpu')
MODEL2, TOKENIZER2 = load_model('KoELECTRA', 26, 'Cross Entropy', 'cpu')
MODEL3, TOKENIZER3 = load_model('RoBERTa', 32, 'Cross Entropy', 'cpu')

MODEL_DICT = {'KoBERT_26_CrossEntropy' : MODEL1,
              'KoELECTRA_26_CrossEntropy' : MODEL2,
              'RoBERTa_32_CrossEntropy' : MODEL3}
TOKEN_DICT = {'KoBERT_26_CrossEntropy' : TOKENIZER1,
              'KoELECTRA_26_CrossEntropy' : TOKENIZER2,
              'RoBERTa_32_CrossEntropy' : TOKENIZER3}


class InferenceData(BaseModel):
    model: str
    name: str

class InferenceData_(BaseModel):
    model_name: str
    file: str

def convert_to_dataframe(data):
    return pd.read_csv(data)

@app.post("/inference_single")
def inference_single(inp: InferenceData):
    model_name = inp.model
    merch_name = inp.name
    MODEL = MODEL_DICT[model_name]
    TOKENIZER = TOKEN_DICT[model_name]
    # if model_name == 'KoBERT_26_CrossEntropy':
    MODEL.model.to(device)
    cate, prob = inference(preprocess_infer(merch_name), 26, MODEL, TOKENIZER, device, topk = True)
    MODEL.model.cpu()
    result = pd.DataFrame({'업종' : cate.flatten(), '예측확률' : prob.flatten()})
    result_json = result.to_json(orient = 'records')
    result_response = JSONResponse(json.loads(result_json))
    return result_response

@app.post("/inference_csv")
def inference_csv(inp: InferenceData_):
    model_name = inp.model_name
    df_json = inp.file
    df = pd.read_json(df_json)
    
    df.reset_index(drop = True, inplace = True)
    
    # inference, 요일 붙이기
    MODEL = MODEL_DICT[model_name]
    TOKENIZER = TOKEN_DICT[model_name]
    # if model_name == 'KoBERT_26_CrossEntropy':
    MODEL.model.to(device)
    inferdf, _ = inference(preprocess_infer(df), 26, MODEL, TOKENIZER, device)
    MODEL.model.cpu()
    weekday_list = ['월','화','수','목','금','토','일']
    day = []
    for i in range(len(inferdf)):
        day.append(weekday_list[datetime.datetime.strptime(inferdf['거래일자'][i], '%Y-%m-%d').weekday()])
    inferdf['요일'] = day

    inferdf_json = inferdf.to_json(orient = 'records')
    inferdf_response = JSONResponse(json.loads(inferdf_json))

    return inferdf_response
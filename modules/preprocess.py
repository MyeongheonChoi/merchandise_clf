from tqdm import tqdm
import pandas as pd
import re

'''
INPUT
-Train : df 형식은 pd.DataFrame / column은 업체명, 업종
-Inference : df 형식은 pd.DataFrame / column은 업체명이 있어야 함
'''

def duplicate_count(df):
    check = {}
    df_drop = df.drop_duplicates(keep = 'first')
    df_count = df_drop['업체명'].value_counts()
    for name in tqdm(df_count[df_count >= 2].index):
        cate = df[df['업체명'] == name]['업종'].value_counts()
        check[name] = cate.index[0]
    return check

def drop_duplicate_by_count(df, check):
    df_drop = df.copy()
    for key, val in tqdm(check.items()):
        df_drop.loc[df_drop['업체명'] == key, '업종'] = val
    df_drop = df_drop.drop_duplicates(keep = 'first').reset_index(drop = True)
    return df_drop

def regularize(df, cased = False):  ### cased = True -> 대소문자 구별 / cased = False -> 영어는 모두 소문자
    if type(df) == pd.DataFrame:
        regularized = []
        for word in tqdm(df['업체명']):
            regularized.append(re.sub("[^가-힣a-zA-Z0-9]", " ", word))

        df['업체명_r'] = regularized
        df['업체명_r'] = df['업체명_r'].str.strip()

        for i in range(10):
            df['업체명_r'] = df['업체명_r'].str.replace('  ',' ')
            df['업체명_r'] = df['업체명_r'].str.replace('   ',' ')
        
        idx = []
        for c in range(50):
            idx.append(list(df[df['업체명_r'] == ' '* c].index))
        idx = sum(idx, [])
        df_ = df.drop(idx).reset_index(drop = True)

        if not cased:
            df_['업체명_r'] = df_['업체명_r'].str.lower()

        return df_
    
    elif type(df) == str:
        word = re.sub("[^가-힣a-zA-Z0-9]", " ", df)
        word = word.strip()
        if word == '':
            raise Exception('Cannot Preprocessing!!!')
        for i in range(10):
            word = word.replace('  ',' ')
            word = word.replace('   ',' ')
        return word
        

def preprocess_train(df):
    df.dropna(how = 'any', inplace = True)
    df.reset_index(drop = True, inplace = True)
    check = duplicate_count(df)
    df_ = df.drop_duplicates(keep = 'fisrt').reset_index(drop = True)
    df_drop = drop_duplicate_by_count(df_, check)
    df_regularized = regularize(df_drop)
    df_regularized = df_regularized[['업체명','업체명_r','업종']]
    df_regularized = df_regularized.drop_duplicates(keep = 'first')
    df_regularized_ = df_regularized.drop_duplicates(subset = ['업체명_r'], keep = False).reset_index(drop = True)
    return df_regularized_

def preprocess_infer(df):
    if type(df) == pd.DataFrame:
        df.dropna(how = 'any', inplace = True)
        return regularize(df)
    elif type(df) == str:
        return regularize(df)

'''
OUTPUT
-Train : 형식은 pd.DataFrame / column은 업체명, 업체명_r, 업종
-Inference : 형식은 pd.DataFrame / column은 기존 column + 업체명_r
'''

if __name__ == '__main__':
# preprocess for training
    # df_ = preprocess_train(df)
    # df_.to_csv(path)
# preprocess for inference
    # import pandas as pd
    # a = preprocess_infer(pd.read_csv('/VOLUME/py_model/data/거래내역.csv', index_col = 0))
    pass

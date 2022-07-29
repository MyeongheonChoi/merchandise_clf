import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

from utils import save_pickle

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)

def load_data(path):
    data = pd.read_csv(path)
    data = data[['업체명_r', '업종']]
    data.columns = ['업체명', '업종']
    return data

def encode(label):
    encoder = LabelEncoder()
    encoder.fit(label)
    new_label = encoder.transform(label)    
    return new_label

def character_cut(data):
    data_list = []
    word_occ = defaultdict(int)
    for idx, voc in tqdm(data.iterrows()):
        text = voc['업체명']
        text = list(text)
        target = voc['업종']
        data_list.append((text, target))
        for word in text:
            word_occ[word] += 1
    return data_list,word_occ

def preprocess(path):
    data = load_data(path)
    data['업종'] = encode(data['업종'])
    data_dat, data_vocab = character_cut(data)
    return data_dat, data_vocab

# if __name__ == '__main__':
#     data_path = os.path.join(ROOT_PROJECT_DIR, 'data/uncased_train.csv')
#     data_dat, data_vocab = preprocess(data_path)
#     test_path = os.path.join(ROOT_PROJECT_DIR, 'data/uncased_test.csv')
#     test_dat, _ = preprocess(test_path)
#     save_pickle(os.path.join(ROOT_PROJECT_DIR, 'data/uncased_train_data.pkl'), data_dat)
#     save_pickle(os.path.join(ROOT_PROJECT_DIR, 'data/uncased_train_vocab.pkl'), data_vocab)
#     save_pickle(os.path.join(ROOT_PROJECT_DIR, 'data/uncased_test_data.pkl'), test_dat)
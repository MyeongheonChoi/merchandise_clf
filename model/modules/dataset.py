import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode(label):
    encoder = LabelEncoder()
    encoder.fit(label)
    new_label = encoder.transform(label)    
    return new_label

class CustomDataset(Dataset):

    def __init__(self, data: pd.DataFrame, pretrained_link: str, max_seq_len: int):
        data = data[['업체명_r', '업종']] 
        data.columns = ['업체명', '업종']
        tokenizer = AutoTokenizer.from_pretrained(pretrained_link)
        labels = encode(data['업종'])
        self.item = list()
        for idx, name in enumerate(tqdm(data['업체명'])):
            item = {key: torch.tensor(val).to(device) for key, val in tokenizer(name, 
                                                                                truncation = True, 
                                                                                padding = 'max_length', 
                                                                                max_length = max_seq_len).items()}

            item['labels'] = torch.tensor(labels[idx]).to(device)
            self.item.append(item)

    def __getitem__(self, idx):
        return self.item[idx]

    def __len__(self):
        return len(self.item)

if __name__ == '__main__':
    pass
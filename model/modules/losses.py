import torch
from torch.nn import CrossEntropyLoss
import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = os.path.join(ROOT_PROJECT_DIR, 'data/uncased_test.csv')
data = pd.read_csv(path, index_col = 0)

def get_loss_fn(loss_function: str):
    if loss_function == 'CrossEntropyLoss':
        loss_fn = CrossEntropyLoss()

    elif loss_function == 'FocalLoss':
        loss_fn = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', 
                                    model = 'focal_loss',
                                    gamma = 2,
                                    reduction = 'mean',
                                    device = device,
                                    dtype = torch.float32,
                                    force_reload = False)
                                    
    elif loss_function == 'WeightedCE':
        loss_fn = CrossEntropyLoss(weight = torch.FloatTensor((1 / data['업종'].value_counts().sort_index()) / (1 / data['업종'].value_counts().sort_index()).sum()).to(device))
    else:
        loss_fn = CrossEntropyLoss()
        print('\n!!!Loss function is Automatically determined (CrossEntropyLoss)!!!\n')
        
    return loss_fn

if __name__ == '__main__':
    pass
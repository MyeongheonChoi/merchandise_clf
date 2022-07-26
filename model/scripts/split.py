
import os
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from modules.utils import load_yaml, save_yaml
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
print(PROJECT_DIR)

script_name = os.path.basename(__file__)
config_name = script_name.replace('.py', '_config.yaml')
config_path = os.path.join(PROJECT_DIR, 'config', config_name)
config = load_yaml(config_path)

DST_DATA_DIR = os.path.join(PROJECT_DIR, 'data', config['DATASET']['dst_dataset'])

def split(data):
    encoder = LabelEncoder()
    label = data['업종']
    encoder.fit(label)
    new_label = encoder.transform(label) 
    data['업종'] = new_label

    train, valid = train_test_split(
                                    data, 
                                    test_size = config['SPLIT']['valid'], 
                                    stratify = new_label, 
                                    random_state = config['SPLIT']['seed'])

    return train.reset_index(drop = True), valid.reset_index(drop = True)


import os, torch, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

from modules.metrics import compute_metrics
from modules.trainer import CustomTrainer
from modules.utils import load_yaml
from modules.preprocess import preprocess_infer


# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'inference_config.yaml')
config = load_yaml(config_path)

#DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(select_model, maxlen, loss):

    name = f'{select_model}_{maxlen}_{"".join(loss.split())}'
    output_dir      = os.path.join(PROJECT_DIR, 'results', config.MODEL.RESULTS[name])
    pretrained_link = config.MODEL.pretrained_link[select_model]
    num_of_classes  = config.MODEL.num_of_classes
    checkpoint_path = output_dir
    
    print('=' * 50)
    print('Get Model & Tokenizer')
    print('=' * 50)

    test_args = TrainingArguments(output_dir=output_dir,
                                    dataloader_pin_memory = False,
                                    do_predict = True
                                    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels = num_of_classes).to(device)

    trainer = CustomTrainer(model = model,
                            args = test_args,
                            compute_metrics = compute_metrics
                            )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_link)

    return trainer, tokenizer

def inference(data, max_seq_len, trainer, tokenizer):

    print('=' * 50)
    print('Tokenizing...')
    print('=' * 50)

    items = list()

    if type(data) == str:
        item = {key: torch.tensor(val).to(device) for key, val in tokenizer(data, 
                                                                            truncation = True, 
                                                                            padding = 'max_length', 
                                                                            max_length = max_seq_len).items()}
        items.append(item)

    elif type(data) == pd.DataFrame:
        for name in tqdm(data['업체명_r']):
            item = {key: torch.tensor(val).to(device) for key, val in tokenizer(name, 
                                                                                truncation = True, 
                                                                                padding = 'max_length', 
                                                                                max_length = max_seq_len).items()}
            items.append(item)

    print('=' * 50)
    print('Predicting...')
    print('=' * 50)

    test_results = trainer.predict(items)
    label_ids = np.argmax(test_results[0], axis = 1)

    if type(data) == str:
        return config.LABELING[label_ids[0]]

    elif type(data) == pd.DataFrame:
        data['업종'] = label_ids
        data['업종'] = data['업종'].replace(config.LABELING.keys(), config.LABELING.values())
        return data
    
if __name__ == '__main__':
    pass

### 문자를 넣으면 업종을 return
### DataFrame을 넣으면 업종을 column에 추가시킨 DataFrame을 return

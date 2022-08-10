import os, math, torch, wandb, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from datetime import datetime, timezone, timedelta
import pandas as pd

from transformers import TrainingArguments, AutoModelForSequenceClassification

from modules.trainer import CustomTrainer
from modules.dataset import CustomDataset
from modules.optimizer import get_optimizer
from modules.metrics import compute_metrics
from modules.utils import load_yaml
from modules.split import split

# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yaml')
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder directory
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data', config.DIRECTORY.dataset)

#DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__': 

    output_dir      = OUTPUT_DIR
    pretrained_link = config.MODEL.pretrained_link[config.MODEL.model_name]
    num_of_classes  = config.MODEL.num_of_classes
    batch_size      = config.TRAIN.batch_size
    num_of_epochs   = config.TRAIN.num_of_epochs
    learning_rate   = config.TRAIN.learning_rate.pretrained
    max_grad_norm   = config.TRAIN.max_grad_norm
    warmup_ratio    = config.TRAIN.warmup_ratio
    run_name        = config.MODEL.model_name + train_serial + '_nonumber'
    max_seq_len     = config.TRAIN.max_seq_len
    
    data = pd.read_csv(DATA_DIR + '_train.csv', index_col = 0)
    train_df, valid_df = split(data)

    train_len = len(train_df)
    loader_len = math.ceil(train_len / batch_size)
    t_total = loader_len * num_of_epochs
    warmup_step = int(t_total * warmup_ratio)

    training_args = TrainingArguments(output_dir = output_dir,                 
                                      per_device_train_batch_size = batch_size,
                                      per_device_eval_batch_size = batch_size, 
                                      num_train_epochs = num_of_epochs,        
                                      learning_rate = learning_rate,           
                                      max_grad_norm = max_grad_norm,           
                                      lr_scheduler_type = 'cosine',
                                      warmup_ratio = warmup_ratio,
                                      evaluation_strategy = 'steps',           
                                      report_to = 'wandb',
                                      run_name = run_name, 
                                      dataloader_pin_memory = False,
                                      logging_steps = 500, 
                                      load_best_model_at_end = True, 
                                      save_steps = 500
                                      )

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_link, num_labels = num_of_classes).to(device)
    optimizer = get_optimizer('AdamW', model, learning_rate, warmup_step, t_total)

    train_dataset = CustomDataset(train_df, pretrained_link, max_seq_len)
    valid_dataset = CustomDataset(valid_df, pretrained_link, max_seq_len)

    wandb.login() 
    
    trainer = CustomTrainer(model = model,
                            args = training_args,
                            optimizers = optimizer,
                            train_dataset = train_dataset,
                            eval_dataset = valid_dataset,
                            compute_metrics = compute_metrics
                            )
    
    # Train
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, f'{config.MODEL.model_name}_{config.TRAIN.max_seq_len}_{config.TRAIN.loss}_{config.DIRECTORY.dataset}'))

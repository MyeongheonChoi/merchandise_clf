import os
from transformers import Trainer

from modules.losses import get_loss_fn
from modules.utils import load_yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)

config_path = os.path.join(ROOT_PROJECT_DIR, 'config/train_config.yaml')
config = load_yaml(config_path)

loss_function = get_loss_fn(config.TRAIN.loss)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        criterion = loss_function
        loss = criterion(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return(loss, outputs) if return_outputs else loss

if __name__ == '__main__':
    pass
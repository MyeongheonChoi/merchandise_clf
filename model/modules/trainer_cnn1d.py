from tqdm import tqdm
import torch

class CustomTrainer():

    def __init__(self,
                model,
                optimizer,
                loss_function,
                metrics,
                device,
                logger,
                amp,
                interval = 100):

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.device = device

        self.train_losses = []
        

    def train(self, dataloader):

        self.model.train()
        for idx, (texts, inputs, targets) in enumerate(tqdm(dataloader)):

            self.optimizer.zero_grad()
            output = self.model(inputs)

            max_vals, max_indicies = torch.max(output, 1)
            train_acc += (max_indicies == targets).sum().data.cpu().numpy()
            loss = self.loss_function(output, targets)

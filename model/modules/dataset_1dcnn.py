import torch
from torch.utils.data import Dataset

class CustomDataset_1dcnn(Dataset):

    def __init__(self, device, data):
        self.data = data
        self.texts = [i[0] for i in data]
        self.inputs = torch.LongTensor([i[1] for i in data]).to(device)
        self.targets = torch.LongTensor([i[2] for i in data]).to(device)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.texts[idx], self.inputs[idx], self.targets[idx])

if __name__ == '__main__':
    pass
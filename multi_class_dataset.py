import torch

class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_label, y_sub):
        self.x = x
        self.y_label = y_label
        self.y_sub = y_sub

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_label[idx], self.y_sub[idx]
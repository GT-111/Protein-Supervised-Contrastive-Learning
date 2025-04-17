import torch.nn as nn
import torch.nn.functional as F

class SupConEncoder(nn.Module):
    def __init__(self, input_dim=32, embed_dim=64):
        super(SupConEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.normalize(x, dim=1)
        return x


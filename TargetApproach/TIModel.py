from torch import nn


class TransimpedanceModel(nn.Module):
    def __init__(self):
        super(TransimpedanceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.network(x)

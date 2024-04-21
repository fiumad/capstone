from torch import nn


class TransimpedanceModel(nn.Module):
    def __init__(self):
        super(TransimpedanceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 3, bias=True),
        )

    def forward(self, x):
        return self.network(x)

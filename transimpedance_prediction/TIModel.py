from torch import nn

class TransimpedanceModel(nn.Module):
    def __init__(self):
        super(TransimpedanceModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)

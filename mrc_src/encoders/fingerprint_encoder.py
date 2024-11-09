from torch import nn

class FingerprintSimpleFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Linear(1024, 512),
            )
    def forward(self, input):
        return self.fc(input)
from dataclasses import dataclass
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class FeedForwardConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int

class FingerprintSimpleFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.fc = nn.Sequential(
                # nn.BatchNorm1d(config.hidden_dim),
                nn.Linear(config.input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.output_dim),
            )
    def forward(self, x):
        x = F.normalize(x, p=2, dim=-1)
        return self.fc(x)
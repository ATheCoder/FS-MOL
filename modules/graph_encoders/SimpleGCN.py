from dataclasses import dataclass
from torch import nn


@dataclass
class SimpleGCNConfig:
    n_layers: int

class SimpleGCN(nn.Module):
    def __init__(self, config: SimpleGCNConfig) -> None:
        super().__init__(*args, **kwargs)
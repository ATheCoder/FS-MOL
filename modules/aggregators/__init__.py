import torch
from torch import nn


class SumNormAggregator(nn.Module):
    def __init__(self, in_dim) -> None:
        super(SumNormAggregator, self).__init__()

        self.norm = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        mol_repr = torch.stack(x)
        mol_repr = torch.sum(mol_repr, dim=0)
        mol_repr = self.batch_norm(mol_repr)

        return mol_repr


class LastAggregator(nn.Module):
    def __init__(self) -> None:
        super(LastAggregator, self).__init__()

    def forward(self, x):
        return x[-1]


class ConcatAggregator(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, n_layers=None) -> None:
        super(ConcatAggregator, self).__init__()

        self.projector = None

        args = [in_dim, out_dim, n_layers]

        if all(arg is not None for arg in args):
            self.projector = nn.Linear(in_dim, out_dim)
        elif any(arg is not None for arg in args):
            raise ValueError(
                "All three arguments must be provided for Linear layer or none of them."
            )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        if self.projector == None:
            return x

        return self.projector(x)

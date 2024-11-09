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


class LastNormAggregator(nn.Module):
    def __init__(self, dim) -> None:
        super(LastNormAggregator, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(self.norm(x[-1]))


class ConcatAggregator(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, n_layers=None) -> None:
        super(ConcatAggregator, self).__init__()

        self.projector = None

        args = [in_dim, out_dim, n_layers]

        if all(arg is not None for arg in args):
            self.projector = nn.Linear(in_dim * n_layers, out_dim)
        elif any(arg is not None for arg in args):
            raise ValueError(
                "All three arguments must be provided for Linear layer or none of them."
            )

    def forward(self, x):
        x = torch.cat(x, dim=1)
        if self.projector == None:
            return x

        return self.projector(x)


class ConcatBatchNormLin(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim):
        super(ConcatBatchNormLin, self).__init__()
        self.norm = nn.LayerNorm(n_layers * in_dim)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.norm(x)

        return x


def make_aggregator(aggr_name: str, **kwargs):
    return molecule_aggr_map[aggr_name](**kwargs)


molecule_aggr_map = dict(
    sum_norm=SumNormAggregator,
    concat=ConcatAggregator,
    last=LastAggregator,
    concat_norm_lin=ConcatBatchNormLin,
    last_norm=LastNormAggregator,
)

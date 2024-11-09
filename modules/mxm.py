import torch
from torch import nn
from torch.nn import Sequential


class LinearXavier(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(LinearXavier, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity="linear")

        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class Res(nn.Module):
    def __init__(self, dim, dropout):
        super(Res, self).__init__()

        self.mlp = MLP([dim, dim, dim], dropout)
        # self.norm = GraphNorm(dim)
        # self.norm = BatchNorm(dim)

    def forward(self, m, batch):
        m1 = self.mlp(m)
        m_out = m1 + m
        return m_out


class Xavier_SiLU_Residual(nn.Module):
    def __init__(self, dim, dropout):
        super(Xavier_SiLU_Residual, self).__init__()

        self.mlp = Xavier_SiLU_MLP([dim, dim, dim], dropout)
        # self.norm = GraphNorm(dim)
        # self.norm = BatchNorm(dim)

    def forward(self, m, batch):
        m1 = self.mlp(m)
        m_out = m1 + m
        return m_out


def MLP(channels, dropout=0.0):
    return Sequential(
        *[
            # Sequential(nn.Linear(channels[i - 1], channels[i]), nn.LeakyReLU(), nn.Dropout(dropout))
            Sequential(
                LinearXavier(channels[i - 1], channels[i]), nn.SELU(), nn.AlphaDropout(dropout)
            )
            for i in range(1, len(channels))
        ]
    )


class LinearXavier_2(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearXavier_2, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.linear.weight, 3 / 4)

        if self.linear.bias is not None:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def Xavier_SiLU_MLP(channels, dropout=0.0):
    return Sequential(
        *[
            Sequential(LinearXavier_2(channels[i - 1], channels[i]), nn.SiLU())
            for i in range(1, len(channels))
        ]
    )

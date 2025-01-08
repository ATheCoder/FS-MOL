import math
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import (
    SetTransformerAggregation,
)
from modules.graph_modules.global_mp import ResidualGIN
from torch_geometric.utils import remove_self_loops, sort_edge_index, to_undirected


@dataclass
class GINEncoderConfig:
    dim: int
    n_layer: int
    cutoff: float
    dropout: float = 0.0


class GINEncoder(nn.Module):
    def __init__(self, config: GINEncoderConfig):
        super(GINEncoder, self).__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff

        self.embeddings = nn.Embedding(16, self.dim)

        self.global_layers = torch.nn.ModuleList()
        for i in range(self.n_layer):
            self.global_layers.append(ResidualGIN(self.dim, config.dropout))

        self.set_transformer = SetTransformerAggregation(self.dim, heads=4, layer_norm=False, dropout=config.dropout)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        nn.init.uniform_(self.embeddings.weight, -stdv, stdv)

    def forward(self, data):
        x = data.x
        edge_index_g = to_undirected(data.edge_index.long())
        # Initialize node embeddings
        h = self.embeddings(x.long())

        edge_index_g, _ = remove_self_loops(edge_index_g)
        edge_index_g = sort_edge_index(edge_index_g, sort_by_row=False)


        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, edge_index_g, batch=data.batch)
        
        return self.set_transformer(h, data.batch)


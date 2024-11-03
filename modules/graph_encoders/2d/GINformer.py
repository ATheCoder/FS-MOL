import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.nn import (
    SetTransformerAggregation,
)

from modules.aggregators import make_aggregator
from modules.graph_modules.global_mp import ResidualGIN
from torch_geometric.utils import remove_self_loops, sort_edge_index, to_undirected


@dataclass
class GINformerConfig:
    dim: int
    n_layer: int
    cutoff: float
    mol_aggr: str | None
    mol_aggr_config: dict = field(default_factory=dict)
    dropout: float = 0.0
    attn_dropout: float = 0.0
    gin_dropout: float = 0.0


class GINformer(nn.Module):
    def __init__(self, config: GINformerConfig):
        super(GINformer, self).__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff

        self.embeddings = nn.Embedding(16, self.dim)

        self.global_layers = torch.nn.ModuleList()
        for _ in range(self.n_layer):
            self.global_layers.append(ResidualGIN(self.dim, config.dropout))
        
        self.aggregators = torch.nn.ModuleList()
        for _ in range(self.n_layer):
            self.aggregators.append(SetTransformerAggregation(self.dim, heads=2, layer_norm=False, dropout=config.attn_dropout))

        if config.mol_aggr != None:
            self.mol_aggr = make_aggregator(config.mol_aggr, **config.mol_aggr_config)

        self.set_transformer = SetTransformerAggregation(self.dim, heads=4, layer_norm=False, dropout=config.attn_dropout)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        nn.init.uniform_(self.embeddings.weight, -stdv, stdv)

    def forward(self, data):
        x = data.x
        batch = data.batch
        edge_index = to_undirected(data.edge_index.long())

        # Initialize node embeddings
        h = self.embeddings(x.long())
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = sort_edge_index(edge_index, sort_by_row=False)

        # Perform the message passing schemes

        layer_reprs = []
        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, edge_index, batch=batch)
            layer_reprs.append(self.aggregators[layer](h, batch))
            

        stacked_reprs = torch.stack(layer_reprs)
        permuted_reprs = stacked_reprs.permute(1, 0, 2)
        reshaped_reprs = permuted_reprs.reshape(-1, self.dim)

        batch_size = layer_reprs[0].shape[0]
        layer_batch = torch.arange(0, batch_size, device=reshaped_reprs.device).repeat_interleave(self.n_layer)
        
        mol_reprs = self.set_transformer(reshaped_reprs, layer_batch)

        return mol_reprs

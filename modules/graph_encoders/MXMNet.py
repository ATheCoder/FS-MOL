import math
from dataclasses import dataclass, field

import torch
from torch import nn
from torch_geometric.nn import (
    SetTransformerAggregation,
    SumAggregation,
)
from torch_geometric.nn.norm import GraphNorm

from modules.aggregators import make_aggregator
from modules.graph_modules.global_mp import ResidualGIN
from modules.mxm import MLP
from modules.node_embeders.basis_layers import BesselBasisLayer
from torch_geometric.utils import remove_self_loops, sort_edge_index, to_undirected


@dataclass
class MXMNetConfig:
    dim: int
    n_layer: int
    cutoff: float
    mol_aggr: str | None
    mol_aggr_config: dict = field(default_factory=dict)
    dropout: float = 0.0


class MXMNet(nn.Module):
    def __init__(self, config: MXMNetConfig, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(MXMNet, self).__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff
        self.norm = nn.BatchNorm1d(self.dim * self.n_layer)

        self.embeddings = nn.Embedding(16, self.dim)

        self.norm = nn.LayerNorm(self.dim)

        self.rbf_g = BesselBasisLayer(16, config.cutoff, envelope_exponent)

        self.norm_rbf_g = nn.LayerNorm(128)

        self.rbf_g_mlp = MLP([16, self.dim], config.dropout)
        self.out_lin = MLP([self.dim, self.dim])

        self.global_layers = torch.nn.ModuleList()
        for i in range(self.n_layer):
            # self.global_layers.append(Global_MP_Attn(self.dim, self.dim, self.dim, dropout, layer_no=i))
            self.global_layers.append(ResidualGIN(self.dim, config.dropout))
        
        self.aggregators = torch.nn.ModuleList()
        for i in range(self.n_layer):
            # self.global_layers.append(Global_MP_Attn(self.dim, self.dim, self.dim, dropout, layer_no=i))
            self.aggregators.append(SetTransformerAggregation(self.dim, heads=2, layer_norm=False, dropout=config.dropout))

        self.g_norm = GraphNorm(self.dim)

        self.sum_module = SumAggregation()

        self.batch_norm = nn.BatchNorm1d(self.dim)

        if config.mol_aggr != None:
            self.mol_aggr = make_aggregator(config.mol_aggr, **config.mol_aggr_config)
        self.distance_batch_norm = nn.BatchNorm1d(self.dim)

        self.set_transformer = SetTransformerAggregation(self.dim, heads=4, layer_norm=False, dropout=config.dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.dim, self.dim, bias=False),
        )

        self.layer_norm = nn.LayerNorm(self.dim)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        nn.init.uniform_(self.embeddings.weight, -stdv, stdv)

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        edge_index_g = to_undirected(data.edge_index.long())
        # Initialize node embeddings
        h = self.embeddings(x.long())
        # if wandb.run is not None:
        #     wandb.log({"mean_node_count": batch.bincount().float().mean()})
        # wandb.log({'avg_graph_size': batch.bincount().mean()})

        # Get the edges pairwise distances in the global layer
        # row, col = radius(pos, pos, self.cutoff, batch, batch, max_num_neighbors=500)
        # edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        edge_index_g = sort_edge_index(edge_index_g, sort_by_row=False)
        # j_g, i_g = edge_index_g
        # dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        # dist_g = dist_g
        # rbf_g = self.rbf_g(dist_g)
        # rbf_g = self.rbf_g_mlp(rbf_g)
        # rbf_g = self.distance_batch_norm(rbf_g)

        # Perform the message passing schemes

        mol_reprs = []
        for layer in range(self.n_layer):
            # wandb.log({"h_min": h.min(), "h_max": h.max()})
            h = self.global_layers[layer](h, edge_index_g, batch=data.batch)
            mol_reprs.append(self.aggregators[layer](h, data.batch))
            
            # We can use a set transformer against the results of each layer.

            # mol_reprs.append(graph_repr)

        # mol_reprs = torch.cat(mol_reprs, dim=-1)
        stacked_reprs = torch.stack(mol_reprs)
        permuted_reprs = stacked_reprs.permute(1, 0, 2)
        reshaped_reprs = permuted_reprs.reshape(-1, self.dim)
        
        q = self.set_transformer(reshaped_reprs, torch.arange(0, mol_reprs[0].shape[0], device='cuda').repeat_interleave(self.n_layer))
        # mol_reprs_2 = self.mlp(mol_reprs)

        return q
        # return self.set_transformer(h, data.batch)
        # return self.mol_aggr(mol_reprs)

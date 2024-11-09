import math

import torch
import torch.nn as nn
from torch_geometric.nn import SumAggregation, radius
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
from modules.graph_modules.global_mp import Generic_Global_MP
from modules.mxm import Xavier_SiLU_MLP
from torch_geometric.nn import SetTransformerAggregation
from modules.node_embeders.basis_layers import BesselBasisLayer


class GenericMXMNet_NoLocalInfo_Config(object):
    def __init__(self, dim, n_layer, cutoff, attn_dropout):
        self.dim = dim
        self.n_layer = n_layer
        self.cutoff = cutoff
        self.attn_dropout = attn_dropout


class GenericMXMNet_NoLocalInfo(nn.Module):
    """
    This is the initial MXMNet with the Local Information layer removed.
    """

    def __init__(
        self,
        config: GenericMXMNet_NoLocalInfo_Config,
        envelope_exponent=5,
        dropout=0.0,
    ):
        super(GenericMXMNet_NoLocalInfo, self).__init__()

        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff = config.cutoff
        self.norm = nn.BatchNorm1d(self.dim * self.n_layer)

        self.embeddings = nn.Embedding(16, self.dim)

        self.rbf_g = BesselBasisLayer(16, config.cutoff, envelope_exponent)

        self.rbf_g_mlp = Xavier_SiLU_MLP([16, self.dim], dropout)
        self.out_lin = Xavier_SiLU_MLP([self.dim * self.n_layer, self.dim])

        self.global_layers = torch.nn.ModuleList()
        for i in range(self.n_layer):
            self.global_layers.append(
                Generic_Global_MP(
                    self.dim, self.dim, self.dim, dropout, attn_dropout=config.attn_dropout
                )
            )

        self.sum_module = SumAggregation()

        # self.set_transformer = SetTransformerAggregation(
        #     self.dim,
        #     heads=4,
        #     num_encoder_blocks=self.n_layer // 2,
        #     num_decoder_blocks=self.n_layer // 2,
        # )

        self.set_transformer = SetTransformerAggregation(
            self.dim * self.n_layer,
            heads=8,
            num_encoder_blocks=1,
            num_decoder_blocks=1,
        )
        self.dist_norm_1 = nn.BatchNorm1d(1)
        self.dist_norm_2 = nn.BatchNorm1d(16)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        nn.init.uniform_(self.embeddings.weight, -stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))

        # Compute the node indices for two-hop angles
        adj_t_row = adj_t[row]  # type: ignore
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k
        idx_i_1, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji_1 = adj_t_row.storage.row()[mask]

        # Compute the node indices for one-hop angles
        adj_t_col = adj_t[col]  # type: ignore

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_2 = row.repeat_interleave(num_pairs)
        idx_j1 = col.repeat_interleave(num_pairs)
        idx_j2 = adj_t_col.storage.col()

        idx_ji_2 = adj_t_col.storage.row()
        idx_jj = adj_t_col.storage.value()

        return idx_i_1, idx_j, idx_k, idx_kj, idx_ji_1, idx_i_2, idx_j1, idx_j2, idx_jj, idx_ji_2

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch
        # Initialize node embeddings
        h = self.embeddings(x.long())

        # Get the edges pairwise distances in the global layer
        row, col = radius(pos, pos, self.cutoff, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        dist_g = dist_g + torch.tensor(1e-8)  # This is added for stability and removing 0s

        # Get the RBF and SBF embeddings
        rbf_g = self.rbf_g(dist_g)
        rbf_g = self.dist_norm_2(rbf_g)
        rbf_g = self.rbf_g_mlp(rbf_g)

        mol_reprs = []
        for layer in range(self.n_layer):
            h = self.global_layers[layer](h, rbf_g, edge_index_g, batch=data.batch)

            mol_reprs.append(h)

        # batch_size = mol_reprs[0].shape[0]
        # device = mol_reprs[0].device

        # flatted_per_layer_mol_reprs = torch.stack(mol_reprs, dim=1).reshape(-1, self.dim)
        # batch_layer_index = torch.repeat_interleave(
        #     torch.arange(batch_size, device=device), self.n_layer
        # )

        # mol_repr = self.set_transformer(flatted_per_layer_mol_reprs, batch_layer_index)

        mol_repr = torch.cat(mol_reprs, dim=-1)

        mol_repr = self.set_transformer(mol_repr, batch)
        return self.out_lin(mol_repr)

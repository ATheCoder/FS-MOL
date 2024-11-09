from typing import Optional, Tuple, Callable
from torch import Tensor
import torch
from torch_geometric.nn.models import DimeNetPlusPlus
from torch_geometric.nn.aggr import SetTransformerAggregation
from torch_geometric.typing import SparseTensor
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter
from torch.nn import Linear
from torch_geometric.nn.inits import glorot_orthogonal


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]  # type: ignore
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji


class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)  # type: ignore
        self.lin.weight.data.fill_(0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce="sum")
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)

        # hidden_channels=128,
        # out_channels=1,
        # num_blocks=4,
        # int_emb_size=64,
        # basis_emb_size=8,
        # out_emb_channels=256,
        # num_spherical=7,
        # num_radial=6,
        # cutoff=5.0,
        # max_num_neighbors=32,
        # envelope_exponent=5,
        # num_before_skip=1,
        # num_after_skip=2,
        # num_output_layers=3,


class AdaptedDimeNetPP(DimeNetPlusPlus):
    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
    ):
        super().__init__(
            hidden_channels,
            hidden_channels,
            num_blocks,
            int_emb_size,
            basis_emb_size,
            out_emb_channels,
            num_spherical,
            num_radial,
            cutoff,
            max_num_neighbors,
            envelope_exponent,
            num_before_skip,
            num_after_skip,
            2,
            "swish",
        )

        self.set_transformer = SetTransformerAggregation(
            hidden_channels, heads=4, layer_norm=True, num_encoder_blocks=4, num_decoder_blocks=4
        )

    def forward(self, data) -> Tensor:
        z = data.x
        pos = data.pos
        batch = data.batch

        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):  # type: ignore
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=pos.size(0))
        return self.set_transformer(P, batch)

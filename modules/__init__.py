from typing import Callable, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    EmbeddingBlock,
    InteractionBlock,
    SphericalBasisLayer,
)
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import scatter
from .similarity_modules import *


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
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


class DimeNet(torch.nn.Module):
    url = "https://github.com/klicperajo/dimenet/raw/master/pretrained/" "dimenet"

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
    ):
        super().__init__()

        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        act = activation_resolver(act)

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels,
                    num_bilinear,
                    num_spherical,
                    num_radial,
                    num_before_skip,
                    num_after_skip,
                    act,
                )
                for _ in range(num_blocks)
            ]
        )

        self.batch_norm = torch.nn.BatchNorm1d(hidden_channels)
        self.out_lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

    def forward(
        self,
        g,
    ) -> Tensor:
        batch = g.batch
        z = g.x
        pos = g.pos
        r"""
        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
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

        mol_embeddings = []
        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            h = scatter(x, i, dim=0, dim_size=pos.size(0), reduce="sum")
            h = scatter(h, batch, dim=0, reduce="sum")
            mol_embeddings.append(h)

        mol_embeddings = torch.stack(mol_embeddings)
        mol_repr = torch.sum(mol_embeddings, dim=0)
        mol_repr = self.batch_norm(mol_repr)

        return self.out_lin(mol_repr)

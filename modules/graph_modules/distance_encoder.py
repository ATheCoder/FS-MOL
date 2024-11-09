from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import radius
from torch_geometric.utils import remove_self_loops

from modules.node_embeders.basis_layers import BesselBasisLayer


class DistanceEncoder(nn.Module):
    def __init__(self, bessel_basis_dim, cutoff, envelope_exponent, dim):
        super(DistanceEncoder, self).__init__()
        self.bessel_basis_dim = bessel_basis_dim
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        self.dim = dim

        self.rbf_g = BesselBasisLayer(bessel_basis_dim, self.cutoff, envelope_exponent)
        self.rbf_g_mlp = nn.Linear(bessel_basis_dim, self.dim)
        self.dist_norm = nn.BatchNorm1d(dim)

    def forward(self, data):
        pos, batch = data.pos, data.batch

        row, col = radius(pos, pos, self.cutoff, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        dist_g = dist_g + torch.tensor(1e-8)  # This is added for stability and removing 0s

        rbf_g = self.rbf_g(dist_g)
        rbf_g = F.relu(self.dist_norm(self.rbf_g_mlp(rbf_g)))

        return edge_index_g, rbf_g

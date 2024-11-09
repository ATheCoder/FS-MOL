from torch import Tensor
import torch
from torch_geometric.data import Data


def remove_nodes_in_edge_index(edge_index: Tensor, nodes_to_remove: Tensor):
    edge_mask = ~(torch.isin(edge_index, nodes_to_remove).any(0))

    new_edge_index = edge_index[:, edge_mask]

    offsets = (new_edge_index.unsqueeze(-1) > nodes_to_remove).sum(-1)

    return new_edge_index - offsets


def removeHs(mol: Data):
    x, edge_index, pos = mol.x, mol.edge_index, mol.pos

    none_hydrogen_indices = (x != 1).nonzero().squeeze()

    # HR sands for Hydrogens Removed
    hr_x = x[none_hydrogen_indices]

    if hr_x.dim() == 0:
        return mol

    hr_pos = None
    if mol.pos != None:
        hr_pos = pos[none_hydrogen_indices]

    hr_edge_index = remove_nodes_in_edge_index(edge_index, (x == 1).nonzero().squeeze())

    if hr_edge_index.shape[1] < 1:
        return mol

    return Data(x=hr_x, edge_index=hr_edge_index, pos=hr_pos)

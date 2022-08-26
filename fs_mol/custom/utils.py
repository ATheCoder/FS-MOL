import torch
from torch_geometric.data import Data

def convert_to_pyg_graph(graph):
    x = torch.tensor(graph.node_features)
    adjacency_lists = graph.adjacency_lists

    single_bonds = adjacency_lists[0]
    double_bonds = adjacency_lists[1]
    triple_bonds =  adjacency_lists[2]

    edge_index = torch.cat(list(map(torch.tensor, adjacency_lists)), dim=0)

    edge_feats = [0 for bond in single_bonds] + [1 for bond in double_bonds] + [2 for bond in triple_bonds]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_feats)
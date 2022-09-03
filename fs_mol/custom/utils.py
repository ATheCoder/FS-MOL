import torch
from torch_geometric.data import Data

def convert_to_pyg_graph(sample, device=None):
    graph = sample.graph
    y = 1 if sample.bool_label else 0

    x = torch.tensor(graph.node_features)
    adjacency_lists = graph.adjacency_lists

    single_bonds = adjacency_lists[0]
    double_bonds = adjacency_lists[1]
    triple_bonds =  adjacency_lists[2]

    edge_index = torch.cat(list(map(torch.tensor, adjacency_lists)), dim=0).t().contiguous()

    edge_feats = [0 for bond in single_bonds] + [1 for bond in double_bonds] + [2 for bond in triple_bonds]

    if device == None:
        return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_feats), y=torch.tensor(y))
    
    return Data(x=x, edge_index=edge_index, edge_attr=torch.tensor(edge_feats), y=torch.tensor(y)).to(device=device)
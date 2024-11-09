from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch
import numpy as np
import random

def nx_to_pyg(G, data):
    # Create a mapping of old node indices to new node indices
    node_mapping = {old: new for new, old in enumerate(G.nodes)}

    # Reconstruct edge_index with new node indices
    edge_index = torch.tensor([(node_mapping[i], node_mapping[j]) for i, j in G.edges()]).t().contiguous()

    # Reconstruct node features with new node indices
    x = data.x[list(node_mapping.keys())]

    # Reconstruct node positions with new node indices
    pos = data.pos[list(node_mapping.keys())]

    return Data(x=x, edge_index=edge_index, pos=pos)

def molclr_removeSubgraph(Graph, center, percent=0.2):
    assert percent <= 1
    G = Graph.copy()
    num = int(np.floor(len(G.nodes)*percent))
    removed = []
    temp = [center]
    
    while len(removed) < num:
        neighbors = []
        for n in temp:
            neighbors.extend([i for i in G.neighbors(n) if i not in temp])      
        for n in temp:
            if len(removed) < num:
                G.remove_node(n)
                removed.append(n)
            else:
                break
        temp = list(set(neighbors))
    return G, removed

def molclr_subgraph_augment(data):
    G = to_networkx(data)
    node_count = data.x.shape[0]
    
    start_i = random.sample(list(range(node_count)), 1)[0]
    
    a_G, removed = molclr_removeSubgraph(G, start_i, 0.2)
    
    return nx_to_pyg(a_G, data)
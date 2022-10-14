from typing import ForwardRef
from torch_geometric.data import Data
import numpy as np
import torch

# What is Sub Graph Augmentation?
class SubGraphAugmentation(torch.nn.Module):
    def __init__(self, aug_ratio):
        super().__init__()

        self.aug_ratio = aug_ratio
    
    def forward(self, data: Data):
        node_num, _ = data.x.size() # Count of Nodes
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * self.aug_ratio)

        edge_index = data.edge_index.numpy()

        idx_sub = [np.random.randint(node_num, size=1)[0]] # [3]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]]) # Neighbors of the choosen Node

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

        edge_index = data.edge_index.numpy()
        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
            data.x = data.x[idx_nondrop]
            data.edge_attr = data.edge_attr[edge_mask]
        except:
            data = data

        return data
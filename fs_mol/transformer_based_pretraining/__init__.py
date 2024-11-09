import math
import operator
import random
from itertools import islice

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx


def girvan_newman(G):
    """
    Applies the Girvan-Newman algorithm on the graph G
    and returns a generator of lists of nodes for each connected component
    after each step.
    """
    # The copy of G here is to avoid any side-effect on the original graph
    G = G.copy()
    while G.number_of_edges() > 0:
        # Compute edge betweenness centrality
        centrality = nx.edge_betweenness_centrality(G)
        # Identify the edge with maximum centrality
        max_centrality_edge = max(centrality.items(), key=operator.itemgetter(1))[0]
        # Remove the edge with highest centrality
        G.remove_edge(*max_centrality_edge)
        # Get the connected components and convert to list of nodes
        components = [list(c) for c in nx.connected_components(G)]
        yield components


def nx_to_pyg(G, data):
    # Create a mapping of old node indices to new node indices
    node_mapping = {old: new for new, old in enumerate(G.nodes)}

    # Reconstruct edge_index with new node indices
    edge_index = (
        torch.tensor([(node_mapping[i], node_mapping[j]) for i, j in G.edges()], dtype=torch.long)
        .t()
        .contiguous()
    )

    # Reconstruct node features with new node indices
    x = data.x[list(node_mapping.keys())]

    # Reconstruct node positions with new node indices
    pos = data.pos[list(node_mapping.keys())]

    return Data(x=x, edge_index=edge_index, pos=pos)


def select_subgraph_by_indices(graph, node_indices):
    subgraph = nx.Graph()

    for index in node_indices:
        node = list(graph.nodes)[index]
        subgraph.add_node(node)

    for u, v in graph.edges:
        if u in node_indices and v in node_indices:
            subgraph.add_edge(u, v)

    return subgraph


def get_a_division(data):
    G = to_networkx(data, to_undirected=True)
    node_count = data.x.shape[0]
    random_max = math.floor(node_count * 0.2) + 1
    random_num = random.choice(list(range(random_max)))
    sccs = list(islice(girvan_newman(G), random_num + 1))

    if len(sccs) == 0:
        return [data]

    sccs = sccs[random_num]

    subgraphs = []

    for subgraph_indices in sccs:
        selected_subgraph = select_subgraph_by_indices(G, subgraph_indices)
        subgraph_data = nx_to_pyg(selected_subgraph, data)
        subgraphs.append(subgraph_data)

    return subgraphs


class MolCombinerDataset(Dataset):
    def __init__(self, datafold) -> None:
        super().__init__()

        self.mols = torch.load(f"/FS-MOL/datasets/all_{datafold}_mxm_mols.pt")

    def __getitem__(self, index):
        src_mol = self.mols[index]

        return get_a_division(src_mol), src_mol

    def __len__(self):
        return len(self.mols)


def collate_fn(batch):
    # Unpack the batch
    division_lists, src_mols = zip(*batch)

    # Concatenate all divisions and src_mols in the batch
    all_divisions = [division for sublist in division_lists for division in sublist]
    all_src_mols = list(src_mols)
    try:
        # Create Batch objects
        division_batch = Batch.from_data_list(all_divisions)

        src_mol_batch = Batch.from_data_list(all_src_mols)

        # Create a tensor that indicates the molecule to which each division belongs
        molecule_index = [i for i, sublist in enumerate(division_lists) for _ in sublist]
        division_batch.molecule_index = torch.tensor(molecule_index)

        return division_batch, src_mol_batch
    except Exception as e:
        print(len(all_divisions))
        for i in all_divisions:
            print(i.x)
            print(i.pos)
            print(i.edge_index)
            print("---")
        raise e


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.scaling_factor = np.sqrt(d_k)
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

    def forward(self, fragments, molecule_index):
        # Compute Q, K, V by applying the linear transformations to the fragments
        Q = self.W_q(fragments)
        K = self.W_k(fragments)
        V = self.W_v(fragments)

        # Calculate the dot product between queries and keys, and scale it.
        attention_scores = torch.matmul(Q / self.scaling_factor, K.transpose(-2, -1))

        # Apply masking to ignore the fragments that do not belong to the current molecule.
        mask = molecule_index[:, None] != molecule_index[None, :]
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        # Apply softmax to get the attention weights.
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Calculate the weighted sum of the values.
        output = torch.matmul(attention_weights, V)

        # Sum the outputs for each molecule to get molecule embeddings.
        molecule_embeddings = torch.zeros(
            (molecule_index.max() + 1, output.shape[-1]), device=output.device
        )
        molecule_embeddings = molecule_embeddings.scatter_add_(
            0, molecule_index.unsqueeze(-1).expand_as(output), output
        )

        return molecule_embeddings

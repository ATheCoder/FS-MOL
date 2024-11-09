import numpy as np
import torch
from torch_geometric.data import Batch

from utils.pyg_mol_utils import make_graph_undirected
from utils.pyg_mol_utils.removehs import removeHs

def make_undirected(data):
    data = removeHs(data)
    data = make_graph_undirected(data)
    
    return data

def subgraph_collate(batch):
    task_names = [b["mols"][0].task_name for b in batch]
    molecules = np.concatenate([b["mols"] for b in batch])
    batch_index = torch.tensor(
        [z for i, b in enumerate(batch) for z in [i] * len(b["mols"])], dtype=torch.long
    )
    labels = torch.tensor([int(m.label) for m in molecules], dtype=torch.long)
    is_query_indices = torch.tensor(np.concatenate([b["is_query"] for b in batch]))

    sub_structure_to_mol_index = [
        z for i, mol in enumerate(molecules) for z in [i] * len(mol.features)
    ]

    sub_structure_graphs = [
        subgraph for b in batch for mol in b["mols"] for subgraph in mol.features
    ]
    
    sub_structure_graphs = [make_undirected(g) for g in sub_structure_graphs]

    sub_structure_graphs = Batch.from_data_list(sub_structure_graphs) # type: ignore

    return {
        "substructures": sub_structure_graphs,
        "substructure_mol_index": torch.tensor(sub_structure_to_mol_index, dtype=torch.long),
        "is_query": is_query_indices,
        "labels": labels,
        "batch_index": batch_index,
        "tasks": task_names,
    }
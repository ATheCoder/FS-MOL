import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data


def get_mol_poses(mol):
    params = AllChem.ETKDG()
    s = AllChem.EmbedMultipleConfs(mol, numConfs=3, params=params)
    mol_pos = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)

        pos = np.array([positions.x, positions.y, positions.z])

        mol_pos.append(pos)

    mol_pos = np.array(mol_pos)

    return mol_pos


def convert_fsmol_task_to_pyg(task):
    task.samples = [convert_to_pyg_graph(sample) for sample in task.samples]

    return task


def convert_to_pyg_graph(sample, device=None):
    graph = sample.graph
    y = 1 if sample.bool_label else 0
    pos = getattr(sample, "pos", None)

    x = torch.tensor(graph.node_features)
    adjacency_lists = graph.adjacency_lists

    single_bonds = adjacency_lists[0]
    double_bonds = adjacency_lists[1]
    triple_bonds = adjacency_lists[2]

    edge_index = torch.cat(list(map(torch.tensor, adjacency_lists)), dim=0).t().contiguous()

    edge_feats = (
        [0 for bond in single_bonds] + [1 for bond in double_bonds] + [2 for bond in triple_bonds]
    )
    fingerprint = torch.tensor(sample.fingerprint)

    pyg_data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.tensor(edge_feats),
        y=torch.tensor(y),
        bool_label=sample.bool_label,
        smiles=sample.smiles,
        fingerprint=fingerprint,
    )

    if pos is not None:
        pyg_data.pos = pos

    if device == None:
        return pyg_data

    return pyg_data.to(device=device)

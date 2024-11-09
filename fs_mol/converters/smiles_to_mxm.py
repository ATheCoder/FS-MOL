import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

from fs_mol.custom.utils import get_mol_poses


def generate_node_features(mol):
    types = {
        "UNK": 0,
        "H": 1,
        "C": 2,
        "N": 3,
        "O": 4,
        "S": 5,
        "Cl": 6,
        "Br": 7,
        "F": 8,
        "P": 9,
        "I": 10,
        "Na": 11,
        "Si": 12,
        "B": 13,
        "Se": 14,
        "K": 15,
    }
    type_idx = []

    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in types:
            type_idx.append(types["UNK"])
            continue
        type_idx.append(types[atom.GetSymbol()])

    return type_idx


def generate_edge_features(mol):
    N = mol.GetNumAtoms()

    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]

    return edge_index


def preprocess_smile(raw_smile, should_add_hydrogens=True, should_add_pos=True):
    mol = Chem.MolFromSmiles(raw_smile)
    if should_add_hydrogens:
        mol = Chem.AddHs(mol)

    x = generate_node_features(mol)

    pos = torch.tensor(get_mol_poses(mol), dtype=torch.float) if should_add_pos else None

    edge_index = generate_edge_features(mol)

    return Data(
        x=torch.tensor(x).to(torch.long),
        edge_index=edge_index,
        pos=pos,
    )


def preprocess_FSMOLSample(sample, should_addHs=True):
    raw_smile = sample.smiles

    mol = Chem.MolFromSmiles(raw_smile)
    if should_addHs:
        mol = Chem.AddHs(mol)

    x = generate_node_features(mol)

    pos = torch.tensor(get_mol_poses(mol), dtype=torch.float)

    edge_index = generate_edge_features(mol)

    return Data(
        x=torch.tensor(x).to(torch.long),
        edge_index=edge_index,
        pos=pos,
        bool_label=torch.tensor(sample.bool_label),
    )

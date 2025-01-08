import sys
import os
import inspect
from pathlib import Path


from fs_mol.custom.utils import get_mol_poses




currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
sys.path.insert(0, str(Path('/FS-MOL/GeoMol').resolve())) 


from rdkit import Chem
# from fs_mol.custom.utils import get_mol_poses
from GeoMol.generate_conf import generate_conformer, generate_conformers
from rdkit.Chem.rdchem import BondType as BT
import torch
from torch_geometric.data import Data
from torch_geometric.data import Data
from fs_mol.data.fsmol_task import MergedFSMOLSample
import os


def generate_node_features(mol):
    types = {
        'UNK': 0,
        'H': 1,
        'C': 2,
        'N': 3,
        'O': 4,
        'S': 5,
        'Cl': 6,
        'Br': 7,
        'F': 8,
        'P': 9,
        'I': 10,
        'Na': 11,
        'Si': 12,
        'B': 13,
        'Se': 14,
        'K': 15,
    }
    type_idx = []
    
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in types:
            type_idx.append(types['UNK'])
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


def generate_rdkit_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    return mol

def preprocess_smiles(samples):
    raw_smiles = [sample['SMILES'] for sample in samples]
    mol = generate_rdkit_mol(raw_smile)
    
    x = generate_node_features(mol)

    pos = torch.tensor(get_mol_poses(generate_conformers(raw_smiles, 1)[0]), dtype=torch.float)
    
    edge_index = generate_edge_features(mol)
    
    graph = Data(
        x=torch.tensor(x).to(torch.long),
        edge_index=edge_index,
        pos=pos,
        bool_label=torch.tensor(sample['Property'] == '1.0'),
    )
    
    
    return MergedFSMOLSample(
        descriptors=sample['descriptors'],
        fingerprints=sample['fingerprints'],
        graph=graph,
        SMILES=sample['SMILES'],
        task_name=sample['Assay_ID'],
        label=torch.tensor(sample['Property'] == '1.0')
    )
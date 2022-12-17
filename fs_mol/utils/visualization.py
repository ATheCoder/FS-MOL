from torch_geometric.data import Data
from rdkit.Chem import RWMol
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
import torch

# Features:
# AtomTypeFeatureExtractor
# AtomDegreeFeatureExtractor
# AtomChargeFeatureExtractor
# AtomNumRadicalElectronsFeatureExtractor
# AtomRingInformationExtractor

i_atomic_table = {
    'U': 0,
    'C': 1,
    'Br': 6,
    'N': 2,
    'O': 3,
    'S': 4,
    'Cl': 5,
    'F': 7,
    'P': 8,
    'I': 9,
    'B': 12,
    'Si': 11,
    'Se': 13,
    'Na': 10,
    'K': 14
}

atomic_table = {j: i for i, j in i_atomic_table.items()}

def get_atomic_symbol(vec): # The first 15 are one_hot_encoded labels indicating the atomic label
    return atomic_table[torch.argmax(vec[:15]).item()]

def get_atom_degree(vec):
    return torch.argmax(vec[15:22]).item()

def get_atom_formal_charge(vec):
    return torch.argmax(vec[22:28]).item() - 2


def get_atom_radical_electron_count(vec):
    return torch.argmax(vec[28:31]).item() - 1
    


def convert_graph_to_mol(graph: Data):
    mol = RWMol()
    node_to_idx = {}
    for index_in_features, node in enumerate(graph.x):
        a = Chem.Atom(get_atomic_symbol(node))
        a.SetAtomMapNum(index_in_features)
        a.SetFormalCharge(get_atom_formal_charge(node))
        a.SetNumRadicalElectrons(get_atom_radical_electron_count(node))
        
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx
    
    
    edges = graph.edge_index.t()
    edge_to_bond_type = graph.edge_attr
    
    rdkit_bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
    
    for edge_i, edge in enumerate(edges):
        bond_type = edge_to_bond_type[edge_i]
        
        mol.AddBond(edge[0].item(), edge[1].item(), rdkit_bond_types[int(bond_type.item())])
        
    Chem.SanitizeMol(mol)
    
    return mol


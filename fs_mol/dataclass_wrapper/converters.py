from networkx import Graph
from rdkit import Chem
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from fs_mol.converters.smiles_to_mxm import preprocess_smile
from utils.pyg_mol_utils.removehs import removeHs

atom_dict = {
    0: "UNK",
    1: "H",
    2: "C",
    3: "N",
    4: "O",
    5: "S",
    6: "Cl",
    7: "Br",
    8: "F",
    9: "P",
    10: "I",
    11: "Na",
    12: "Si",
    13: "B",
    14: "Se",
    15: "K",
}


def convert_Smiles_to_Data(smiles: str) -> Data:
    return removeHs(preprocess_smile(smiles))


def convert_Data_to_Smiles(data: Data) -> str:
    mol = Chem.RWMol()

    # Add atoms
    atomic_nums = data.x.tolist()
    for atomic_num in atomic_nums:
        print(atom_dict[atomic_num])
        atom = Chem.Atom(atom_dict[atomic_num])
        mol.AddAtom(atom)

    # Add unique bonds
    edge_index = data.edge_index.t().tolist()
    edge_set = set()
    for i, j in edge_index:
        if (i, j) in edge_set or (j, i) in edge_set:
            continue
        # Add a bond between atoms i and j, here assuming it's a single bond.
        mol.AddBond(i, j, Chem.BondType.SINGLE)
        edge_set.add((i, j))

    # Set 3D coordinates for atoms
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        pos = data.pos[i]
        conf.SetAtomPosition(i, Point3D(pos[0].item(), pos[1].item(), pos[2].item()))
    mol.AddConformer(conf)

    return mol


def convert_Data_to_NetworkXGraph(data: Data) -> Graph:
    return to_networkx(data, to_undirected=True)

from os import makedirs
import os
import torch
from rdkit import Chem
from rdkit.Chem import Draw, RWMol, rdDepictor
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D
from torch_geometric.data import Data

from fs_mol.dataclass_wrapper import graph_dynamic
from utils.pyg_mol_utils.removehs import removeHs
from IPython.core.display import SVG
import nglview as nv


# Features:
# AtomTypeFeatureExtractor
# AtomDegreeFeatureExtractor
# AtomChargeFeatureExtractor
# AtomNumRadicalElectronsFeatureExtractor
# AtomRingInformationExtractor

i_atomic_table = {
    "U": 0,
    "C": 1,
    "Br": 6,
    "N": 2,
    "O": 3,
    "S": 4,
    "Cl": 5,
    "F": 7,
    "P": 8,
    "I": 9,
    "B": 12,
    "Si": 11,
    "Se": 13,
    "Na": 10,
    "K": 14,
}

atomic_table = {j: i for i, j in i_atomic_table.items()}


def get_atomic_symbol(vec):  # The first 15 are one_hot_encoded labels indicating the atomic label
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


@graph_dynamic
def visualize_mol(input: Data):
    return visualize_pyg_mol_with_poses(input)


def visualize_rdkit_mol(mol):
    return Draw.MolToImage(mol)


def construct_2d_mol(data: Data):
    mol = Chem.RWMol()

    # Add atoms
    atomic_nums = data.x.tolist()
    for atomic_num in atomic_nums:
        atom = Chem.Atom(atom_dict[atomic_num])
        mol.AddAtom(atom)

    # Add unique bonds
    edge_index = data.edge_index.t().tolist()
    print(len(edge_index))
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

    # Compute 2D coordinates for visualization
    rdDepictor.Compute2DCoords(mol)

    return mol


def save_mol_as_svg(mol, path):
    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)  # Here, 300x300 is the canvas size
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(svg)


def generate_mol_from_pyg_data(data: Data, _3d=False):
    # Initialize empty editable molecule
    mol = Chem.RWMol()

    # Add atoms
    atomic_nums = data.x.tolist()
    for atomic_num in atomic_nums:
        atom = Chem.Atom(atom_dict[atomic_num])
        mol.AddAtom(atom)

    # Add unique bonds
    edge_index = data.edge_index.t().tolist()
    print(len(edge_index))
    edge_set = set()
    for i, j in edge_index:
        if (i, j) in edge_set or (j, i) in edge_set:
            continue
        # Add a bond between atoms i and j, here assuming it's a single bond.
        mol.AddBond(i, j, Chem.BondType.SINGLE)
        edge_set.add((i, j))

    # Set 3D coordinates for atoms

    if _3d == False:
        rdDepictor.Compute2DCoords(mol)
        return mol

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        pos = data.pos[i]
        conf.SetAtomPosition(i, Point3D(pos[0].item(), pos[1].item(), pos[2].item()))
    mol.AddConformer(conf)
    # Compute 2D coordinates for visualization

    return mol


def visualize_pyg_mol_with_poses(data: Data, show_indices=False, _3d=False, remove_Hs=False):
    if remove_Hs:
        data = removeHs(data)
    # Initialize empty editable molecule
    mol = generate_mol_from_pyg_data(data, _3d)

    if _3d:
        return nv.show_rdkit(mol)

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    if show_indices == True:
        mol_with_labels = Chem.Mol(mol)
        for atom in mol_with_labels.GetAtoms():
            atom.SetProp("atomLabel", str(atom.GetIdx()))
            drawer.DrawMolecule(mol_with_labels)
    else:
        drawer.DrawMolecule(mol)
        
    drawer.FinishDrawing()
    svg = SVG(drawer.GetDrawingText())

    # Draw the molecule
    return svg


# def convert_svg_to_


# def display_molecules(mol_list, rows=None, cols=None):
#     if rows is None and cols is None:
#         raise ValueError("You must specify either the 'rows' or 'cols' parameter.")
#     if rows is not None and cols is not None:
#         raise ValueError("You can only specify either the 'rows' or 'cols' parameter, not both.")

#     num_mols = len(mol_list)

#     if rows:
#         rows = min(rows, num_mols - 1)

#     if cols:
#         cols = min(cols, num_mols - 1)

#     if rows is None:
#         cols = cols if cols is not None else int(num_mols**0.5)
#         rows = (num_mols + cols - 1) // cols
#     else:
#         rows = rows if rows is not None else int(num_mols**0.5)
#         cols = (num_mols + rows - 1) // rows

#     fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
#     fig.tight_layout(pad=0.3)

#     for i, mol in enumerate(mol_list):
#         row = i // cols
#         col = i % cols

#         ax = axes[row, col]
#         ax.axis("off")

#         ax.imshow(mol)

#     # Remove any extra empty axes
#     if num_mols < rows * cols:
#         for i in range(num_mols, rows * cols):
#             ax = axes.flatten()[i]
#             ax.axis("off")
#             ax.set_visible(False)

#     plt.show()
from IPython.display import display


from IPython.core.display import display, HTML


def display_molecules(svg_list, rows=None, cols=None):
    if rows is None and cols is None:
        raise ValueError("You must specify either the 'rows' or 'cols' parameter.")
    if rows is not None and cols is not None:
        raise ValueError("You can only specify either the 'rows' or 'cols' parameter, not both.")

    num_svgs = len(svg_list)

    if rows:
        rows = min(rows, num_svgs)

    if cols:
        cols = min(cols, num_svgs)

    if rows is None:
        cols = cols if cols is not None else int(num_svgs**0.5)
        rows = (num_svgs + cols - 1) // cols
    else:
        rows = rows if rows is not None else int(num_svgs**0.5)
        cols = (num_svgs + rows - 1) // rows

    table_html = '<table style="background: white;">'
    for i in range(rows):
        table_html += "<tr>"
        for j in range(cols):
            idx = i * cols + j
            if idx < num_svgs:
                svg_str = str(svg_list[idx].data)
                table_html += f"<td>{svg_str}</td>"
            else:
                table_html += "<td></td>"
        table_html += "</tr>"
    table_html += "</table>"

    display(HTML(table_html))

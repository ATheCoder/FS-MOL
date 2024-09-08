import sys
import os
import inspect

import ray



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pickle
from rdkit import Chem
from fs_mol.custom.utils import get_mol_poses
from rdkit.Chem.rdchem import BondType as BT
import torch
from torch_geometric.data import Data
from torch_geometric.data import Data
import gzip
import json
from pathlib import Path
from fs_mol.data.fsmol_task import FSMolTask, MergedFSMOLSample
import pickle
import os
from ray.experimental import tqdm_ray



    

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

def preprocess_smile(sample):
    raw_smile = sample['SMILES']
    
    mol = Chem.MolFromSmiles(raw_smile)
    mol = Chem.AddHs(mol)
    
    x = generate_node_features(mol)
    
    pos = torch.tensor(get_mol_poses(mol), dtype=torch.float)
    
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
    
    
ray.init()

fold = 'train'


def parse_jsongz(p):
# Open the gzipped JSONL file
    with gzip.open(p, 'rt', encoding='utf-8') as file:
        # Iterate over each line in the file
        return [json.loads(line.strip()) for line in file]

dest_root = f'/FS-MOL/datasets/fs-mol-merged-cleaned/{fold}/'

os.makedirs(dest_root, exist_ok=True)

@ray.remote
def single_path_processor(path: Path, preprocessor_func, bar):
    task = parse_jsongz(path)
    task_name = path.name.split('.')[0]
    
    samples = []
    
    for sample in task:
        try:
            new_sample = preprocessor_func(sample)
            samples.append(new_sample)
        except:
            print(f'Skipped One Molecule in {task_name}')
            
    if len(samples) < 32:
        return None
    task = FSMolTask(task_name, samples)
    
    
    if len(samples) > 0:
        with open(f'{dest_root}/{task.name}.pkl', 'w+b') as f:
            pickle.dump(task, f, protocol=pickle.HIGHEST_PROTOCOL)
    bar.update.remote(1)
    print('done')
    return task
    
raw_path = f'/FS-MOL/datasets/fs-mol/{fold}/'

fsmol_root_dir = Path(f'/FS-MOL/datasets/fs-mol')
fsmol_task_list = json.load(open('/FS-MOL/datasets/fsmol-0.1.json', 'r+b'))[fold] # Original FSMOL tasks

fsmol_task_paths = [ fsmol_root_dir / fold / f'{tn}.jsonl.gz' for tn in fsmol_task_list]

remote_tqdm = ray.remote(tqdm_ray.tqdm)
bar = remote_tqdm.remote(total=len(fsmol_task_paths))

tasks = [single_path_processor.remote(path, preprocess_smile, bar) for path in fsmol_task_paths]

ray.get(tasks)



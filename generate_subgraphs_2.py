import os
from pathlib import Path
import pickle
from joblib import Memory, Parallel, delayed
import torch
from tqdm import tqdm
from fewshot_utils.tqdm_joblib import tqdm_joblib
from preprocessing.geometric import break_down_graphs

start_path = Path('/data/data/no_hydrogens_no_3d')
new_path = Path('/data/data/pretrain_data_using_subgraphs')

memory = Memory("/data/data/cache_folder", verbose=0)

os.makedirs(new_path, exist_ok=True)

def generate_subgraphs(mol_path):
    mol = torch.load(mol_path)
    first_broken_down = break_down_graphs(mol)
    result = [(first_broken_down, mol)]
    to_check = [] + first_broken_down
    while len(to_check) > 0:
        current_graph = to_check.pop()
        if current_graph.x.shape[0] > 5:
            broken_down = break_down_graphs(current_graph)
            if len(broken_down) > 1:
                to_check = to_check + broken_down
                result.append((current_graph, broken_down))
        
    return result

def generate_subgraphs_recursive(mol, result=None):
    if result is None:
        result = []
    
    if mol.x.shape[0] > 5:
        broken_down = break_down_graphs(mol)
        if len(broken_down) <= 1:
            return result
        result.append((mol, broken_down))
        for subgraph in broken_down:
            generate_subgraphs_recursive(subgraph, result)
    
    return result

def generate_x_y(path: Path):
    try:
        mol = torch.load(path)
        samples = generate_subgraphs_recursive(mol)
        sample_no = path.name.split('.')[0]
        
        results = []
        
        for i, sample in enumerate(samples):
            torch.save(sample, new_path / f'{sample_no}_{i}.pt', pickle_protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Could not do it:')
        print(path)
        print(e)

all_paths = os.listdir(start_path)

already_mades = os.listdir(new_path)

already_mades = set([i.split('_')[0] + '.pt' for i in already_mades])

all_paths = [i for i in all_paths if i not in already_mades]

with tqdm_joblib(tqdm("Molecules Processed!", total=len(all_paths))):
    Parallel(n_jobs=6)(delayed(generate_x_y)(start_path / p) for p in all_paths)

# generate_subgraphs('/data/data/no_hydrogens_no_3d/648606.pt')
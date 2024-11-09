from dataclasses import dataclass
import os
from pathlib import Path
import random
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from fs_mol.data_modules.MXM_datamodule import MXMValidationDataset
from fs_mol.utils.collate_fns import subgraph_collate
from preprocessing.geometric import get_biconnected_subgraphs
from utils.pyg_mol_utils import make_graph_undirected
from utils.pyg_mol_utils.removehs import removeHs


class Pretrain10Dataset(Dataset):
    def __init__(self, dataset_path, broken_path) -> None:
        super().__init__()

        self.root_path = Path(dataset_path)
        self.broken_path = Path(broken_path)
        self.available_graphs = torch.tensor([int(i.split('.')[0]) for i in os.listdir(self.broken_path)], dtype=torch.long)
        
        

    def __getitem__(self, index):
        file_name = f'{self.available_graphs[index]}.pt'
        full_graph = torch.load(self.root_path / file_name)
        broken_down = torch.load(self.broken_path / file_name)
        choosen_breakage = random.sample(list(broken_down), 1)[0]
        
        random.shuffle(choosen_breakage)
        return choosen_breakage, full_graph
        

    def __len__(self):
        return self.available_graphs.shape[0]

class PretrainWithSubgraphsTransform(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.molecules = os.listdir(self.data_path)
        
    def __getitem__(self, index):
        mol_smiles = torch.load(self.data_path / self.molecules[index])
        # result = generate_subgraphs_recursive(mol_smiles, [])
        
        return [(mol_smiles, get_biconnected_subgraphs(mol_smiles))]
        # return result
    
    def __len__(self):
        return len(self.molecules)

class PretrainWithSubgraphs(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.all_datas = os.listdir(self.data_path)
        
    def __getitem__(self, index):
        file_index = index // 10
        index_in_file = index % 10
        
        loaded_file = torch.load(self.data_path / self.all_datas[file_index])
        return loaded_file[index_in_file][0]
        

    def __len__(self):
        return len(self.all_datas) * 10 - 1

class PretrainWithSubgraphsDataModule(LightningDataModule):
    def __init__(self, train_dataset_path, train_broken_path, batch_size, validation_datapath, valid_support_size) -> None:
        super().__init__()
        
        self.train_dataset_path = train_dataset_path
        self.train_broken_path = train_broken_path
        self.batch_size = batch_size
        self.validation_datapath = validation_datapath
        self.valid_support_size = valid_support_size
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            # self.train = PretrainWithSubgraphs(self.train_dataset_path)
            self.train = PretrainWithSubgraphsTransform(self.train_dataset_path)
            self.valid = MXMValidationDataset(
                self.validation_datapath,
                support_size=self.valid_support_size,
                split="test",
                use_subgraph=True,
            )
    
    def preprocess_graph(self, data):
        data = removeHs(data)
        data = make_graph_undirected(data)
        
        return data
    
    def collate_fn(self, batch):
        batch = [g for b in batch for g in b]
        graphs, subgraphs = zip(*batch)
        
        subgraph_graph_index = [i for b_index, s in enumerate(subgraphs) for i in len(s) * [b_index]]
        
        # For Graphs we use -1
        # subgraph_graph_index = subgraph_graph_index + [-1] * len(graphs)
        
        flat_subgraphs = [s for broken_down in subgraphs for s in broken_down]
        all_graphs = flat_subgraphs + list(graphs)
        all_graphs = [self.preprocess_graph(g) for g in all_graphs]
        all_graphs = Batch.from_data_list(flat_subgraphs + list(graphs))
        
        return {'graphs': all_graphs, 'subgraph_graph_index': torch.tensor(subgraph_graph_index, dtype=torch.long), 'mol_size': len(graphs)}
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=6, prefetch_factor=1)
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid, batch_size=1, shuffle=False, collate_fn=subgraph_collate, drop_last=True)
        


@dataclass
class MoleculesWithSubMoleculesDataPath:
    dataset_path: str
    broken_path: str

class Pretrain10DataModule(LightningDataModule):
    def __init__(self, train_dataset_path, train_broken_path, batch_size, validation_datapath, valid_support_size) -> None:
        super().__init__()
        
        self.train_dataset_path = train_dataset_path
        self.train_broken_path = train_broken_path
        self.batch_size = batch_size
        self.validation_datapath = validation_datapath
        self.valid_support_size = valid_support_size
        
    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = Pretrain10Dataset(self.train_dataset_path, self.train_broken_path)
            self.valid = MXMValidationDataset(
                self.validation_datapath,
                support_size=self.valid_support_size,
                split="test",
                use_subgraph=True,
            )
    
    def preprocess_graph(self, data):
        data = removeHs(data)
        data = make_graph_undirected(data)
        
        return data
    
    def collate_fn(self, batch):
        subgraphs, graphs = zip(*batch)
        
        subgraph_graph_index = [i for b_index, s in enumerate(subgraphs) for i in len(s) * [b_index]]
        
        # For Graphs we use -1
        # subgraph_graph_index = subgraph_graph_index + [-1] * len(graphs)
        
        flat_subgraphs = [s for broken_down in subgraphs for s in broken_down]
        all_graphs = flat_subgraphs + list(graphs)
        all_graphs = [self.preprocess_graph(g) for g in all_graphs]
        all_graphs = Batch.from_data_list(flat_subgraphs + list(graphs))
        
        return {'graphs': all_graphs, 'subgraph_graph_index': torch.tensor(subgraph_graph_index, dtype=torch.long), 'mol_size': len(graphs)}
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train, self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=6)
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid, batch_size=1, shuffle=False, collate_fn=subgraph_collate, drop_last=True)
        

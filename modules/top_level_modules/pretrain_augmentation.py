import os
from pathlib import Path
from fs_mol.augmentation_transforms.k_hop_subgraph import remove_k_hop_subgraph
from lightning import LightningDataModule, LightningModule
import numpy as np
from torch import nn
import torch
from torch.utils.data import Dataset
from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder
from torch.utils.data import DataLoader
from torch_geometric.data import Data

class RemoveKHopSubgraph(nn.Module):
    def __init__(self, k = 3) -> None:
        super().__init__()
        self.k = k
    
    def forward(self, mol: Data):
        return remove_k_hop_subgraph(mol, self.k)

class PretrainWithAugmentationDataset(Dataset):
    def __init__(self, dataset_path, transform, n_views) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.n_views = n_views
        
        self.all_datapoints = os.listdir(self.dataset_path)
    def __getitem__(self, index):
        mol = torch.load(self.dataset_path / f'{index}.pt')
        # transformed_mol = self.transform(mol)
        return [self.transform(mol) for _ in range(self.n_views)]
        # return self.n_views * [self.transform(mol)]

class PretrainWithAugmentation(LightningDataModule):
    def __init__(self, data_path, batch_size) -> None:
        super().__init__()
        
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        
    def setup(self, stage):
        if stage == 'fit':
            self.train = PretrainWithAugmentationDataset(str(self.data_path))
            
    def collate_fn(batch):
        x, y = zip(*batch)
        
        
            
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0)

class PretrainUsingAugmentations(LightningModule):
    def __init__(self, n_layers, dim) -> None:
        super().__init__()
        
        self.graph_encoder = ResidualGatedGraphEncoder(dim, None, dim, n_layers)
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        
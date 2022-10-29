import sys

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))
import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fs_mol.augmentation_transforms import SubGraphAugmentation
from fs_mol.data.self_supervised_learning import FSMolSelfSupervisedInMemory
from fs_mol.modules.pyg_gnn import PyG_GraphFeatureExtractor
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractorConfig


device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_subgraph = FSMolSelfSupervisedInMemory('./datasets/self-supervised', transform=SubGraphAugmentation(0.2), device=device)

number_of_epochs = 1000
batch_size = 32

dl = DataLoader(dataset_subgraph, batch_size=batch_size)
dl2 = DataLoader(dataset_subgraph, batch_size=batch_size)

model = PyG_GraphFeatureExtractor(GraphFeatureExtractorConfig())

model.to(device)

optm = Adam(model.parameters())

def calculate_contrastive_loss(x1, x2):
    T = 0.1
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

for epoch in range(number_of_epochs):
    
    for batch_1, batch_2 in tqdm(zip(dl, dl2), total=len(dl)):
        optm.zero_grad()
        features_1 = model(batch_1)
        features_2 = model(batch_2)
        loss = calculate_contrastive_loss(features_1, features_2)
        loss.backward()
        optm.step()
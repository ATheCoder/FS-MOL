import sys

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.models.protonet import PyG_PrototypicalNetwork
from fs_mol.data.protonet import PyG_ProtonetBatch, get_protonet_batcher, task_sample_to_pn_task_sample
from fs_mol.data.pyg_task_reader import pyg_task_reader_fn
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.utils.torch_utils import torchify
from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainerConfig, evaluate_protonet_model, run_on_batches
from fs_mol.utils.test_utils import eval_model, FSMolTaskSampleEvalResults
from fs_mol.data import FSMolDataset, FSMolTaskSample, DataFold
from dpu_utils.utils.richpath import RichPath

import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fs_mol.augmentation_transforms import SubGraphAugmentation
from fs_mol.data.self_supervised_learning import FSMolSelfSupervisedInMemory
from fs_mol.modules.pyg_gnn import PyG_GraphFeatureExtractor
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractorConfig
from torch_geometric.loader import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_subgraph = FSMolSelfSupervisedInMemory('./datasets/self-supervised', transform=SubGraphAugmentation(0.2, device=device), device=device)
dataset_subgraph_2 = FSMolSelfSupervisedInMemory('./datasets/self-supervised', transform=SubGraphAugmentation(0.2, device=device), device=device)

number_of_epochs = 1000
batch_size = 32

dl = DataLoader(dataset_subgraph, batch_size=batch_size)
dl2 = DataLoader(dataset_subgraph_2, batch_size=batch_size)

model = PyG_GraphFeatureExtractor(GraphFeatureExtractorConfig()).to(device)

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

def validate_model(encoderModel):
    model = PyG_PrototypicalNetwork(encoderModel)
    dataset = FSMolDataset.from_directory('./datasets/fs-mol', num_workers=0)
    
    def test_model_fn(task_sample: FSMolTaskSample, temp_out_folder: str, seed: int) -> BinaryEvalMetrics:
        batch_size = 320 # default value on `evaluate_protonet_model` in `protonet_utils.py`
        
        # Batch size is 320 and all of the support sets should fit into a single batch
        train_batch = list(DataLoader(task_sample.train_samples, batch_size=batch_size))
        
        if len(train_batch) > 1:
            raise ValueError('Batch Size for support set is too small')
        
        support_features = train_batch[0]
        
        test_batches = list(DataLoader(task_sample.test_samples, batch_size=batch_size - len(train_batch)))
        
        batches = list(map(lambda batch: PyG_ProtonetBatch(support_graphs=support_features, query_graphs=batch), test_batches))
        
        batch_labels = list(map(lambda batch: batch.query_graphs.y, batches))
        

        _, result_metrics = run_on_batches(
            model,
            batches=batches,
            batch_labels=batch_labels,
            train=False,
        )

        return result_metrics
    
    return eval_model(
            test_model_fn=test_model_fn,
            dataset=dataset,
            train_set_sample_sizes=[16, 128], # This is from parse_command_line in `protonet_train.py`
            out_dir=None, # What is save Dir? Seems like this is None
            num_samples=5, # This is from `--validation-num-samples` parse_command_line in `protonet_train.py` 
            test_size_or_ratio=512, # `--validation-query-set-size` parse_command_line in `protonet_train.py` 
            task_reader_fn=pyg_task_reader_fn,
            fold=DataFold.VALIDATION, # This is used from `validate_by_finetuning_on_tasks` in `protonet_utils.py` This is validation because during the training loop inside the `train_loop` function FS-MOL calls `validate_by_finetuning_on_tasks`
            seed=0 # This is from `validate_by_finetuning_on_tasks`'s default parameters
        )
    
for epoch in range(1, number_of_epochs + 1):
    # Pre_training:
    for batch_1, batch_2 in tqdm(zip(dl, dl2), total=len(dl)):
        optm.zero_grad()
        features_1 = model(batch_1)
        features_2 = model(batch_2)
        loss = calculate_contrastive_loss(features_1, features_2)
        loss.backward()
        optm.step()
    
    # Validation:
    if epoch % 100 == 0:
        validate_model(encoderModel=model)
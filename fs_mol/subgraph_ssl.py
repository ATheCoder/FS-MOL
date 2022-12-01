from functools import partial
import sys

from pyprojroot import here as project_root
from dataclasses import asdict, fields


sys.path.insert(0, str(project_root()))

from fs_mol.utils.protonet_utils import run_on_batches
from fs_mol.utils.test_utils import eval_model
from fs_mol.models.protonet import PyG_PrototypicalNetwork
from fs_mol.data.protonet import PyG_ProtonetBatch, get_protonet_batcher, task_sample_to_pn_task_sample
from fs_mol.data.pyg_task_reader import pyg_task_reader_fn
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.data import FSMolDataset, FSMolTaskSample, DataFold
from fs_mol.models.abstract_torch_fsmol_model import linear_warmup

import torch
import wandb
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from fs_mol.augmentation_transforms import SubGraphAugmentation
from fs_mol.data.self_supervised_learning import FSMolSelfSupervisedInMemory
from fs_mol.modules.pyg_gnn import PyG_GraphFeatureExtractor
from fs_mol.modules.graph_feature_extractor import GraphFeatureExtractorConfig
from torch_geometric.loader import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


number_of_epochs = 1000

config = {
    "graph_feature_extractor_config": GraphFeatureExtractorConfig(),
    "pretraining_epochs": number_of_epochs,
    "pre_training_batch_size": 32,
    "testing_batch_size": 320,
    "learning_rate": 0.0001,
    "warmup_steps": 100,
    "keep_ratio": 0.2,
}

config_snapshot = {
    **config,
    "graph_feature_extractor_config": asdict(GraphFeatureExtractorConfig())
}

dataset_subgraph = FSMolSelfSupervisedInMemory('./datasets/self-supervised', transform=SubGraphAugmentation(config['keep_ratio'], device=device), device=device)
dataset_subgraph_2 = FSMolSelfSupervisedInMemory('./datasets/self-supervised', transform=SubGraphAugmentation(config['keep_ratio'], device=device), device=device)

run = wandb.init(project="FS-MOL-GraphCL", config=config_snapshot)

dl = DataLoader(dataset_subgraph, batch_size=config["pre_training_batch_size"])
dl2 = DataLoader(dataset_subgraph_2, batch_size=config["pre_training_batch_size"])

model = PyG_GraphFeatureExtractor(GraphFeatureExtractorConfig()).to(device)

wandb.watch(model)

run.define_metric('epoch')

run.define_metric('optimistic_delta_auc_pr/*', step_metric='epoch')

run.define_metric('optimistic_delta_auc_pr/*', step_metric='epoch')

optm = Adam(model.parameters(), lr=config["learning_rate"])

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer=optm,
    lr_lambda=partial(linear_warmup, warmup_steps=config["warmup_steps"]),  # for loaded GNN params
)

def calculate_contrastive_loss(x1, x2, size):
    T = 0.1
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(size), range(size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    wandb.log({"contrastive_loss": loss})
    return loss


model_artifact = wandb.Artifact('feature_extractor', type='model')

validation_dataset = FSMolDataset.from_directory('./datasets/fs-mol', task_list_file="./datasets/fsmol-0.1.json", num_workers=0)

def validate_model(encoderModel):
    model = PyG_PrototypicalNetwork(encoderModel)
    
    def test_model_fn(task_sample: FSMolTaskSample, temp_out_folder: str, seed: int) -> BinaryEvalMetrics:
        batch_size = 256 # `PrototypicalNetworkTrainerConfig` from `protonet_utils.py`
        
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
        
        wandb.log(result_metrics.__dict__)
        print(result_metrics)

        return result_metrics
    
    return eval_model(
            test_model_fn=test_model_fn,
            dataset=validation_dataset,
            train_set_sample_sizes=[16, 128], # This is from parse_command_line in `protonet_train.py`
            out_dir=None, # What is save Dir? Seems like this is None
            num_samples=5, # `PrototypicalNetworkTrainerConfig` from `protonet_utils.py`
            test_size_or_ratio=256, # `PrototypicalNetworkTrainerConfig` from `protonet_utils.py`
            task_reader_fn=pyg_task_reader_fn,
            fold=DataFold.VALIDATION, # This is used from `validate_by_finetuning_on_tasks` in `protonet_utils.py` This is validation because during the training loop inside the `train_loop` function FS-MOL calls `validate_by_finetuning_on_tasks`
            seed=0 # This is from `validate_by_finetuning_on_tasks`'s default parameters
        )
    
for epoch in range(1, number_of_epochs + 1):
    # Pre_training:
    wandb.log({'epoch': epoch})
    for batch_1, batch_2 in tqdm(zip(dl, dl2), total=len(dl)):
        optm.zero_grad()
        features_1 = model(batch_1)
        features_2 = model(batch_2)
        loss = calculate_contrastive_loss(features_1, features_2, batch_1.num_graphs)
        loss.backward()
        optm.step()
        lr_scheduler.step()

    result = validate_model(encoderModel=model)
    file_name = './pretraining_feature_extractor_{epoch}.pt'
    torch.save(model, file_name)
    # model_artifact.add_file(file_name)
    # run.log_artifact(model_artifact)

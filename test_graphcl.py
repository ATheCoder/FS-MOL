import torch
import wandb
from fs_mol.data import FSMolDataset
from fs_mol.models.protonet import PyG_PrototypicalNetwork

from fs_mol.data.fsmol_task import FSMolTaskSample
from fs_mol.models.protonet import PyG_PrototypicalNetwork
from fs_mol.utils.metrics import BinaryEvalMetrics
from torch_geometric.loader import DataLoader
from fs_mol.data.protonet import PyG_ProtonetBatch
from fs_mol.utils.protonet_utils import run_on_batches
from fs_mol.utils.test_utils import eval_model
from fs_mol.data.pyg_task_reader import pyg_task_reader_fn
from fs_mol.data import FSMolTaskSample, DataFold, FSMolDataset

MODEL_LOCATION = './pretraining_feature_extractor_48.pt'

encoderModel = torch.load(MODEL_LOCATION)

validation_dataset = FSMolDataset.from_directory('./datasets/fs-mol', task_list_file="./datasets/fsmol-0.1.json", num_workers=0)

def validate_model(encoderModel):
    model = PyG_PrototypicalNetwork(encoderModel)
    
    def test_model_fn(task_sample: FSMolTaskSample, temp_out_folder: str, seed: int) -> BinaryEvalMetrics:
        batch_size = 320 # `PrototypicalNetworkTrainerConfig` from `protonet_utils.py`
        
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
            dataset=validation_dataset,
            train_set_sample_sizes=[16, 32, 64, 128, 256], # This is from parse_command_line in `protonet_train.py`
            # out_dir='outputs/FSMol_Eval_ProtoNet_2022-11-26_14-19-50', # What is save Dir? Seems like this is None
            num_samples=10, # `PrototypicalNetworkTrainerConfig` from `protonet_utils.py`
            task_reader_fn=pyg_task_reader_fn,
            fold=DataFold.TEST, # This is used from `validate_by_finetuning_on_tasks` in `protonet_utils.py` This is validation because during the training loop inside the `train_loop` function FS-MOL calls `validate_by_finetuning_on_tasks`
            seed=0 # This is from `validate_by_finetuning_on_tasks`'s default parameters
        )

wandb.init('Validate-Using-Test')
validate_model(encoderModel)
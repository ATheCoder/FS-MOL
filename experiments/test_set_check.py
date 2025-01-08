import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder, ResidualGatedGraphEncoderConfig
import torch
import lightning as L
from dataclasses import dataclass
from typing import Any
from lightning.pytorch.loggers import WandbLogger
import wandb
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat

from modules.graph_encoders.GINEncoder import GINEncoder, GINEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FeedForwardConfig, FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics
from fs_mol.data.torch_dl import MRCDataset, MRCDataLoader


@dataclass(frozen=True)
class TrainConfig:
    representation: str = "2d_gated"
    batch_size: int = 16

    train_support_count: int = 16
    train_query_count: int = 64

    train_shuffle: bool = True
    beta: float = 1.0

    valid_support_count: int = 64
    valid_batch_size: int = 1

    envelope_exponent: int = 6
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 256

    cutoff: float = 5.0
    layer: int = 5

    accumulate_grad_batches: int = 1

    learning_rate: float = 1e-4
    weight_decay: float = 0
    dropout: float = 0.0
    train_n_repeats: int = 5
    val_n_repeats: int = 5
    dataloader_workers: int = 18
    gradient_clip_val: float = 1.0
    preload_dataset: bool = False
    isProd: bool = True

config = TrainConfig()

ENCODER_REGISTRY = {
    "2d": {
        "encoder_cls": GINEncoder,
        "encoder_cfg": GINEncoderConfig(config.dim, config.layer, config.cutoff, dropout=config.dropout),
    },
    "fingerprint": {
        "encoder_cls": FingerprintSimpleFeedForward,
        "encoder_cfg": FeedForwardConfig(2048, 1024, 512),
    },
    "descriptors": {
        "encoder_cls": FingerprintSimpleFeedForward,
        "encoder_cfg": FeedForwardConfig(200, 128, 512),
    },
    "fingerprint+descriptors": {
        "encoder_cls": FingerprintSimpleFeedForward,
        "encoder_cfg": FeedForwardConfig(2248, 1024, 512)
    },
    "2d_gated": {
        "encoder_cls": ResidualGatedGraphEncoder,
        "encoder_cfg": ResidualGatedGraphEncoderConfig(init_node_embedding=config.dim, n_layers=config.layer, graph_embedding=config.dim, eigen_vector_embedding=16),
    }
}

def build_encoder(representation_name: str):
    if representation_name not in ENCODER_REGISTRY:
        raise ValueError(f"Unsupported representation: {representation_name}")
    entry = ENCODER_REGISTRY[representation_name]
    encoder_cls = entry["encoder_cls"]
    encoder_cfg = entry["encoder_cfg"]
    return encoder_cls(encoder_cfg)

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")
    def update(self, value: torch.Tensor):
        self.values.append(value)
    def compute(self):
        return torch.std(dim_zero_cat(self.values), correction=0) 

metrics_table_cols = [
    'task_name','size','acc','balanced_acc','f1','prec','recall','roc_auc',
    'avg_precision','kappa','delta_auc_pr','optimistic_auc_pr','optimistic_delta_auc_pr'
]

class MRCLightningModule(L.LightningModule):
    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs.pop("config", config)
        super().__init__(*args, **kwargs)
        self.encoder = build_encoder(self.config.representation)
        self.std_dev_metric = StandardDeviationMetric()
        self.similarity_module = CNAPSProtoNetSimilarityModule(self.config.beta, True)
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def calculate_feats(self, batch):
        return self.encoder(batch)

    def calc_loss(self, input):
        batch, labels, is_query, batch_index, task_names = input
        feats = self.encoder(batch)
        print(f'Feats: {feats}')
        logits, batch_labels = self.similarity_module(feats, labels, is_query, batch_index)
        loss = self.similarity_module.calc_loss_from_logits(logits, batch_labels)
        return loss, logits, labels[is_query == 1], task_names

    def training_step(self, batches):
        loss, _, _, _ = self.calc_loss(batches)
        self.log('train_step', loss, on_step=True, on_epoch=False, batch_size=self.config.batch_size)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.config.batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def on_validation_epoch_end(self):
        std_dev = self.std_dev_metric.compute()
        self.log('delta_auc_pr_std', std_dev, on_epoch=True)
        if wandb.run is not None:
            wandb.log({'metrics_table': self.wandb_table})

    def validation_step(self, batches):
        valid_loss, logits, query_labels, task_names = self.calc_loss(batches)
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)
        
        batch_preds = self.similarity_module.get_probabilities_from_logits(logits)
        metrics = compute_binary_task_metrics(predictions=batch_preds.cpu(), labels=query_labels.detach().cpu().numpy())
        self.wandb_table.add_data(task_names[0].split('_')[0], *metrics.__dict__.values())
        self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item()))
        for k, v in metrics.__dict__.items():
            self.log(f"valid_{k}", v, on_epoch=True, on_step=False, batch_size=1)

    def test_step(self,batch,batch_idx):
        test_loss,logits,query_labels,task_names=self.calc_loss(batch)
        self.log("test_loss",test_loss,prog_bar=True,batch_size=1)
        preds=self.similarity_module.get_probabilities_from_logits(logits).cpu()
        print(f'Logits: {logits}, Preds: {preds}')
        metrics=compute_binary_task_metrics(preds,query_labels.detach().cpu().numpy())
        for k,v in metrics.__dict__.items():
            self.log(f"test_{k}",v,batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, fused=True)

# Create the test dataset and loader:
test_dataset=MRCDataset(
    "test",
    mol_type="pyg",
    n_repeats=config.val_n_repeats,
    should_preload=config.preload_dataset,
    debug=not config.isProd
)
test_dl=MRCDataLoader(
    config.representation,
    test_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=config.dataloader_workers,
    support_count=16,
    query_count=16
)

# Load the checkpointed model:
checkpoint_path="/FS-MOL/MRC_Runner/best-checkpoint-v150.ckpt"
model=MRCLightningModule.load_from_checkpoint(checkpoint_path,config=config)

# Run test:
trainer=L.Trainer(logger=WandbLogger() if config.isProd else None)
trainer.test(model,dataloaders=test_dl)

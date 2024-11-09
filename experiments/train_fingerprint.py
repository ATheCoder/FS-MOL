import sys
import os
import inspect



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from fs_mol.utils.metrics import compute_binary_task_metrics
from dataclasses import dataclass
from torch.nn import functional as F
import torch
from fs_mol.modules.gat import TrainConfig
from fs_mol.models.protonet import calculate_mahalanobis_logits
from fs_mol.data.torch_dl import FSMOLMergedTorchDataset_NOPRELOAD, FSMOLTorchDataloader_Merged
from typing import Any
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from MXMNet.model import Config, MXMNet
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
from lightning.pytorch.loggers import WandbLogger
from fs_mol.data.fsmol_task import MergedFSMOLSample
import wandb
from torch import nn




@dataclass(frozen=True)
class TrainConfig:
    # Training Settings:
    batch_size: int = 64
    train_support_count: int = 16
    train_query_count: int = 16
    train_shuffle: bool = True

    temprature: float = 0.07

    # Validation Settings:
    valid_support_count: int = 64
    valid_batch_size: int = 256

    # Model Settings:
    envelope_exponent: int = 6
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 128
    cutoff: int = 5.0
    layer: int = 5

    accumulate_grad_batches: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    dropout: float = 0.2

    encoder_dims = [128, 128, 256, 256, 512, 512]
    
    train_n_repeats = 40
    val_n_repeats = 40
    

def fingerprint_tensorizer(sample: MergedFSMOLSample):
    fingerprints = torch.tensor(sample.fingerprints, dtype=torch.float32)
    
    return fingerprints

config = TrainConfig()

model = MXMNet(
            Config(config.dim, config.layer, config.cutoff, config.encoder_dims, 512),
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            envelope_exponent=config.envelope_exponent,
            dropout=config.dropout,
        )



train_dataset = FSMOLMergedTorchDataset_NOPRELOAD("train", "pyg", config.train_n_repeats)
valid_dataset = FSMOLMergedTorchDataset_NOPRELOAD("valid", "pyg", config.val_n_repeats)

train_dl = FSMOLTorchDataloader_Merged(
    train_dataset,
    batch_size=config.batch_size,
    datatype="pyg",
    num_workers=4,
    shuffle=config.train_shuffle,
    support_count=config.train_support_count,
    query_count=config.train_query_count,
    sample_tensorizer=fingerprint_tensorizer,
    sample_batcher=torch.stack
)

valid_dls = FSMOLTorchDataloader_Merged(
    valid_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=4,
    support_count=16,
    query_count=16,
    sample_tensorizer=fingerprint_tensorizer,
    sample_batcher=torch.stack
)

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: torch.Tensor):
        self.values.append(value)

    def compute(self):
        vals = dim_zero_cat(self.values)
        return torch.std(vals, correction=0) 


class ClipLike(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = config
        super().__init__(*args, **kwargs)
        self.graph_encoder = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        )
        
        self.std_dev_metric = StandardDeviationMetric() 
        
        
    def calculate_feats(self, batch):
        encoded_graphs = self.graph_encoder(batch)
        # feats = torch.cat([encoded_graphs, batch.fingerprint.reshape(-1, 2048)], dim=1)

        return encoded_graphs

    def calc_loss(self, input):
        batch, labels, index_map = input
        feats = self.graph_encoder(batch)
        # feats = F.normalize(feats, dim=-1)

        support_feats = feats[index_map == 0]
        query_feats = feats[index_map == 1]

        support_labels = labels[index_map == 0]
        query_labels = labels[index_map == 1]
        
        logits = calculate_mahalanobis_logits(
            support_feats, support_labels, query_feats, torch.device("cuda")
        )
        loss = F.cross_entropy(logits / config.temprature, query_labels)

        return loss, logits, query_labels


    def training_step(self, batches):
        loss, _, _ = self.calc_loss(batches)
        
        self.log('train_step', loss, on_step=True, on_epoch=False, batch_size=1)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=1)
        
        return loss

    def on_validation_epoch_end(self) -> None:
        epoch_std_dev = self.std_dev_metric.compute()
        self.log('delta_auc_pr_std', epoch_std_dev, on_epoch=True)
    
    
    def validation_step(self, batches):
            valid_loss, logits, query_labels = self.calc_loss(batches)
            
            self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)

            batch_preds = F.softmax(logits, dim=1).detach().cpu().numpy()

            metrics = compute_binary_task_metrics(
                predictions=batch_preds[:, 1], labels=query_labels.detach().cpu().numpy()
            )
            
            self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item()))

            for k, v in metrics.__dict__.items():
                self.log(
                    f"valid_{k}",
                    v,
                    on_epoch=True,
                    on_step=False,
                    batch_size=1,
                )
                

    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=config.weight_decay,
            fused=True,
        )
        
checkpoint_callback = ModelCheckpoint(
    dirpath='MXM_exp_checkpoint',
    filename='best-checkpoint',
    save_top_k=2,
    verbose=True,
    monitor='valid_optimistic_delta_auc_pr',
    mode='max',
    save_last=True
)
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
cliplike = ClipLike()
wandb_enabled = True
if wandb_enabled:
    wandb.init(project="lightning_logs", config=config)
# trainer = L.Trainer(limit_train_batches=10,  check_val_every_n_epoch=1, max_epochs=5, logger=WandbLogger())
# trainer = L.Trainer(callbacks=[checkpoint_callback], limit_train_batches=10,  check_val_every_n_epoch=1, max_epochs=5, logger=WandbLogger())
trainer = L.Trainer(callbacks=[checkpoint_callback], check_val_every_n_epoch=1,logger=WandbLogger() if wandb_enabled else None, max_epochs=100, accumulate_grad_batches=config.accumulate_grad_batches)
trainer.fit(model=cliplike, train_dataloaders=train_dl, val_dataloaders=valid_dls)
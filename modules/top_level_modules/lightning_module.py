import lightning as L
import torch
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
import wandb
from dataclasses import dataclass
from typing import Any

from modules.graph_encoders.GINEncoder import GINEncoder, GINEncoderConfig
from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder, ResidualGatedGraphEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FeedForwardConfig, FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics

@dataclass(frozen=True)
class TrainConfig:
    representation: str = "2d_gated"
    batch_size: int = 32
    train_support_count: int = 6
    train_query_count: int = 16
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
    learning_rate: float = 1e-5
    weight_decay: float = 0
    dropout: float = 0.0
    train_n_repeats: int = 5
    val_n_repeats: int = 5
    dataloader_workers: int = 18
    gradient_clip_val: float = 1.0
    preload_dataset: bool = False
    isProd: bool = True

ENCODER_REGISTRY = {
    "2d": {
        "encoder_cls": GINEncoder,
        "encoder_cfg": GINEncoderConfig(256, 5, 5.0, dropout=0.0),
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
        "encoder_cfg": FeedForwardConfig(2248, 1024, 512),
    },
    "2d_gated": {
        "encoder_cls": ResidualGatedGraphEncoder,
        "encoder_cfg": ResidualGatedGraphEncoderConfig(init_node_embedding=256, n_layers=5, graph_embedding=256, eigen_vector_embedding=16),
    }
}

def build_encoder(representation_name: str):
    if representation_name not in ENCODER_REGISTRY:
        raise ValueError(f"Unsupported representation: {representation_name}")
    entry = ENCODER_REGISTRY[representation_name]
    return entry["encoder_cls"](entry["encoder_cfg"])

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
    'avg_precision','kappa','delta_auc_pr','optimistic_auc_pr','optimistic_delta_auc_pr', 'number_of_half'
]

class MRCLightningModule(L.LightningModule):
    def __init__(self, config: TrainConfig, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs.pop("config", config)
        super().__init__(*args, **kwargs)
        self.encoder = build_encoder(self.config.representation)
        self.similarity_module = CNAPSProtoNetSimilarityModule(self.config.beta, True)
        self.std_dev_metric = StandardDeviationMetric()
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def calculate_feats(self, batch):
        return self.encoder(batch)

    def calc_loss(self, input):
        batch, labels, is_query, batch_index, task_names = input
        feats = self.encoder(batch)
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

    def validation_step(self, batches):
        valid_loss, logits, query_labels, task_names = self.calc_loss(batches)
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)
        preds = self.similarity_module.get_probabilities_from_logits(logits)
        metrics = compute_binary_task_metrics(preds.cpu(), query_labels.cpu().numpy())
        self.wandb_table.add_data(task_names[0].split('_')[0], *metrics.__dict__.values())
        self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item()))
        for k, v in metrics.__dict__.items():
            self.log(f"valid_{k}", v, on_epoch=True, on_step=False, batch_size=1)

    def on_validation_epoch_end(self):
        std_dev = self.std_dev_metric.compute()
        self.log('delta_auc_pr_std', std_dev, on_epoch=True)
        if wandb.run is not None:
            wandb.log({'metrics_table': self.wandb_table})

    def test_step(self,batch,batch_idx):
        test_loss, logits, query_labels, task_names = self.calc_loss(batch)
        self.log("test_loss", test_loss, prog_bar=True, batch_size=1)
        preds = self.similarity_module.get_probabilities_from_logits(logits).cpu()
        metrics = compute_binary_task_metrics(preds, query_labels.cpu().numpy())
        for k,v in metrics.__dict__.items():
            self.log(f"test_{k}", v, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay, fused=True)

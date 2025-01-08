import torch
import lightning as L
import wandb
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
from typing import Any

from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder, ResidualGatedGraphEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FingerprintSimpleFeedForward, FeedForwardConfig
from fs_mol.utils.metrics import compute_binary_task_metrics

metrics_table_cols = [
    'task_name', 'size', 'acc', 'balanced_acc', 'f1', 'prec', 'recall', 'roc_auc',
    'avg_precision', 'kappa', 'delta_auc_pr', 'optimistic_auc_pr', 'optimistic_delta_auc_pr'
]

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: torch.Tensor):
        self.values.append(value)

    def compute(self):
        return torch.std(dim_zero_cat(self.values), correction=0)

class MRCLightningModule(L.LightningModule):
    def __init__(self, config: dict, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs.pop("config", config)

        self.REPR_TO_ENCODER_MAP = {
            "2d": ResidualGatedGraphEncoder,
            "fingerprint": FingerprintSimpleFeedForward,
            "fingerprint+descriptors": FingerprintSimpleFeedForward
        }

        self.REPR_ENCODER_CONFIG_MAP = {
            "2d": ResidualGatedGraphEncoderConfig(init_node_embedding=config['dim'], n_layers=config['layer'], graph_embedding=config['dim'], eigen_vector_embedding=16),
            "fingerprint": FeedForwardConfig(2048, 1024, 512),
            "fingerprint+descriptors": FeedForwardConfig(2248, 1024, 512)
        }

        super().__init__(*args, **kwargs)

        self.encoder = self.REPR_TO_ENCODER_MAP[self.config['representation']](
            self.REPR_ENCODER_CONFIG_MAP[self.config['representation']]
        )

        self.similarity_module = CNAPSProtoNetSimilarityModule(self.config['beta'], True)

        self.std_dev_metric = StandardDeviationMetric()
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def calc_loss(self, batch):
        batch_data, labels, is_query, batch_index, task_names = batch
        batch_data = torch.nan_to_num(batch_data, nan=0.0)
        feats = self.encoder(batch_data)
        logits, batch_labels = self.similarity_module(feats, labels, is_query, batch_index)
        loss = self.similarity_module.calc_loss_from_logits(logits, batch_labels)
        return loss, logits, labels[is_query == 1], task_names

    def training_step(self, batch):
        loss, _, _, _ = self.calc_loss(batch)
        self.log('train_step', loss, on_step=True, on_epoch=False, batch_size=self.config['batch_size'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.config['batch_size'])
        return loss

    def on_validation_epoch_start(self):
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def on_validation_epoch_end(self):
        std_dev = self.std_dev_metric.compute()
        self.log('delta_auc_pr_std', std_dev, on_epoch=True)
        if wandb.run is not None:
            wandb.log({'metrics_table': self.wandb_table})

    def validation_step(self, batch):
        valid_loss, logits, query_labels, task_names = self.calc_loss(batch)
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)
        batch_preds = self.similarity_module.get_probabilities_from_logits(logits)
        metrics = compute_binary_task_metrics(predictions=batch_preds.cpu(), labels=query_labels.detach().cpu().numpy())
        self.wandb_table.add_data(task_names[0].split('_')[0], *metrics.__dict__.values())
        self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item()))
        for k, v in metrics.__dict__.items():
            self.log(f"valid_{k}", v, on_epoch=True, on_step=False, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            fused=True
        )

import lightning as L
import torch
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
import wandb
from dataclasses import dataclass, field
from typing import Any

from MXMNet.model import MXMNet
from modules.graph_encoders.GINEncoder import GINEncoder, GINEncoderConfig
from modules.graph_encoders.PAMNet import Config, PAMNet
from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder, ResidualGatedGraphEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FeedForwardConfig, FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics

@dataclass
class GINHyperparams:
    dim: int = 256
    layer: int = 5
    cutoff: float = 5.0
    dropout: float = 0.0

@dataclass
class GatedHyperparams:
    init_node_embedding: int = 256
    n_layers: int = 5
    graph_embedding: int = 256
    eigen_vector_embedding: int = 16

@dataclass
class FingerprintHyperparams:
    input_dim: int = 2048
    hidden_dim1: int = 1024
    hidden_dim2: int = 512

@dataclass
class DescriptorsHyperparams:
    input_dim: int = 200
    hidden_dim1: int = 128
    hidden_dim2: int = 512

@dataclass
class FingerprintDescriptorsHyperparams:
    input_dim: int = 2248
    hidden_dim1: int = 1024
    hidden_dim2: int = 512

@dataclass
class MXMGraphEncoderConfig:
    dim: int
    n_layer: int
    cutoff: float
    mol_aggr: str | None
    mol_aggr_config: dict = field(default_factory=dict)
    dropout: float = 0.0

@dataclass
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
    
    # Per-encoder hyperparams, so we can tune them if needed:
    gin_hp: GINHyperparams = field(default_factory=GINHyperparams)
    gated_hp: GatedHyperparams = field(default_factory=GatedHyperparams)
    fingerprint_hp: FingerprintHyperparams = field(default_factory=FingerprintHyperparams)
    descriptors_hp: DescriptorsHyperparams = field(default_factory=DescriptorsHyperparams)
    fp_desc_hp: FingerprintDescriptorsHyperparams = field(default_factory=FingerprintDescriptorsHyperparams)

def build_encoder_cfg(config: TrainConfig):
    """
    Given the top-level config (with sub-configs),
    returns the correct encoder config for the chosen representation.
    """
    if config.representation == "2d":
        hp = config.gin_hp
        return GINEncoderConfig(
            cutoff=hp.cutoff,
            dropout=config.dropout,
            dim=config.dim,
            n_layer=config.layer,
        )
    elif config.representation == "2d_gated":
        hp = config.gated_hp
        return ResidualGatedGraphEncoderConfig(
            init_node_embedding=hp.init_node_embedding,
            n_layers=config.layer,
            graph_embedding=config.dim,
            eigen_vector_embedding=hp.eigen_vector_embedding,
        )
    elif config.representation == "fingerprint":
        hp = config.fingerprint_hp
        return FeedForwardConfig(hp.input_dim, hp.hidden_dim1, hp.hidden_dim2)
    elif config.representation == "descriptors":
        hp = config.descriptors_hp
        return FeedForwardConfig(hp.input_dim, hp.hidden_dim1, hp.hidden_dim2)
    elif config.representation == "fingerprint+descriptors":
        hp = config.fp_desc_hp
        return FeedForwardConfig(hp.input_dim, hp.hidden_dim1, hp.hidden_dim2)
    elif config.representation == '3d_mxm':
        return MXMGraphEncoderConfig(
            dim=config.dim,
            n_layer=config.layer,
            cutoff=config.cutoff,
            mol_aggr=None,
            mol_aggr_config={},
            dropout=config.dropout
        )
    elif config.representation == '3d_pam':
        return Config(
            dataset='qm9',
            dim=config.dim,
            cutoff_l=config.cutoff,
            cutoff_g=config.cutoff,
            n_layer=config.layer,
        )
    else:
        raise ValueError(f"Unsupported representation: {config.representation}")

def build_encoder(config: TrainConfig):
    """
    Construct the encoder from the top-level config,
    using the result of build_encoder_cfg.
    """
    if config.representation == "2d":
        return GINEncoder(build_encoder_cfg(config))
    elif config.representation == "2d_gated":
        return ResidualGatedGraphEncoder(build_encoder_cfg(config))
    elif config.representation in ["fingerprint", "descriptors", "fingerprint+descriptors"]:
        return FingerprintSimpleFeedForward(build_encoder_cfg(config))
    elif config.representation == '3d_mxm':
        return MXMNet(build_encoder_cfg(config))
    elif config.representation == '3d_pam':
        return PAMNet(build_encoder_cfg(config))
    else:
        raise ValueError(f"Unsupported representation: {config.representation}")


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
    def __init__(self, config: TrainConfig|dict, *args: Any, **kwargs: Any) -> None:
        if isinstance(config, dict):
            config = TrainConfig(**config)
        self.config = kwargs.pop("config", config)
        super().__init__(*args, **kwargs)
        
        # Build the right encoder for the chosen representation
        self.encoder = build_encoder(self.config)
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

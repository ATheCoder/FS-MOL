from dataclasses import dataclass, field
import math
from typing import Literal
import torch
from torch.optim.optimizer import Optimizer
import wandb


from pytorch_lightning.utilities.grads import grad_norm

from lightning.pytorch import LightningModule
from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder
from modules.similarity_modules import (
    CNAPSProtoNetSimilarityModule,
    CosineWeightedMeanSimilarity,
    SingleBatch_CNAPSProtoNetSimilarityModule,
)
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.utils import unbatch

from utils.batch import batch_variable_length, generate_mask


@dataclass(frozen=True)
class GenericMXM3DGraphConfig:
    project_name: str = "Tarjan+MXMNet"
    # Training Settings:
    batch_size: int = 8
    train_support_count: int = 16
    train_query_count: int = 8
    train_shuffle: bool = True

    # Validation Settings:
    valid_support_size: int = 16
    valid_query_size: int = 32

    num_encoder_blocks: int = 2

    mol_aggr: str = "last"
    # mol_aggr_config = dict(dim=128)
    mol_aggr_config: dict = field(default_factory=dict)
    # mol_aggr_config = dict(in_dim=128, out_dim=1024, n_layers=8)

    temprature: float = 0.07

    # Model Settings:
    envelope_exponent: int = 5
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 256
    cutoff: int = 4
    layer: int = 5

    # Training Settings:
    accumulate_grad_batches: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0
    padding_size = 12
    prediction_scaling = 0.325

    attn_dropout = 0.0

    eigen_vec_dim = None

    dropout: float = 0.0
    similarity_module: Literal[
        "batch_protonet", "fsmol_protonet", "cosine_similarity"
    ] = "batch_protonet"


class GenericMXM3DGraphLightningModule(LightningModule):
    def __init__(self, config: GenericMXM3DGraphConfig, data_module=None) -> None:
        super().__init__()
        self.data_module = data_module
        self.config = config

        # self.graph_encoder = GenericMXMNet_NoLocalInfo(
        #     GenericMXMNet_NoLocalInfo_Config(
        #         dim=config.dim,
        #         cutoff=config.cutoff,
        #         n_layer=config.layer,
        #         attn_dropout=config.attn_dropout,
        #     )
        # )

        self.graph_encoder = ResidualGatedGraphEncoder(256, config.eigen_vec_dim, 256, 16)
        # self.graph_encoder = AdaptedDimeNetPP(
        #     hidden_channels=config.dim,
        #     num_blocks=config.layer,
        #     int_emb_size=64,
        #     basis_emb_size=8,
        #     out_emb_channels=256,
        #     num_spherical=config.num_spherical,
        #     num_radial=config.num_radial,
        #     cutoff=config.cutoff,
        #     max_num_neighbors=500,
        #     envelope_exponent=config.envelope_exponent,
        #     num_before_skip=1,
        #     num_after_skip=2,
        # )

        # self.graph_encoder = MXMNet(
        #     MXMNetConfig(
        #         config.dim,
        #         config.layer,
        #         config.cutoff,
        #         mol_aggr=config.mol_aggr,
        #         mol_aggr_config=config.mol_aggr_config,
        #         dropout=config.dropout,
        #     )
        # )

        self.validation_step_output = []
        self.train_step_output = []
        self.task_names = []
        self.validation_step_loss = []

        if config.similarity_module == "fsmol_protonet":
            self.similarity_module = SingleBatch_CNAPSProtoNetSimilarityModule()
        if config.similarity_module == "batch_protonet":
            self.similarity_module = CNAPSProtoNetSimilarityModule(config.prediction_scaling)

        if config.similarity_module == "cosine_similarity":
            self.similarity_module = CosineWeightedMeanSimilarity(config.prediction_scaling)

    def get_support_query(self, input_tensor, is_query_index):
        support_indices = (is_query_index == 0).nonzero().squeeze(1)
        query_indices = (is_query_index == 1).nonzero().squeeze(1)

        return input_tensor[support_indices], input_tensor[query_indices]

    def norm_tensor(self, tensor):
        norms = tensor.norm(dim=-1, keepdim=True)
        mask = norms > 0
        return tensor * mask / (norms + ~mask)

    def encode_graphs(self, input_graphs):
        num_graphs = input_graphs.num_graphs
        num_nodes = input_graphs.x.shape[0]

        mean_node_count = num_nodes / num_graphs

        self.log("mean_node_count", mean_node_count)

        subgraph_reprs = self.graph_encoder(input_graphs)

        return subgraph_reprs

    def get_query_labels(self, labels, is_query, batch_index):
        if batch_index.max() < 1:
            mask = is_query == 1
            query_labels = labels[mask]

            _, query_batch_index = torch.unique(batch_index[mask], return_inverse=True)

            return torch.stack(unbatch(query_labels, query_batch_index), dim=0)

        mask = is_query == 1
        query_labels = labels[mask]

        _, query_batch_index = torch.unique(batch_index[mask], return_inverse=True)

        batch, lengths = batch_variable_length(unbatch(query_labels, query_batch_index))

        return batch, generate_mask(lengths)

    def training_step(self, batch):
        input_graphs, is_query, labels, batch_index = (
            batch["graphs"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )
        graph_representations = self.encode_graphs(input_graphs)

        logits = self.similarity_module(graph_representations, labels, is_query, batch_index)

        query_labels, mask = self.get_query_labels(labels, is_query, batch_index)

        loss = self.similarity_module.calc_loss_from_logits(logits[mask], query_labels[mask])
        self.log("loss", loss, on_step=True, on_epoch=False, batch_size=self.config.batch_size)
        self.log(
            "loss_per_epoch", loss, on_epoch=True, on_step=False, batch_size=self.config.batch_size
        )

        self.task_names = self.task_names + batch["tasks"]

        return loss

    def on_train_epoch_end(self) -> None:
        wandb.log(
            {"tasks": wandb.Table(columns=["task_name"], data=[[t] for t in self.task_names])}
        )
        self.task_names.clear()

    def validation_step(self, batch, batch_idx):
        input_graphs, is_query, labels, batch_index = (
            batch["graphs"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )

        input_graphs, is_query, labels, batch_index = (
            batch["graphs"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )
        graph_representations = self.encode_graphs(input_graphs)

        logits = self.similarity_module(graph_representations, labels, is_query, batch_index)

        query_labels = self.get_query_labels(labels, is_query, batch_index)

        return {
            "logits": logits,
            "query_labels": query_labels,
            "task_names": batch["tasks"],
        }

        auc_pr = self.similarity_module.calculate_delta_auc_pr(logits, query_labels)
        loss = self.similarity_module.calc_loss_from_logits(logits, query_labels)

        self.validation_step_output.append(auc_pr)
        self.validation_step_loss.append(loss.detach().cpu().numpy())

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    # def on_validation_epoch_end(self) -> None:
    #     mean_delta_auc_pr = np.mean(self.validation_step_output)
    #     mean_train_delta_auc_pr = np.mean(self.train_step_output)
    #     mean_loss = np.mean(self.validation_step_loss)
    #     self.log("mean_delta_auc_pr", mean_delta_auc_pr)
    #     self.log("train_mean_delta_auc_pr", mean_train_delta_auc_pr)
    #     self.log("valid_mean_loss", mean_loss)

    #     self.validation_step_output.clear()
    #     self.train_step_output.clear()
    #     self.validation_step_loss.clear()

    def configure_optimizers(self):
        optm = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if True:
            return optm
        initial_lr = 1e-5
        epochs_per_step = 10_000
        decay_factor = 0.5

        def smooth_decay(epoch):
            return math.pow(math.pow(decay_factor, 1.0 / epochs_per_step), epoch % epochs_per_step)

        scheduler = LambdaLR(optm, lr_lambda=smooth_decay)

        return {"optimizer": optm, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self.graph_encoder, norm_type=2)

        self.log_dict(norms)

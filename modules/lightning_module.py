from dataclasses import dataclass, field
from typing import Literal

import torch

from modules.graph_encoders.MXMNet import MXMNet, MXMNetConfig
from torch import nn

import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from torch.optim.optimizer import Optimizer
from MHNfs.mhnfs.modules import similarity_module

from modules.graph_encoders.MXMNet import MXMNet, MXMNetConfig

from pytorch_lightning.utilities.grads import grad_norm
import torch
from torch.nn import functional as F
import lightning as pl


from torch import nn
from torch_geometric.nn import SetTransformerAggregation
from torch_geometric.utils import unbatch
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class TarjanTrainingConfig:
    project_name: str = "Tarjan+MXMNet"
    # Training Settings:
    batch_size: int = 64
    train_support_count: int = 16
    train_query_count: int = 1
    train_shuffle: bool = True

    mol_aggr: str = "last"
    # mol_aggr_config = dict(dim=128)
    mol_aggr_config: dict = field(default_factory=dict)
    # mol_aggr_config = dict(in_dim=128, out_dim=1024, n_layers=8)

    temprature: float = 0.07

    # Validation Settings:
    valid_support_count: int = 64
    valid_batch_size: int = 256

    # Model Settings:
    envelope_exponent: int = 6
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 128
    cutoff: int = 4
    layer: int = 5

    accumulate_grad_batches: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0
    padding_size = 12
    prediction_scaling = 0.325

    dropout: float = 0.0

    encoder_dims = [128, 128, 256, 256, 512, 512]


class TarjanLightningModule(pl.LightningModule):
    def __init__(
        self,
        dim: int,
        layer: int,
        cutoff: float,
        mol_aggr: Literal["sum_norm", "concat", "last", "concat_norm_lin", "last_norm"] | None,
        mol_aggr_config: dict | None,
        dropout: float,
        padding_size: int,
        prediction_scaling: float,
        batch_size: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.graph_encoder = MXMNet(
            MXMNetConfig(
                dim,
                layer,
                cutoff,
                mol_aggr=mol_aggr,
                mol_aggr_config=mol_aggr_config,
                dropout=dropout,
            )
        )

        self.subgraph_set_transformer = SetTransformerAggregation(dim, heads=8, layer_norm=True)

        self.padding_size = padding_size

        self.register_parameter(
            "prediction_scaling",
            nn.Parameter(torch.ones([]) * np.log(1 / prediction_scaling)),
        )

        self.validation_step_output = []
        self.train_step_output = []
        self.validation_step_loss = []

    def get_support_query(self, input_tensor, is_query_index):
        support_indices = (is_query_index == 0).nonzero().squeeze(1)
        query_indices = (is_query_index == 1).nonzero().squeeze(1)

        return input_tensor[support_indices], input_tensor[query_indices]

    def select_batch(self, input_tensor, batch_index, batch_no):
        current_batch_indices = (batch_index == batch_no).nonzero().squeeze(1)

        return input_tensor[current_batch_indices]

    def norm_tensor(self, tensor):
        norms = tensor.norm(dim=-1, keepdim=True)
        mask = norms > 0
        return tensor * mask / (norms + ~mask)

    def get_support_with_label(self, graph_reprs, labels, is_query, batch_index, label):
        mask = (is_query == 0) & (labels == label)

        support_positive_graphs = graph_reprs[mask]
        support_positive_batch_index = batch_index[mask]
        support_positive_graphs = unbatch(support_positive_graphs, support_positive_batch_index)
        support_positive_lenghts = torch.tensor(
            [g.shape[0] for g in support_positive_graphs], dtype=torch.long, device=self.device
        )
        support_positive_graphs = pad_sequence(support_positive_graphs, batch_first=True)

        return support_positive_graphs, support_positive_lenghts

    def separate_qsl(self, graph_reprs, labels, is_query, batch_index):
        # Support Vectors Positive, Support Labels
        support_negative_graphs, support_negative_lengths = self.get_support_with_label(
            graph_reprs, labels, is_query, batch_index, 0
        )
        support_positive_graphs, support_positive_lengths = self.get_support_with_label(
            graph_reprs, labels, is_query, batch_index, 1
        )
        # Query Vectors, Query Labels
        query_graphs = graph_reprs[is_query == 1]
        query_labels = labels[is_query == 1]
        query_batch_index = batch_index[is_query == 1]

        query_graphs = torch.stack(unbatch(query_graphs, query_batch_index), dim=0)
        query_labels = torch.stack(unbatch(query_labels, query_batch_index), dim=0)

        return (
            support_negative_graphs,
            support_negative_lengths,
            support_positive_graphs,
            support_positive_lengths,
            query_graphs,
            query_labels,
        )

    def get_logits_with_attn(
        self, support_pos, support_pos_sizes, support_neg, support_neg_sizes, query
    ):
        # Batch, Queries
        # Be nazar miad ke data dare beyne batch ha share mishe.

        if True:
            support_pos = self.norm_tensor(support_pos)
            support_neg = self.norm_tensor(support_neg)

        pos_vote = similarity_module(
            query,
            support_pos,
            support_pos_sizes,
        )
        neg_vote = similarity_module(
            query,
            support_neg,
            support_neg_sizes,
        )

        self.log("mean_pre_sigmoid", torch.mean((pos_vote - neg_vote)))
        logit_scale = self.prediction_scaling.exp()
        self.log("logit_scale", logit_scale)

        logits = (pos_vote - neg_vote) * logit_scale

        return logits

    def encode_graphs(self, input_graphs, batch_index):
        subgraph_reprs = self.graph_encoder(input_graphs)
        graph_representations = self.subgraph_set_transformer(subgraph_reprs, batch_index)

        return graph_representations

    def training_step(self, batch):
        input_graphs, substructure_mol_index, is_query, labels, batch_index = (
            batch["substructures"],
            batch["substructure_mol_index"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )
        graph_representations = self.encode_graphs(input_graphs, substructure_mol_index)
        (
            support_neg,
            support_neg_sizes,
            support_pos,
            support_pos_sizes,
            query_graphs,
            query_labels,
        ) = self.separate_qsl(graph_representations, labels, is_query, batch_index)

        logits = self.get_logits_with_attn(
            support_pos, support_pos_sizes, support_neg, support_neg_sizes, query_graphs
        )

        loss = F.binary_cross_entropy_with_logits(logits, query_labels.float())

        self.log("loss", loss, on_step=True, on_epoch=False, batch_size=self.batch_size)
        self.log("loss_per_epoch", loss, on_epoch=True, on_step=False, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        input_graphs, substructure_mol_index, is_query, labels, batch_index = (
            batch["substructures"],
            batch["substructure_mol_index"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )

        print(batch["tasks"])

        graph_representations = self.encode_graphs(input_graphs, substructure_mol_index)
        (
            support_neg,
            support_neg_sizes,
            support_pos,
            support_pos_sizes,
            query_graphs,
            query_labels,
        ) = self.separate_qsl(graph_representations, labels, is_query, batch_index)

        logits = self.get_logits_with_attn(
            support_pos, support_pos_sizes, support_neg, support_neg_sizes, query_graphs
        )

        auc_pr = self.calculate_delta_auc_pr(logits, query_labels)
        loss = F.binary_cross_entropy_with_logits(logits, query_labels.float())

        self.validation_step_output.append(auc_pr)
        self.validation_step_loss.append(loss.detach().cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        mean_delta_auc_pr = np.mean(self.validation_step_output)
        mean_train_delta_auc_pr = np.mean(self.train_step_output)
        mean_loss = np.mean(self.validation_step_loss)
        self.log("mean_delta_auc_pr", mean_delta_auc_pr)
        self.log("train_mean_delta_auc_pr", mean_train_delta_auc_pr)
        self.log("valid_mean_loss", mean_loss)

        self.validation_step_output.clear()
        self.train_step_output.clear()
        self.validation_step_loss.clear()

    def calculate_delta_auc_pr(self, batch_logits, batch_targets):
        predictions = F.sigmoid(batch_logits.reshape(-1))
        targets = batch_targets.reshape(-1)
        try:
            precision, recall, _ = precision_recall_curve(
                targets.detach().cpu().numpy(), predictions.detach().cpu().numpy()
            )

            auc_score = auc(recall, precision)

            random_classifier_auc_pr = np.mean(targets.detach().cpu().numpy())
            res = auc_score - random_classifier_auc_pr

            return res
        except ValueError:
            print("Got Value error!")
            return -1

    def configure_optimizers(self):
        optm = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        return optm

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self.graph_encoder, norm_type=2)

        self.log_dict(norms)

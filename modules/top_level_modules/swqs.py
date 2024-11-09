from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
import torch
from torch.optim.optimizer import Optimizer
import wandb
from MHNfs.mhnfs.modules import similarity_module

from modules.graph_encoders.MXMNet import MXMNet, MXMNetConfig

from pytorch_lightning.utilities.grads import grad_norm
from modules import SetTransformerSimilarityModule


from torch import Tensor, nn
from torch_geometric.nn import SetTransformerAggregation
from torch_geometric.utils import unbatch
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch import LightningModule
from torch.nn import functional as F


@dataclass(frozen=True)
class SWQSTrainingConfig:
    project_name: str = "Tarjan+MXMNet"
    # Training Settings:
    batch_size: int = 64
    train_support_count: int = 16
    train_query_count: int = 1
    train_shuffle: bool = True
    num_encoder_blocks: int = 2
    num_decoder_blocks: int = 2

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
    dim: int = 512
    cutoff: int = 4
    layer: int = 5

    accumulate_grad_batches: int = 4
    learning_rate: float = 0.00001
    weight_decay: float = 0
    padding_size = 12
    prediction_scaling = 0.0187

    dropout: float = 0.0

    encoder_dims = [128, 128, 256, 256, 512, 512]


class SWQSLightningModule(LightningModule):
    def __init__(self, config: SWQSTrainingConfig, data_module=None) -> None:
        super().__init__()
        self.data_module = data_module
        self.config = config

        self.graph_encoder = MXMNet(
            MXMNetConfig(
                config.dim,
                config.layer,
                config.cutoff,
                mol_aggr=config.mol_aggr,
                mol_aggr_config=config.mol_aggr_config,
                dropout=config.dropout,
            )
        )

        self.subgraph_set_transformer = SetTransformerAggregation(
            config.dim,
            heads=8,
            num_encoder_blocks=self.config.num_encoder_blocks,
            num_decoder_blocks=self.config.num_decoder_blocks,
        )

        self.query_set_transformer = SetTransformerAggregation(
            config.dim,
            heads=8,
            num_encoder_blocks=self.config.num_encoder_blocks,
            num_decoder_blocks=self.config.num_decoder_blocks,
        )

        self.padding_size = config.padding_size

        self.prediction_scaling: Tensor
        self.register_parameter(
            "prediction_scaling",
            nn.Parameter(torch.ones([]) * np.log(1 / config.prediction_scaling)),
        )

        self.validation_step_output = []
        self.train_step_output = []
        self.task_names = []
        self.validation_step_loss = []

        self.similarity_module = SetTransformerSimilarityModule(config.dim, 8, True, 0, 1, 1)

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
        assert query.shape[0] == support_pos.shape[0]
        assert query.shape[0] == support_neg.shape[0]
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

    def encode_graphs(self, input_graphs):
        subgraph_reprs = self.graph_encoder(input_graphs)
        return subgraph_reprs

    def training_step(self, batch):
        input_graphs, substructure_mol_index, is_query, labels, batch_index = (
            batch["substructures"],
            batch["substructure_mol_index"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )
        logits = self.get_logits(
            input_graphs, substructure_mol_index, is_query, labels, batch_index
        )

        query_labels = labels[is_query == 1]

        loss = F.cross_entropy(logits, query_labels)

        self.log("loss", loss, on_step=True, on_epoch=False, batch_size=self.config.batch_size)
        self.log(
            "loss_per_epoch", loss, on_epoch=True, on_step=False, batch_size=self.config.batch_size
        )

        self.task_names = self.task_names + batch["tasks"]

        return loss

    def adaptiveIndexer(self, tensor):
        unique_values, inverse_indices = torch.unique(tensor, return_inverse=True)
        if torch.all(unique_values == torch.arange(len(unique_values), device=tensor.device)):
            return tensor, unique_values
        else:
            return inverse_indices, unique_values

    def get_logits(self, input_graphs, substructure_mol_index, is_query, labels, batch_index):
        graph_representations = self.encode_graphs(input_graphs)

        negative_subgraph_indices = (labels[substructure_mol_index] == 0) & (
            is_query[substructure_mol_index] == 0
        )
        positive_subgraph_indices = (labels[substructure_mol_index] == 1) & (
            is_query[substructure_mol_index] == 0
        )

        negative_subgraphs = graph_representations[negative_subgraph_indices]
        positive_subgraphs = graph_representations[positive_subgraph_indices]

        negative_subgraph_batch_index = batch_index[substructure_mol_index][
            negative_subgraph_indices
        ]
        positive_subgraph_batch_index = batch_index[substructure_mol_index][
            positive_subgraph_indices
        ]

        negative_prototypes = self.subgraph_set_transformer(
            negative_subgraphs, negative_subgraph_batch_index
        ).nan_to_num()

        positive_prototypes = self.subgraph_set_transformer(
            positive_subgraphs, positive_subgraph_batch_index
        ).nan_to_num()

        support_prototypes = torch.stack([negative_prototypes, positive_prototypes], dim=1)

        query_subgraph_indices = is_query[substructure_mol_index] == 1

        a, b = self.adaptiveIndexer(substructure_mol_index[query_subgraph_indices])
        query_prototypes_unbatched = self.query_set_transformer(
            graph_representations[query_subgraph_indices],
            a,
        )

        query_prototypes = torch.stack(unbatch(query_prototypes_unbatched, batch_index[b]), dim=0)

        logits = torch.bmm(query_prototypes, support_prototypes.transpose(1, 2)).squeeze()
        return logits

    def on_train_epoch_end(self) -> None:
        wandb.log(
            {"tasks": wandb.Table(columns=["task_name"], data=[[t] for t in self.task_names])}
        )
        self.task_names.clear()

    def validation_step(self, batch, batch_idx):
        input_graphs, substructure_mol_index, is_query, labels, batch_index = (
            batch["substructures"],
            batch["substructure_mol_index"],
            batch["is_query"],
            batch["labels"],
            batch["batch_index"],
        )

        print(batch["tasks"])

        logits = self.get_logits(
            input_graphs, substructure_mol_index, is_query, labels, batch_index
        )

        query_labels = labels[is_query == 1]
        # logits = self.get_logits_with_attn(
        #     support_pos, support_pos_sizes, support_neg, support_neg_sizes, query_graphs
        # )

        auc_pr = self.calculate_delta_auc_pr(logits, query_labels)
        loss = F.cross_entropy(logits, query_labels)

        self.validation_step_output.append(auc_pr)
        self.validation_step_loss.append(loss.detach().cpu().numpy())

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

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
        predictions = F.softmax(batch_logits, dim=-1)[:, 1]
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
            # [{"params": no_decay, 'weight_decay': 0}, {"params": with_decay, "weight_decay": config.weight_decay}],
            self.parameters(),
            lr=self.config.learning_rate,
            # lr=config.learning_rate,
            # amsgrad=True,
            weight_decay=self.config.weight_decay,
            # fused=True,
        )

        return optm
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optm, base_lr=1e-7, max_lr=1e-3, step_size_up=2 * 6000, cycle_momentum=False
        # )

        # scheduler = torch.optim.lr_scheduler.LinearLR(optm, start_factor=0.1, total_iters=3000)

        # return {"optimizer": optm, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self.graph_encoder, norm_type=2)

        self.log_dict(norms)

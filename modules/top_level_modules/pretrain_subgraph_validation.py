from lightning import LightningModule
from modules.configs.PretrainConfig import PretrainGatedGraphConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule, CosineWeightedMeanSimilarity, SingleBatch_CNAPSProtoNetSimilarityModule
import torch
from torch_geometric.utils import unbatch
import wandb
import numpy as np

class PretrainSubgraphValidator(LightningModule):
    def __init__(self, config: PretrainGatedGraphConfig, graph_encoder, subgraph_aggregator) -> None:
        super().__init__()
        self.config = config

        self.graph_encoder = graph_encoder

        self.subgraph_aggregator = subgraph_aggregator

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
        mask = is_query == 1
        query_labels = labels[mask]

        _, query_batch_index = torch.unique(batch_index[mask], return_inverse=True)

        return torch.stack(unbatch(query_labels, query_batch_index), dim=0)

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
        subgraph_representations = self.graph_encoder(input_graphs)
        graph_representations = self.subgraph_aggregator(
            subgraph_representations, substructure_mol_index
        )

        logits = self.similarity_module(graph_representations, labels, is_query, batch_index)

        query_labels = self.get_query_labels(labels, is_query, batch_index)

        auc_pr = self.similarity_module.calculate_delta_auc_pr(logits, query_labels)
        loss = self.similarity_module.calc_loss_from_logits(logits, query_labels)

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

    # def configure_optimizers(self):
    #     optm = torch.optim.Adam(
    #         self.parameters(),
    #         lr=self.config.learning_rate,
    #         weight_decay=self.config.weight_decay,
    #     )

    #     return optm

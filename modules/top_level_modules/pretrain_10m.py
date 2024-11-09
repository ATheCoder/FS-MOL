from lightning import LightningModule
import numpy as np
import torch

from modules.graph_modules.residual_gated_graph import ResidualGatedGraphEncoder
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch.nn.functional as F
from torch import nn
from modules.similarity_modules import CNAPSProtoNetSimilarityModule, CosineWeightedMeanSimilarity, SingleBatch_CNAPSProtoNetSimilarityModule
from torch_geometric.utils import unbatch


def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss

class PretrainGatedGraph(LightningModule):
    def __init__(self, config, data_module) -> None:
        super().__init__()
        self.config = config

        self.graph_encoder = ResidualGatedGraphEncoder(config.dim, None, config.attn_dim, config.n_layers)
        self.subgraph_aggregator = SetTransformerAggregation(
            config.attn_dim, heads=config.aggregator_heads, num_encoder_blocks=config.aggregator_encoder_blocks, num_decoder_blocks=config.aggregator_decoder_blocks, layer_norm=True
        )
        
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        
        # Validation Logic:
        
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
        
    def clip_loss(self, x1, x2):
        logit_scale = self.logits_scale.exp()
        
        
        logits = x1 @ x2.T * logit_scale
         
        targets = torch.arange(logits.shape[0], device=x1.device)
        x1_loss = F.cross_entropy(logits, targets)
        x2_loss = F.cross_entropy(logits.T, targets)
        
        loss = (x1_loss + x2_loss) / 2.0
        
        return loss.mean()
    
    def get_query_labels(self, labels, is_query, batch_index):
        mask = is_query == 1
        query_labels = labels[mask]

        _, query_batch_index = torch.unique(batch_index[mask], return_inverse=True)

        return torch.stack(unbatch(query_labels, query_batch_index), dim=0)
    
    def training_step(self, batch):
        graphs, subgraph_graph_index, mol_size = batch['graphs'], batch['subgraph_graph_index'], batch['mol_size']
        graph_reprs = self.graph_encoder(graphs)
        
        subgraph_reprs = graph_reprs[:-mol_size]
        
        molecule_reprs = graph_reprs[-mol_size:]
        
        calculated_molecule_reprs = self.subgraph_aggregator(subgraph_reprs, subgraph_graph_index)
        
        calculated_molecule_reprs = calculated_molecule_reprs / calculated_molecule_reprs.norm(dim=1, keepdim=True)
        molecule_reprs = molecule_reprs / molecule_reprs.norm(dim=1, keepdim=True)
        
        logit_scale = self.logits_scale.exp()
        
        logits_per_calculated_molecule_reprs = logit_scale * calculated_molecule_reprs @ molecule_reprs.t()
        logits_per_molecule_reprs = logits_per_calculated_molecule_reprs.t()
        
        targets = torch.arange(logits_per_molecule_reprs.shape[0], dtype=torch.long, device=self.device)
        
        loss = (F.cross_entropy(logits_per_molecule_reprs, targets) + F.cross_entropy(logits_per_calculated_molecule_reprs, targets)) / 2
        
        self.log('loss', loss)
        
        return loss
    
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
    
    
    def configure_optimizers(self):
        optm = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        return optm
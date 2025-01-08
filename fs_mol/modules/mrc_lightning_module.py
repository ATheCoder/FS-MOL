from modules.graph_encoders.GINEncoder import GINEncoder, GINEncoderConfig
from modules.similarity_modules import CNAPSProtoNetSimilarityModule
from mrc_src.encoders.fingerprint_encoder import FingerprintSimpleFeedForward
from fs_mol.utils.metrics import compute_binary_task_metrics
import torch
from typing import Any
import lightning as L

# from MXMNet.model import Config, MXMNet
import torchmetrics as tm
from torchmetrics.utilities import dim_zero_cat
import wandb


# from MXMNet.model import Config, MXMNet
import wandb


metrics_table_cols = ['task_name', 'size', 'acc', 'balanced_acc', 'f1', 'prec', 'recall', 'roc_auc', 'avg_precision', 'kappa', 'delta_auc_pr', 'optimistic_auc_pr', 'optimistic_delta_auc_pr']

class StandardDeviationMetric(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: torch.Tensor):
        self.values.append(value)

    def compute(self):
        vals = dim_zero_cat(self.values)
        return torch.std(vals, correction=0) 

class MRCLightningModule(L.LightningModule):
    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        self.config = kwargs.pop("config", config)
        self.REPR_TO_ENCODER_MAP = {
            "2d": GINEncoder,
            "fingerprint": FingerprintSimpleFeedForward
        }

        self.REPR_ENCODER_CONFIG_MAP = {
            "2d": GINEncoderConfig(self.config['dim'], self.config['layer'], self.config['cutoff'], dropout=self.config['dropout']),
            "fingerprint": config
        }
        super().__init__(*args, **kwargs)
        self.encoder = self.REPR_TO_ENCODER_MAP[self.config['representation']](self.REPR_ENCODER_CONFIG_MAP[self.config['representation']])
        
        self.std_dev_metric = StandardDeviationMetric() 
        
        self.similarity_module = CNAPSProtoNetSimilarityModule(self.config['beta'], True)

        self.wandb_table = wandb.Table(columns=metrics_table_cols)

        
        
    def calculate_feats(self, batch):
        encoded_graphs = self.encoder(batch)
        # feats = torch.cat([encoded_graphs, batch.fingerprint.reshape(-1, 2048)], dim=1)

        return encoded_graphs

    def calc_loss(self, input):
        batch, labels, is_query, batch_index, task_names = input
        feats = self.encoder(batch)
        # feats = F.normalize(feats, dim=-1)

        logits, batch_labels = self.similarity_module(feats, labels, is_query, batch_index)
        
        loss = self.similarity_module.calc_loss_from_logits(logits, batch_labels)
        

        return loss, logits, labels[is_query == 1], task_names
            

    def training_step(self, batches):
        loss, _, _, _ = self.calc_loss(batches)
        
        self.log('train_step', loss, on_step=True, on_epoch=False, batch_size=self.config['batch_size'])
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.config['batch_size'])
        
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.wandb_table = wandb.Table(columns=metrics_table_cols)

    def on_validation_epoch_end(self) -> None:
        epoch_std_dev = self.std_dev_metric.compute()
        self.log('delta_auc_pr_std', epoch_std_dev, on_epoch=True)
        if wandb.run is not None:
            wandb.log({'metrics_table': self.wandb_table})
    
    
    def validation_step(self, batches):
            valid_loss, logits, query_labels, task_names = self.calc_loss(batches)
            
            self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)

            batch_preds = self.similarity_module.get_probabilities_from_logits(logits)

            metrics = compute_binary_task_metrics(
                predictions=batch_preds.cpu(), labels=query_labels.detach().cpu().numpy() # type: ignore
            )

            self.wandb_table.add_data(task_names[0].split('_')[0], *metrics.__dict__.values())

            self.std_dev_metric(torch.tensor(metrics.delta_auc_pr.item())) # type: ignore

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
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            fused=True,
        )
        
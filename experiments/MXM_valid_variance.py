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
from fs_mol.data.torch_dl import FSMOLHTorchDataset, FSMOLTorchDataloader
from typing import Any
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import csv
import statistics

from MXMNet.model import Config, MXMNet




@dataclass(frozen=True)
class TrainConfig:
    # Training Settings:
    batch_size: int = 64
    train_support_count: int = 32
    train_query_count: int = 256
    train_shuffle: bool = True

    temprature: float = 0.07

    # Validation Settings:
    valid_support_count: int = 64
    valid_batch_size: int = 256

    # Model Settings:
    envelope_exponent: int = 6
    num_spherical: int = 7
    num_radial: int = 5
    dim: int = 256
    cutoff: int = 5.0
    layer: int = 7

    accumulate_grad_batches: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    dropout: float = 0.2

    encoder_dims = [128, 128, 256, 256, 512, 512]


config = TrainConfig()

model = MXMNet(
            Config(config.dim, config.layer, config.cutoff, config.encoder_dims, 512),
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            envelope_exponent=config.envelope_exponent,
            dropout=config.dropout,
        )


valid_repeat = 40

valid_dataset = FSMOLHTorchDataset("valid", "pyg", valid_repeat)

valid_dls = FSMOLTorchDataloader(
    valid_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=4,
    support_count=16,
    query_count=16,
)


class ClipLike(L.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.config = config
        super().__init__(*args, **kwargs)
        self.graph_encoder = MXMNet(
            Config(config.dim, config.layer, config.cutoff, config.encoder_dims, 512),
            num_spherical=config.num_spherical,
            num_radial=config.num_radial,
            envelope_exponent=config.envelope_exponent,
            dropout=config.dropout,
        )
        
        
    def calculate_feats(self, batch):
        encoded_graphs = self.graph_encoder(batch)
        # feats = torch.cat([encoded_graphs, batch.fingerprint.reshape(-1, 2048)], dim=1)

        return encoded_graphs

    def calc_loss(self, input):
        batch, labels, index_map = input
        feats = self.graph_encoder(batch)
        feats = F.normalize(feats, dim=-1)

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
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=1)
        
        return loss
    
    
    def validation_step(self, batches):
            valid_loss, logits, query_labels = self.calc_loss(batches)
            
            self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, batch_size=1)

            batch_preds = F.softmax(logits, dim=1).detach().cpu().numpy()

            metrics = compute_binary_task_metrics(
                predictions=batch_preds[:, 1], labels=query_labels.detach().cpu().numpy()
            )

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
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
cliplike = ClipLike()
# trainer = L.Trainer(limit_train_batches=10,  check_val_every_n_epoch=1, max_epochs=5, logger=WandbLogger())
console_logger = CSVLogger(save_dir=".", name=f"console_logs/{valid_repeat}")
trainer = L.Trainer(check_val_every_n_epoch=1, max_epochs=5, logger=console_logger)
# trainer.fit(model=cliplike, train_dataloaders=train_dl, val_dataloaders=valid_dls)
# Create a ConsoleLogger to log metrics in the console
for i in range(10):
    trainer.validate(model=cliplike, dataloaders=valid_dls)

# Open the CSV file
with open(f'/FS-MOL/console_logs/{valid_repeat}/version_0/metrics.csv', 'r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    
    # Read the first row (column names)
    column_names = next(csv_reader)
    
    # Initialize a dictionary to store the values for each column
    column_values = {name: [] for name in column_names}
    
    # Iterate over each remaining row in the CSV file
    for row in csv_reader:
        # Iterate over each value in the row
        for column_name, value in zip(column_names, row):
            # Append the value to the corresponding column
            column_values[column_name].append(float(value))
    
    # Calculate and print the standard deviation for each column
    for column_name, values in column_values.items():
        std_dev = statistics.stdev(values)
        print(f"Standard Deviation of '{column_name}': {std_dev}")
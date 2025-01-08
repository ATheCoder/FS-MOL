import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from fs_mol.data.torch_dl import MRCDataset, MRCDataLoader
from modules.top_level_modules.lightning_module import MRCLightningModule, TrainConfig

config = TrainConfig()  # Or pass in a constructor with desired overrides
model = MRCLightningModule(config=config)

train_dataset = MRCDataset("train","pyg",config.train_n_repeats,config.preload_dataset,not config.isProd)
valid_dataset = MRCDataset("valid","pyg",config.val_n_repeats,config.preload_dataset,not config.isProd)

train_dl = MRCDataLoader(
    config.representation,
    train_dataset,
    batch_size=config.batch_size,
    datatype="pyg",
    num_workers=config.dataloader_workers,
    shuffle=config.train_shuffle,
    support_count=config.train_support_count,
    query_count=config.train_query_count,
)

valid_dl = MRCDataLoader(
    config.representation,
    valid_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=config.dataloader_workers,
    support_count=16,
    query_count=16
)

checkpoint_callback = ModelCheckpoint(
    dirpath='MRC_Runner',
    filename='best-checkpoint',
    save_top_k=2,
    verbose=True,
    monitor='valid_optimistic_delta_auc_pr',
    mode='max',
    save_last=True
)

if config.isProd:
    wandb.init(project=f"{config.representation}_molecular_representation_comparison_final",
               config=config.__dict__)
    wandb.watch(model, log='all')

trainer = L.Trainer(
    callbacks=[checkpoint_callback],
    check_val_every_n_epoch=1,
    logger=WandbLogger() if config.isProd else None,
    max_epochs=10_000,
    accumulate_grad_batches=config.accumulate_grad_batches,
)

trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

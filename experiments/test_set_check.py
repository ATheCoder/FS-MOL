import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from fs_mol.data.torch_dl import MRCDataset, MRCDataLoader
from modules.top_level_modules.lightning_module import MRCLightningModule, TrainConfig

config = TrainConfig(batch_size=16,learning_rate=1e-4)  # Example override
model_ckpt_path = "/FS-MOL/MRC_Runner/best-checkpoint-v150.ckpt"

test_dataset = MRCDataset(
    "test","pyg",
    n_repeats=config.val_n_repeats,
    should_preload=config.preload_dataset,
    debug=not config.isProd
)

test_dl = MRCDataLoader(
    config.representation,
    test_dataset,
    batch_size=config.valid_batch_size,
    datatype="pyg",
    num_workers=config.dataloader_workers,
    support_count=16,
    query_count=16
)

model = MRCLightningModule.load_from_checkpoint(model_ckpt_path, config=config)

trainer = L.Trainer(logger=WandbLogger() if config.isProd else None)
trainer.test(model, dataloaders=test_dl)

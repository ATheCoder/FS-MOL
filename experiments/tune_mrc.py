import sys
import os
import inspect



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

print(sys.path)

from fs_mol.modules.mrc_lightning_module_reuseable import MRCLightningModule
from lightning import Callback, LightningModule, Trainer
from ray import tune, air
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from fs_mol.data_modules.mrc_datamodule import MRCDataModule
from ray.air.config import RunConfig, CheckpointConfig

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from lightning.pytorch.loggers.wandb import WandbLogger
from ray.air.integrations.wandb import setup_wandb

num_epochs = 50
accelerator = "gpu"

class TuneReportCallback_2(Callback):
    def __init__(self, val_to_monitor) -> None:
        super().__init__()
        self.val_to_monitor = val_to_monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        metric_value = metrics[self.val_to_monitor]

        session.report(
            metrics={self.val_to_monitor: metric_value.item()},
            # checkpoint=Checkpoint.from_dict(self.state_dict()),
        )


run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2, checkpoint_score_attribute="valid_delta_auc_pr", checkpoint_score_order="max"
    )
)


project_name = "MXMNet_Hyperparameter_Search_20"


def start_training(config):
    setup_wandb(config, project=project_name, rank_zero_only=False)
    model = MRCLightningModule(
        config=config
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            "/FS-MOL/checkpoints/tarjan-m/", monitor="valid_delta_auc_pr", save_top_k=2, mode="max"
        ),
        TuneReportCallback_2("valid_delta_auc_pr"),
    ]

    trainer = Trainer(callbacks=callbacks, check_val_every_n_epoch=1, logger=WandbLogger() if config['isProd'] else None, accumulate_grad_batches=config['accumulate_grad_batches'], gradient_clip_val=1.0, gradient_clip_algorithm='value')

    data_module = MRCDataModule(config)

    trainer.fit(model, data_module)


trainable_with_gpu = tune.with_resources(
    start_training, {"cpu": 6, "gpu": 1.0, "memory": 11_200_000_000}
)

tune_storage_path = "/FS-MOL/tune_storage"

if tune.Tuner.can_restore(f"{tune_storage_path}/{project_name}"):
    tuner = tune.Tuner.restore(
        f"{tune_storage_path}/{project_name}", trainable_with_gpu, resume_errored=True
    )
else:
    tuner = tune.Tuner(
        trainable_with_gpu,
        tune_config=tune.TuneConfig(
            metric="valid_delta_auc_pr",
            reuse_actors=False,
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=50,
            scheduler=ASHAScheduler(
                max_t=num_epochs, grace_period=num_epochs // 5, reduction_factor=2
            ),
        ),
        run_config=air.RunConfig(
            storage_path=tune_storage_path,
            name=project_name,
            checkpoint_config=CheckpointConfig(
                2,
                checkpoint_score_order="max",
                checkpoint_score_attribute="valid_delta_auc_pr",
            ),
        ),
        param_space={
            # Training Settings
            "representation": "fingerprint+descriptors",
            "batch_size": 32,
            "train_support_count": tune.choice([2, 4, 8, 16]),
            "train_query_count": tune.choice([2, 4, 8, 16]),
            "train_shuffle": True,
            
            # Model Architecture
            "dim": 128,
            "layer": 5,
            "envelope_exponent": 6,
            "num_spherical": 7,
            "num_radial": 5,
            "cutoff": 5.0,
            
            # Training Hyperparameters
            "beta": tune.loguniform(1e-2, 1e2),  # Inverse of temperature
            "accumulate_grad_batches": 1,
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "weight_decay": 0.0,
            "dropout": 0.0,
            
            # Validation Settings
            "valid_support_count": 64,
            "valid_batch_size": 1,
            
            # Encoder Architecture
            "encoder_dims": [128, 128, 256, 256, 512, 512],
            
            # Repetition Settings
            "train_n_repeats": 5,
            "val_n_repeats": 5,
            
            # Environment Settings
            "dataloader_workers": 18,
            "preload_dataset": False,
            "isProd": True
        },
    )

    results = tuner.fit()

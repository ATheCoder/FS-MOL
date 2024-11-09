from lightning import Callback, LightningModule, Trainer
from fs_mol.data_modules.MXM_datamodule import MXMDataModule
from ray import tune, air
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ray.air.config import RunConfig, CheckpointConfig

from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from lightning.pytorch.loggers.wandb import WandbLogger
from ray.air.integrations.wandb import setup_wandb

from modules.top_level_modules.mxm_3dgraph import (
    GenericMXM3DGraphConfig,
    GenericMXM3DGraphLightningModule,
)

num_epochs = 4
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
        num_to_keep=2, checkpoint_score_attribute="loss", checkpoint_score_order="min"
    )
)


project_name = "3DGraph_HyperParam_Search_1"


def start_training(config):
    setup_wandb(config, project=project_name, rank_zero_only=False)
    model = GenericMXM3DGraphLightningModule(
        GenericMXM3DGraphConfig(learning_rate=config["learning_rate"])
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint("/FS-MOL/checkpoints/3dgraph-m/", monitor="loss", save_top_k=2, mode="min"),
        TuneReportCallback_2("loss"),
    ]

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        val_check_interval=10000,
        log_every_n_steps=1,
        logger=WandbLogger(),
        default_root_dir="/FS-MOL/MXM_Checkpoint/",
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    data_module = MXMDataModule(
        "/FS-MOL/data/mxm/",
        batch_size=2,
        support_size=16,
        query_size=8,
        train_num_workers=6,
    )

    trainer.fit(model, data_module)


trainable_with_gpu = tune.with_resources(
    start_training, {"cpu": 6, "gpu": 1.0, "memory": 11_200_000_000}
)

tune_storage_path = "/FS-MOL/tune_storage"

# if tune.Tuner.can_restore(f"{tune_storage_path}/{project_name}"):
#     tuner = tune.Tuner.restore(
#         f"{tune_storage_path}/{project_name}", trainable_with_gpu, resume_errored=True
#     )
# else:
tuner = tune.Tuner(
    trainable_with_gpu,
    tune_config=tune.TuneConfig(
        metric="loss",
        reuse_actors=False,
        mode="min",
        search_alg=OptunaSearch(),
        num_samples=20,
        scheduler=ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2),
    ),
    run_config=air.RunConfig(
        storage_path=tune_storage_path,
        name=project_name,
        checkpoint_config=CheckpointConfig(
            2,
            checkpoint_score_order="min",
            checkpoint_score_attribute="loss",
        ),
    ),
    param_space={
        "learning_rate": tune.loguniform(1e-7, 1e-2),
    },
)

results = tuner.fit()

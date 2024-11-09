from lightning import Callback, LightningModule, Trainer
from fs_mol.data_modules.MXM_datamodule import TarjanDataModule
from ray import tune, air
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ray.air.config import RunConfig, CheckpointConfig

from modules.lightning_module import TarjanLightningModule
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from lightning.pytorch.loggers.wandb import WandbLogger
from ray.air.integrations.wandb import setup_wandb

num_epochs = 20
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
        num_to_keep=2, checkpoint_score_attribute="mean_delta_auc_pr", checkpoint_score_order="max"
    )
)


project_name = "MXMNet_Hyperparameter_Search_20"


def start_training(config):
    setup_wandb(config, project=project_name, rank_zero_only=False)
    model = TarjanLightningModule(
        config["dim"],
        config["layer"],
        config["cutoff"],
        None,
        None,
        config["dropout"],
        padding_size=config["padding_size"],
        prediction_scaling=config["prediction_scaling"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            "/FS-MOL/checkpoints/tarjan-m/", monitor="mean_delta_auc_pr", save_top_k=2, mode="max"
        ),
        TuneReportCallback_2("mean_delta_auc_pr"),
    ]

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        val_check_interval=1000,
        log_every_n_steps=1,
        logger=WandbLogger(),
        default_root_dir="/FS-MOL/MXM_Checkpoint/",
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    data_module = TarjanDataModule("/FS-MOL/data/tarjan", query_size=1, batch_size=64)

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
            metric="mean_delta_auc_pr",
            reuse_actors=False,
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=20,
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
                checkpoint_score_attribute="mean_delta_auc_pr",
            ),
        ),
        param_space={
            "dim": tune.choice([128, 256]),
            "layer": tune.choice([5, 7]),
            "cutoff": 4.0,
            "dropout": 0.0,
            "padding_size": 12,
            "prediction_scaling": tune.uniform(1e-2, 1.0),
            "learning_rate": 1e-5,
            "batch_size": 64,
        },
    )

    results = tuner.fit()

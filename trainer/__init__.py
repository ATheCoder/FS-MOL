from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import Trainer

import wandb
from fewshot_utils.is_debugger_attached import is_debugger_attached


def init_wandb(config, model, checkpoint_path):
    if is_debugger_attached():
        return

    run_id = checkpoint_path.split("/")[3].split("-")[1] if checkpoint_path is not None else None
    wandb.init(
        project=config.project_name,
        config=config,
        id=run_id if run_id is not None else None,
        resume=True if run_id is not None else False,
    )

    wandb.define_metric("mean_delta_auc_pr", summary="max")
    wandb.define_metric("valid_mean_loss", summary="min")

    wandb.watch(model, log="all")


def train_lightning_module(module, config, data_module, check_point_path=None, wandb_enabled=True):
    # model = (
    #     module.load_from_checkpoint(check_point_path, config=config, data_module=data_module)
    #     if check_point_path is not None
    #     else module(config, data_module)
    # )

    model = module(config, data_module)

    if wandb_enabled:
        init_wandb(config, model, check_point_path)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            f"/FS-MOL/checkpoints/tarjan-{wandb.run.id if wandb.run != None else 'debug'}/",
            monitor="mean_delta_auc_pr",
            save_top_k=2,
            mode="max",
            save_on_train_epoch_end=True,
            save_last=True,
        ),
    ]

    trainer = Trainer(
        # detect_anomaly=True,
        # overfit_batches=100,
        # profiler="simple",
        accelerator="gpu",
        devices=1,
        # max_epochs=20,
        # max_steps=1000,
        val_check_interval=1000,
        log_every_n_steps=1,
        logger=WandbLogger() if wandb_enabled else None,
        default_root_dir="/FS-MOL/MXM_Checkpoint/",
        callbacks=callbacks,
        precision=16,
        gradient_clip_val=1.0,
        # gradient_clip_algorithm="value",
        accumulate_grad_batches=config.accumulate_grad_batches,
    )
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=check_point_path,
    )

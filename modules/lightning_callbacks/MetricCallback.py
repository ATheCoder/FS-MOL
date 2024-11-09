from dataclasses import dataclass, fields
from typing import Any, List
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
from sklearn.metrics import roc_auc_score
from statistics import mean

import wandb

from modules.similarity_modules import SimilarityModule


@dataclass
class _Metrics:
    valid_mean_loss: float
    mean_delta_auc_pr: float
    auc_roc: float


def calculate_means(metrics_array: List[_Metrics]):
    return {f.name: mean(getattr(m, f.name) for m in metrics_array) for f in fields(_Metrics)}


def calculate_stds(metrics_array: List[_Metrics]):
    return {
        f.name: np.std(np.array([getattr(m, f.name) for m in metrics_array]), ddof=1)
        for f in fields(_Metrics)
    }


class MetricCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.to_log: List[_Metrics] = []
        self.task_to_auc_pr = wandb.Table(columns=["task_name", "delta_auc_pr"])

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits, query_labels = outputs["logits"], outputs["query_labels"]

        task_names = outputs["task_names"]

        similarity_module = getattr(pl_module, "similarity_module", None)

        if isinstance(similarity_module, SimilarityModule):
            auc_pr = similarity_module.calculate_delta_auc_pr(logits, query_labels)

            assert len(task_names) == 1
            self.task_to_auc_pr.add_data(task_names[0], auc_pr)

            loss = similarity_module.calc_loss_from_logits(logits, query_labels)

            probs = similarity_module.get_probabilities_from_logits(logits)

            try:
                auroc = roc_auc_score(
                    query_labels.reshape(-1).detach().cpu().numpy(), probs.detach().cpu().numpy()
                )
            except ValueError:
                return

            self.to_log.append(_Metrics(loss.item(), auc_pr, float(auroc)))

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_epoch_end(trainer, pl_module)

    def _clear_data(self):
        self.to_log.clear()
        self.task_to_auc_pr = wandb.Table(columns=["task_name", "delta_auc_pr"])

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mean_of_saved_metrics = calculate_means(self.to_log)
        stds_of_saved_metrics = calculate_stds(self.to_log)

        for f in fields(_Metrics):
            k = f.name
            pl_module.log(k, mean_of_saved_metrics[k])
            pl_module.log(f"std-{k}", stds_of_saved_metrics[k])

        confidence_interval = 1.96 * (
            stds_of_saved_metrics["mean_delta_auc_pr"] / np.sqrt(len(self.to_log))
        )

        pl_module.log(f"ci_mean_delta_auc_pr", confidence_interval)

        if wandb.run != None:
            wandb.run.log({"per_task_delta_auc_pr": self.task_to_auc_pr})

        self._clear_data()

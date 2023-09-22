from dataclasses import dataclass, fields
from typing import Any, List
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import roc_auc_score
from statistics import mean

from modules.similarity_modules import SimilarityModule


@dataclass
class _Metrics:
    valid_mean_loss: float
    mean_delta_auc_pr: float
    auc_roc: float


def calculate_means(metrics_array: List[_Metrics]):
    return {f.name: mean(getattr(m, f.name) for m in metrics_array) for f in fields(_Metrics)}


class MetricCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.to_log: List[_Metrics] = []

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

        similarity_module = getattr(pl_module, "similarity_module", None)

        if isinstance(similarity_module, SimilarityModule):
            auc_pr = similarity_module.calculate_delta_auc_pr(logits, query_labels)
            loss = similarity_module.calc_loss_from_logits(logits, query_labels)

            probs = similarity_module.get_probabilities_from_logits(logits)

            auroc = roc_auc_score(
                query_labels.reshape(-1).detach().cpu().numpy(), probs.detach().cpu().numpy()
            )

            self.to_log.append(_Metrics(loss.item(), auc_pr, float(auroc)))

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mean_of_saved_metrics = calculate_means(self.to_log)

        for f in fields(_Metrics):
            k = f.name
            pl_module.log(k, mean_of_saved_metrics[k])

        self.to_log.clear()

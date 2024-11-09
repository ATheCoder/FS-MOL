import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import wandb

import numpy as np
import torch
from torch import nn
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


from fs_mol.models.abstract_torch_fsmol_model import linear_warmup
from fs_mol.data import FSMolDataset, FSMolTaskSample, DataFold
from fs_mol.data.protonet import (
    ProtoNetBatch,
    get_protonet_task_sample_iterable,
    get_protonet_batcher,
    task_sample_to_pn_task_sample,
)
from fs_mol.models.protonet import PrototypicalNetwork, PrototypicalNetworkConfig, _compute_class_means_and_precisions, _estimate_cov, _extract_class_indices, calculate_mahalanobis_logits
from fs_mol.models.abstract_torch_fsmol_model import MetricType
from fs_mol.utils.metrics import (
    BinaryEvalMetrics,
    compute_binary_task_metrics,
    avg_metrics_over_tasks,
    avg_task_metrics_list,
)
from fs_mol.utils.metric_logger import MetricLogger
from fs_mol.utils.torch_utils import torchify
from fs_mol.utils.test_utils import eval_model, FSMolTaskSampleEvalResults


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PrototypicalNetworkTrainerConfig(PrototypicalNetworkConfig):
    batch_size: int = 256
    tasks_per_batch: int = 16
    support_set_size: int = 16
    query_set_size: int = 256

    num_train_steps: int = 10000
    validate_every_num_steps: int = 50
    validation_support_set_sizes: Tuple[int] = (16, 128)
    validation_query_set_size: int = 256
    validation_num_samples: int = 5

    learning_rate: float = 0.001
    clip_value: Optional[float] = None
    
    use_attention: bool = False


def run_on_batches(
    model: PrototypicalNetwork,
    batches: List[ProtoNetBatch],
    batch_labels: List[torch.Tensor],
    train: bool = False,
    tasks_per_batch: int = 1,
) -> Tuple[float, BinaryEvalMetrics]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_num_samples = 0.0, 0
    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []

    num_gradient_accumulation_steps = len(batches) * tasks_per_batch
    for batch_features, batch_labels in zip(batches, batch_labels):
        # Compute task loss
        batch_logits = model(batch_features)
        batch_loss = model.compute_loss(batch_logits, batch_labels)
        # divide this batch loss by the total number of accumulation steps
        batch_loss = batch_loss / num_gradient_accumulation_steps
        if train:
            batch_loss.backward()
        total_loss += (
            batch_loss.detach() * batch_features.num_query_samples * num_gradient_accumulation_steps
        )
        total_num_samples += batch_features.num_query_samples
        batch_preds = torch.nn.functional.softmax(batch_logits, dim=1).detach().cpu().numpy()
        task_preds.append(batch_preds[:, 1])
        task_labels.append(batch_labels.detach().cpu().numpy())

    metrics = compute_binary_task_metrics(
        predictions=np.concatenate(task_preds, axis=0), labels=np.concatenate(task_labels, axis=0)
    )

    # we will report loss per sample as before.
    return total_loss.cpu().item() / total_num_samples, metrics


def evaluate_protonet_model(
    model: PrototypicalNetwork,
    dataset: FSMolDataset,
    support_sizes: List[int] = [16, 128],
    num_samples: int = 5,
    seed: int = 0,
    batch_size: int = 320,
    query_size: Optional[int] = None,
    data_fold: DataFold = DataFold.TEST,
    save_dir: Optional[str] = None,
) -> Dict[str, List[FSMolTaskSampleEvalResults]]:

    batcher = get_protonet_batcher(max_num_graphs=batch_size)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        pn_task_sample = torchify(
            task_sample_to_pn_task_sample(task_sample, batcher), device=model.device
        )

        _, result_metrics = run_on_batches(
            model,
            batches=pn_task_sample.batches,
            batch_labels=pn_task_sample.batch_labels,
            train=False,
        )

        num_support_samples = pn_task_sample.__dict__['num_support_samples']

        wandb.log({f"{num_support_samples}-precision": result_metrics.prec, f"{num_support_samples}-recall": result_metrics.recall})
        wandb.log({**result_metrics.__dict__})

        logger.info(
            f"{pn_task_sample.task_name}:"
            f" {pn_task_sample.num_support_samples:3d} support samples,"
            f" {pn_task_sample.num_query_samples:3d} query samples."
            f" Avg. prec. {result_metrics.avg_precision:.5f}.",
        )

        return result_metrics

    return eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=support_sizes,
        out_dir=save_dir,
        num_samples=num_samples,
        test_size_or_ratio=query_size,
        fold=data_fold,
        seed=seed,
    )


def validate_by_finetuning_on_tasks(
    model: PrototypicalNetwork,
    dataset: FSMolDataset,
    seed: int = 0,
    aml_run=None,
    metric_to_use: MetricType = "avg_precision",
) -> float:
    model.train(False)
    """
    Validation function for prototypical networks. Similar to test function;
    each validation task is used to evaluate the model more than once, the
    final results are a mean value for all tasks over the requested metric.
    """

    task_results = evaluate_protonet_model(
        model,
        dataset,
        support_sizes=model.config.validation_support_set_sizes,
        num_samples=model.config.validation_num_samples,
        seed=seed,
        batch_size=model.config.batch_size,
        query_size=model.config.validation_query_set_size,
        data_fold=DataFold.VALIDATION,
    )

    # take the dictionary of task_results and return correct mean over all tasks
    mean_metrics = avg_metrics_over_tasks(task_results)
    if aml_run is not None:
        for metric_name, (metric_mean, _) in mean_metrics.items():
            aml_run.log(f"valid_task_test_{metric_name}", float(metric_mean))

    return mean_metrics[metric_to_use][0]


class PrototypicalNetworkTrainer(nn.Module):
    def __init__(self, config: PrototypicalNetworkTrainerConfig, encoder_model: nn.Module):
        super().__init__()
        self.config = config
        
        self.encoder = encoder_model
        
        base_lr = self.config.learning_rate
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=base_lr)
        
        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: ProtoNetBatch):
        encoded_support_features = self.encoder(input_batch.support_features)
        support_labels = input_batch.support_labels
        
        encoded_query_features = self.encoder(input_batch.query_features)

        if self.config.distance_metric == "mahalanobis":
            return calculate_mahalanobis_logits(encoded_support_features, support_labels, encoded_query_features, self.device)

        else:  # euclidean
            logits = self._protonets_euclidean_classifier(
                encoded_support_features,
                encoded_query_features,
                support_labels,
            )

        return logits
            

    def compute_class_means_and_precisions(self,
        features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _compute_class_means_and_precisions(features=features, labels=labels, device=self.device)

    @staticmethod
    def _estimate_cov(
        examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
    ) -> torch.Tensor:
        return _estimate_cov(examples=examples, rowvar=rowvar, inplace=inplace)

    @staticmethod
    def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
        return _extract_class_indices(labels=labels, which_class=which_class)

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def _protonets_euclidean_classifier(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        class_prototypes = self._compute_class_prototypes(support_features, support_labels)
        logits = self._euclidean_distances(query_features, class_prototypes)
        return logits

    def _compute_class_prototypes(
        self, support_features: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        means = []
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(
                support_features, 0, self._extract_class_indices(support_labels, c)
            )
            means.append(torch.mean(class_features, dim=0))
        return torch.stack(means)

    def _euclidean_distances(
        self, query_features: torch.Tensor, class_prototypes: torch.Tensor
    ) -> torch.Tensor:
        num_query_features = query_features.shape[0]
        num_prototypes = class_prototypes.shape[0]

        distances = (
            (
                query_features.unsqueeze(1).expand(num_query_features, num_prototypes, -1)
                - class_prototypes.unsqueeze(0).expand(num_query_features, num_prototypes, -1)
            )
            .pow(2)
            .sum(dim=2)
        )

        return -distances


    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ):
        data = self.get_model_state()

        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            data["epoch"] = epoch

        torch.save(data, path)

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        for name, param in pretrained_state_dict["model_state_dict"].items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

        optimizer_weights = pretrained_state_dict.get("optimizer_state_dict")
        if optimizer_weights is not None:
            for name, param in optimizer_weights.items():
                self.optimizer.state_dict()[name].copy_(param)

    def load_model_gnn_weights(
        self,
        path: str,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        gnn_model_state_dict = pretrained_state_dict["model_state_dict"]
        our_state_dict = self.state_dict()

        # Load parameters (names specialised to GNNMultitask model), but also collect
        # parameters for GNN parts / rest, so that we can create a LR warmup schedule:
        gnn_params, other_params = [], []
        gnn_feature_extractor_param_name = "graph_feature_extractor."
        for our_name, our_param in our_state_dict.items():
            if (
                our_name.startswith(gnn_feature_extractor_param_name)
                and "final_norm_layer" not in our_name
            ):
                generic_name = our_name[len(gnn_feature_extractor_param_name) :]
                if generic_name.startswith("readout_layer."):
                    generic_name = f"readout{generic_name[len('readout_layer'):]}"
                our_param.copy_(gnn_model_state_dict[generic_name])
                logger.debug(f"I: Loaded parameter {our_name} from {generic_name} in {path}.")
                gnn_params.append(our_param)
            else:
                logger.debug(f"I: Not loading parameter {our_name}.")
                other_params.append(our_param)

        self.optimizer = torch.optim.Adam(
            [
                {"params": other_params, "lr": self.config.learning_rate},
                {"params": gnn_params, "lr": self.config.learning_rate / 10},
            ],
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=[
                partial(linear_warmup, warmup_steps=0),  # for all params
                partial(linear_warmup, warmup_steps=100),  # for loaded GNN params
            ],
        )

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "PrototypicalNetworkTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        if not quiet:
            logger.info(f" Loading model configuration from {model_file}.")

        model = PrototypicalNetworkTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            load_task_specific_weights=True,
            device=device,
        )
        return model

    def train_loop(self, out_dir: str, dataset: FSMolDataset, device: torch.device, aml_run=None):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))
        counter = Counter()
        train_task_sample_iterator = iter(
            get_protonet_task_sample_iterable(
                dataset=dataset,
                data_fold=DataFold.TRAIN,
                num_samples=1,
                max_num_graphs=self.config.batch_size,
                support_size=self.config.support_set_size,
                query_size=self.config.query_set_size,
                repeat=True,
            )
        )

        best_validation_avg_prec = 0.0
        metric_logger = MetricLogger(
            log_fn=lambda msg: logger.info(msg),
            aml_run=aml_run,
            window_size=max(10, self.config.validate_every_num_steps / 5),
        )

        for step in range(1, self.config.num_train_steps + 1):
            self.train(True)
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            

            task_batch_losses: List[float] = []
            task_batch_metrics: List[BinaryEvalMetrics] = []
            for _ in range(self.config.tasks_per_batch):
                task_sample = next(train_task_sample_iterator)
                counter.update([task_sample.task_name])
                wandb.log(dict(counter))
                train_task_sample = torchify(task_sample, device=device)
                task_loss, task_metrics = run_on_batches(
                    self,
                    batches=train_task_sample.batches,
                    batch_labels=train_task_sample.batch_labels,
                    train=True,
                    tasks_per_batch=self.config.tasks_per_batch,
                )
                task_batch_losses.append(task_loss)
                task_batch_metrics.append(task_metrics)
                wandb.log({**task_metrics.__dict__, "task_loss": task_loss})

            # Now do a training step - run_on_batches will have accumulated gradients
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_value)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            task_batch_avg_metrics = avg_task_metrics_list(task_batch_metrics)
            wandb.log({"loss": task_batch_mean_loss, "avg_task_prec": task_batch_avg_metrics["avg_precision"][0], "avg_task_kappa": task_batch_avg_metrics["kappa"][0], "avg_task_acc": task_batch_avg_metrics["acc"][0]})
            metric_logger.log_metrics(
                loss=task_batch_mean_loss,
                avg_prec=task_batch_avg_metrics["avg_precision"][0],
                kappa=task_batch_avg_metrics["kappa"][0],
                acc=task_batch_avg_metrics["acc"][0],
            )

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = validate_by_finetuning_on_tasks(self, dataset, aml_run=aml_run)
                wandb.log({"valid_mean_avg_prec": valid_metric})

                if aml_run:
                    # printing some measure of loss on all validation tasks.
                    aml_run.log(f"valid_mean_avg_prec", valid_metric)

                logger.info(
                    f"Validated at train step [{step}/{self.config.num_train_steps}],"
                    f" Valid Avg. Prec.: {valid_metric:.3f}",
                )
                
                wandb.log({"validation_prec": valid_metric})
                
                torch.save(self.state_dict(), os.path.join(out_dir, "latest_model.pt"))
                torch.save(self.optimizer.state_dict(), os.path.join(out_dir, "optimizer_state.pt"))

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_avg_prec:
                    best_validation_avg_prec = valid_metric
                    model_path = os.path.join(out_dir, "best_validation.pt")
                    self.save_model(model_path)
                    logger.info(f"Updated {model_path} to new best model at train step {step}")

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained.pt"))

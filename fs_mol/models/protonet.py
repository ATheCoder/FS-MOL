from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn

from fs_mol.modules.graph_feature_extractor import (
    GraphFeatureExtractor,
    GraphFeatureExtractorConfig,
)
from fs_mol.data.protonet import MoleculeProtoNetFeatures, ProtoNetBatch, PyG_ProtonetBatch
from fs_mol.modules.pyg_gnn import PyG_GraphFeatureExtractor
from fs_mol.modules.bidirectional_attention import BidirectionalAttention


FINGERPRINT_DIM = 2048
PHYS_CHEM_DESCRIPTORS_DIM = 42


def _estimate_cov(
    examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
) -> torch.Tensor:
    """
    SCM: Function based on the suggested implementation of Modar Tensai
    and his answer as noted in:
    https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

    Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        examples: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if examples.dim() == 3:
        examples = examples.transpose(1, 2)  # 1*256*16
        factor = 1.0 / (examples.size(2) - 1)
        examples = examples - torch.mean(examples, dim=2, keepdim=True)
        examples_t = examples.transpose(1, 2)  # 1 * 16 * 256
        return factor * examples.bmm(examples_t)
    if examples.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if examples.dim() < 2:
        examples = examples.view(1, -1)
    if not rowvar and examples.size(0) != 1:
        examples = examples.t()
    factor = 1.0 / (examples.size(1) - 1)
    if inplace:
        examples -= torch.mean(examples, dim=1, keepdim=True)
    else:
        examples = examples - torch.mean(examples, dim=1, keepdim=True)
    examples_t = examples.t()
    return factor * examples.matmul(examples_t).squeeze()


def _extract_class_indices_batched(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
    batch_size, num_samples = labels.shape
    class_mask = labels == which_class.view(-1, 1)
    class_mask_indices = torch.nonzero(class_mask)
    return class_mask_indices


def _estimate_cov_batched(
    examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
) -> torch.Tensor:
    batch_size, num_samples, num_features = examples.shape
    factor = 1.0 / (num_samples - 1)

    if not rowvar:
        examples = examples.permute(0, 2, 1)

    means = torch.mean(examples, dim=-1, keepdim=True)

    if inplace:
        examples -= means
    else:
        examples = examples - means

    examples_t = examples.permute(0, 2, 1)

    covariances = factor * torch.bmm(examples, examples_t)

    return covariances


def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def batch_covariance_matrix(tensor):
    # Shape: (batch_size, num_observations, num_features)
    batch_size, num_observations, num_features = tensor.shape

    # Center the data by subtracting the mean
    tensor_mean = torch.mean(tensor, dim=1, keepdim=True)  # Shape: (batch_size, 1, num_features)
    tensor_centered = tensor - tensor_mean  # Shape: (batch_size, num_observations, num_features)

    # Calculate covariance matrix
    cov_matrix = torch.matmul(tensor_centered.transpose(1, 2), tensor_centered) / (
        num_observations - 1
    )
    # Shape: (batch_size, num_features, num_features)

    return cov_matrix


def batch_compute_class_means_and_precisions(
    features: torch.Tensor, labels: torch.Tensor, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = features.shape[0]
    feature_size = features.shape[-1]
    means = []
    precisions = []
    task_covariance_estimate = batch_covariance_matrix(features)
    for c in torch.unique(labels):
        # filter out feature vectors which have class c
        mask = labels == c
        class_features = features[mask.unsqueeze(-1).expand_as(features)].reshape(
            batch_size, -1, feature_size
        )
        # mean pooling examples to form class means
        means.append(torch.mean(class_features, dim=1, keepdim=True).squeeze())
        lambda_k_tau = class_features.size(1) / (class_features.size(1) + 1)
        lambda_k_tau = min(lambda_k_tau, 0.1)
        precisions.append(
            torch.inverse(
                (lambda_k_tau * batch_covariance_matrix(class_features))
                + ((1 - lambda_k_tau) * task_covariance_estimate)
                + 0.1
                * torch.eye(feature_size, feature_size)
                .to(device)
                .expand(batch_size, feature_size, feature_size)
            )
        )

    means = torch.stack(means)
    precisions = torch.stack(precisions)

    return means, precisions


def _compute_class_means_and_precisions(
    features: torch.Tensor, labels: torch.Tensor, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = []
    precisions = []
    task_covariance_estimate = _estimate_cov(features)
    for c in torch.unique(labels):
        # filter out feature vectors which have class c
        class_features = torch.index_select(features, 0, _extract_class_indices(labels, c))
        # mean pooling examples to form class means
        means.append(torch.mean(class_features, dim=0, keepdim=True).squeeze())
        lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
        lambda_k_tau = min(lambda_k_tau, 0.1)
        precision = torch.inverse(
            (lambda_k_tau * _estimate_cov(class_features))
            + ((1 - lambda_k_tau) * task_covariance_estimate)
            + 0.1 * torch.eye(class_features.size(1), class_features.size(1)).to(device)
        )
        precisions.append(precision)

    means = torch.stack(means)
    precisions = torch.stack(precisions)

    return means, precisions


def calculate_mahalanobis_logits(
    support_features_flat, support_labels, query_features_flat, device=None
):
    class_means, class_precision_matrices = _compute_class_means_and_precisions(
        support_features_flat, support_labels, device
    )

    # grabbing the number of classes and query examples for easier use later
    number_of_classes = class_means.size(0)
    number_of_targets = query_features_flat.size(0)

    """
    Calculating the Mahalanobis distance between query examples and the class means
    including the class precision estimates in the calculations, reshaping the distances
    and multiplying by -1 to produce the sample logits
    """
    # query_features_flat.repeat(1, number_of_classes) -> 16 x 512 -> 32 * 256
    repeated_target = query_features_flat.repeat(1, number_of_classes).view(-1, class_means.size(1))
    repeated_class_means = class_means.repeat(number_of_targets, 1)
    repeated_difference = repeated_class_means - repeated_target
    repeated_difference = repeated_difference.view(
        number_of_targets, number_of_classes, repeated_difference.size(1)
    ).permute(1, 0, 2)

    first_half = torch.matmul(repeated_difference, class_precision_matrices)
    logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1

    return logits


@dataclass(frozen=True)
class PrototypicalNetworkConfig:
    # Model configuration:
    graph_feature_extractor_config: GraphFeatureExtractorConfig = GraphFeatureExtractorConfig()
    used_features: Literal[
        "gnn", "ecfp", "pc-descs", "gnn+ecfp", "ecfp+fc", "pc-descs+fc", "gnn+ecfp+pc-descs+fc"
    ] = "gnn+ecfp+fc"
    distance_metric: Literal["mahalanobis", "euclidean"] = "mahalanobis"
    use_attention: bool = False


class PyG_PrototypicalNetwork(nn.Module):
    def __init__(self, graphFeatureExtractor: PyG_GraphFeatureExtractor):
        super().__init__()

        self.graph_feature_extractor = graphFeatureExtractor

        self.device = graphFeatureExtractor.device

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def forward(self, input_batch: PyG_ProtonetBatch):
        support_features = self.graph_feature_extractor(input_batch.support_graphs)
        query_features = self.graph_feature_extractor(input_batch.query_graphs)

        support_labels = input_batch.support_graphs.y

        return calculate_mahalanobis_logits(
            support_features, support_labels, query_features, device=self.device
        )


@dataclass(frozen=True)
class AttentionBasedEncoderConfig(PrototypicalNetworkConfig):
    n_heads: int = 8
    d_ff: int = 3072
    attn_output_dim: int = 512


class AttentionBasedEncoder(nn.Module):
    def __init__(self, config: AttentionBasedEncoderConfig) -> None:
        super().__init__()
        self.config = config
        graph_repr_dim = config.graph_feature_extractor_config.readout_config.output_dim

        self.graph_feature_extractor = GraphFeatureExtractor(config.graph_feature_extractor_config)

        self.use_fc = self.config.used_features.endswith("+fc")

        if self.use_fc:
            # Determine dimension:
            fc_in_dim = 0
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

        desc_length = 512 if self.use_fc else fc_in_dim
        self.attention = BidirectionalAttention(
            graph_repr_dim, desc_length, config.n_heads, config.d_ff, config.attn_output_dim
        )

    def forward(self, raw_features: MoleculeProtoNetFeatures):
        graph_features = self.graph_feature_extractor(raw_features)

        secondary_features = []

        if "ecfp" in self.config.used_features:
            secondary_features.append(raw_features.fingerprints.to(torch.float32))
        if "pc-descs" in self.config.used_features:
            secondary_features.append(raw_features.descriptors.to(torch.float32))

        secondary_features = torch.cat(secondary_features, dim=-1)

        if self.use_fc:
            secondary_features = self.fc(secondary_features)

        return self.attention(graph_features, secondary_features)



class VanillaFSMolEncoder(nn.Module):
    def __init__(self, config: PrototypicalNetworkConfig):
        super().__init__()
        self.config = config
        graph_repr_dim = config.graph_feature_extractor_config.readout_config.output_dim

        # Create GNN if needed:
        if self.config.used_features.startswith("gnn"):
            self.graph_feature_extractor = GraphFeatureExtractor(
                config.graph_feature_extractor_config
            )

        self.use_fc = self.config.used_features.endswith("+fc")

        # Create MLP if needed:
        if self.use_fc:
            # Determine dimension:
            fc_in_dim = 0
            if "gnn" in self.config.used_features:
                fc_in_dim += graph_repr_dim
            if "ecfp" in self.config.used_features:
                fc_in_dim += FINGERPRINT_DIM
            if "pc-descs" in self.config.used_features:
                fc_in_dim += PHYS_CHEM_DESCRIPTORS_DIM

            self.fc = nn.Sequential(
                nn.Linear(fc_in_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, raw_features: MoleculeProtoNetFeatures):
        final_features: List[torch.Tensor] = []

        if "gnn" in self.config.used_features:
            final_features.append(self.graph_feature_extractor(raw_features))
        if "ecfp" in self.config.used_features:
            final_features.append(raw_features.fingerprints.to(torch.float32))
        if "pc-descs" in self.config.used_features:
            final_features.append(raw_features.descriptors.to(torch.float32))

        final_features = torch.cat(final_features, dim=1)

        if self.use_fc:
            final_features = self.fc(final_features)

        return final_features


class PrototypicalNetwork(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        self.encoder = model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch: ProtoNetBatch):
        encoded_support_features = self.encoder(input_batch.support_features)
        support_labels = input_batch.support_labels

        encoded_query_features = self.encoder(input_batch.query_features)

        if self.config.distance_metric == "mahalanobis":
            return calculate_mahalanobis_logits(
                encoded_support_features, support_labels, encoded_query_features, self.device
            )

        else:  # euclidean
            logits = self._protonets_euclidean_classifier(
                encoded_support_features,
                encoded_query_features,
                support_labels,
            )

        return logits

    def compute_class_means_and_precisions(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _compute_class_means_and_precisions(
            features=features, labels=labels, device=self.device
        )

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

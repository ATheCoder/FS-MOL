import numpy as np
from torch import nn
from torch_geometric.nn.aggr.utils import (
    PoolingByMultiheadAttention,
    SetAttentionBlock,
)
import torch

from fs_mol.models.protonet import calculate_mahalanobis_logits
from utils.batch import batch_2d_tensor, separate_qsl


class SetTransformerSimilarityModule(nn.Module):
    def __init__(
        self, dim, heads, layer_norm, dropout, num_encoder_blocks=1, num_decoder_blocks=1
    ) -> None:
        super().__init__()

        self.encoders = torch.nn.ModuleList(
            [SetAttentionBlock(dim, heads, layer_norm, dropout) for _ in range(num_encoder_blocks)]
        )

        self.pma = PoolingByMultiheadAttention(dim, 1, heads, layer_norm, dropout)

        self.decoders = torch.nn.ModuleList(
            [SetAttentionBlock(dim, heads, layer_norm, dropout) for _ in range(num_decoder_blocks)]
        )

    def aggregate(self, graph_reprs):
        res = graph_reprs

        for encoder in self.encoders:
            res = encoder(graph_reprs)

        res = self.pma(res)

        return res

    def forward(self, support_pos_vectors, support_neg_vectors, query_vectors):
        support_pos_prototypes = self.aggregate(support_pos_vectors)
        support_neg_prototypes = self.aggregate(support_neg_vectors)

        support_prototypes = torch.cat([support_neg_prototypes, support_pos_prototypes], dim=1)

        logits = torch.bmm(query_vectors, support_prototypes.transpose(1, 2))

        return logits


class CNAPSProtoNetSimilarityModule(nn.Module):
    def __init__(self, init_prediction_scaling) -> None:
        super().__init__()

        self.prediction_scaling = nn.Parameter(torch.ones([]) * np.log(1 / init_prediction_scaling))

    def calc_label_cov(self, task_cov, label_cov):
        """
        Calculates the inverse of the covariance matrix for a particular label

        Parameters:
        task_cov -- task covariance matrix
        label_cov -- covariance matrix for the label
        """
        lambda_k_tau = 0.1
        return torch.linalg.inv(
            (lambda_k_tau * label_cov)
            + ((1 - lambda_k_tau) * task_cov)
            + 0.1 * torch.eye(task_cov.shape[-1], task_cov.shape[-1], device=task_cov.device)
        )

    def batch_calculate_mahalanobis_logits(
        self, support_set, support_labels, query_set, support_set_lengths, query_set_lengths
    ):
        """
        Calculates mahalanobis logits for a given query_set using the support_set and its labels.
        Note that this function takes a batched and padded support_set, support_labels and query_set.

        Parameters:
        support_set -- The support set for the episode, Has a shape of [Batch, Support_Count, Embedding_dim]
        support_labels --  The labels for each vector of the support_set.  Has a shape of [Batch, Support_Count]
        query_set -- The query set for which the logits will be generated. Has a shape of [Batch, Query_count, Embedding_dim]
        support_set_lengths -- The unpadded lengths of the support_set, this is used to distinguish padding indices. of shape [Batch,]
        query_set_lengths -- The unpadded lengths of query_set, this is used to distinguish padding indices. of shape [Batch,]
        """

        # Step 1: Calculate mean vectors for positive and negative support sets
        mask_positive = support_labels.unsqueeze(-1).eq(1).float()
        mask_negative = support_labels.unsqueeze(-1).eq(0).float()

        support_mask = (
            torch.arange(support_set.shape[1], device=support_set.device)[None, :]
            < support_set_lengths[:, None]
        )
        support_mask = support_mask.unsqueeze(-1).float()

        mask_positive *= support_mask
        mask_negative *= support_mask

        positive_mean = (support_set * mask_positive).sum(dim=1) / mask_positive.sum(dim=1)
        negative_mean = (support_set * mask_negative).sum(dim=1) / mask_negative.sum(dim=1)

        # Step 2: Compute covariance matrices for each batch
        cov_mats = self._torch_cov(support_set, support_set_lengths)
        # cov_mats_inv = torch.linalg.inv(cov_mats)
        pos_counts = mask_positive.sum(dim=[1, 2])
        neg_counts = mask_negative.sum(dim=[1, 2])

        pos_mats = self._torch_cov(
            batch_2d_tensor(support_set[mask_positive.squeeze(-1).bool()], pos_counts.long()),
            pos_counts,
        )

        neg_mats = self._torch_cov(
            batch_2d_tensor(support_set[mask_negative.squeeze(-1).bool()], neg_counts.long()),
            mask_negative.sum(dim=[1, 2]),
        )

        pos_mats_inv = self.calc_label_cov(cov_mats, pos_mats)
        neg_mats_inv = self.calc_label_cov(cov_mats, neg_mats)

        # Step 3 & 4: Mahalanobis distance & label assignment
        query_mask = (
            torch.arange(query_set.shape[1], device=support_set.device)[None, :]
            < query_set_lengths[:, None]
        )
        query_mask = query_mask.unsqueeze(-1).float()

        pos_diff = query_set - positive_mean[:, None, :]
        neg_diff = query_set - negative_mean[:, None, :]

        pos_maha = torch.einsum("bik,bkx,bij->bi", pos_diff, pos_mats_inv, pos_diff)
        neg_maha = torch.einsum("bik,bkx,bij->bi", neg_diff, neg_mats_inv, neg_diff)

        pos_maha *= query_mask.squeeze(-1)
        neg_maha *= query_mask.squeeze(-1)

        # pos_maha = torch.sqrt(pos_maha)
        # neg_maha = torch.sqrt(neg_maha)

        return torch.stack([neg_maha, pos_maha], dim=-1)

    def _torch_cov(self, tensor, lengths):
        """
        Generates the covariance matrix for a padded 3D tensor of shape [Batch, Len, Dim]

        Parameters:
        tensor -- The tensor of shape [Batch, Len, Dim]
        lengths -- The unpadded lengths of each batch [Batch, ]
        """
        batch_size, _, num_features = tensor.shape

        tensor_mask = (
            torch.arange(tensor.shape[1], device=tensor.device)[None, :] < lengths[:, None]
        )
        tensor_mask = tensor_mask.unsqueeze(-1).float()

        tensor *= tensor_mask

        tensor_mean = tensor.sum(dim=1) / lengths[:, None]
        tensor_mean = tensor_mean.unsqueeze(1)
        tensor_centered = tensor - tensor_mean

        tensor_centered *= tensor_mask

        cov_matrix = torch.matmul(tensor_centered.transpose(1, 2), tensor_centered)
        cov_matrix /= lengths[:, None, None] - 1

        return cov_matrix

    def forward(self, graph_reprs, labels, is_query, batch_index):
        """
        Parameters:

        graph_reprs -- The representation of all graphs. Of (2D) shape [Batch * (Support_Count + Query_Count), Embedding_Dim]
        labels -- The labels of all graphs. Of (1D) shape [Batch * (Support_Count + Query_Count)]
        is_query -- Whether a given graph representation vector in `graph_reprs` is a query vector (1) or a support vector (0).
        batch_index -- Shows which batch each graph in `graph_reprs` belongs to. Of (1D) Shape [Batch * (Support_Count + Query_Count)]
        """
        (
            batch_support_graphs,
            batch_support_labels,
            support_graph_lengths,
            batch_query_graphs,
            batch_query_labels,
            query_graph_lengths,
        ) = separate_qsl(graph_reprs, labels, is_query, batch_index)
        # Get divide support and query sets.
        logits = self.batch_calculate_mahalanobis_logits(
            batch_support_graphs,
            batch_support_labels,
            batch_query_graphs,
            support_graph_lengths,
            query_graph_lengths,
        )

        logit_scale = self.prediction_scaling.exp()

        return logits * logit_scale


class SingleBatch_CNAPSProtoNetSimilarityModule(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, graph_reprs, labels, is_query, batch_index):
        (
            batch_support_graphs,
            batch_support_labels,
            support_graph_lengths,
            batch_query_graphs,
            batch_query_labels,
            query_graph_lengths,
        ) = separate_qsl(graph_reprs, labels, is_query, batch_index)

        assert len(batch_support_graphs) == 1
        assert len(batch_support_labels) == 1
        assert len(batch_query_graphs) == 1

        logits = calculate_mahalanobis_logits(
            batch_support_graphs[0],
            batch_support_labels[0],
            batch_query_graphs[0],
            batch_support_graphs.device,
        )

        return logits

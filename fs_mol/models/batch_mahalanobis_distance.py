import torch


def batch_calculate_mahalanobis_logits(support_set, support_labels, query_set):
    batch_size, _, features = support_set.shape

    # Step 1: Calculate mean vectors for positive and negative support sets
    mask_positive = support_labels.unsqueeze(-1).eq(1).float()
    mask_negative = support_labels.unsqueeze(-1).eq(0).float()

    positive_mean = (support_set * mask_positive).sum(dim=1) / mask_positive.sum(dim=1)
    negative_mean = (support_set * mask_negative).sum(dim=1) / mask_negative.sum(dim=1)

    # Step 2: Compute covariance matrices for each batch
    cov_mats = _torch_cov(support_set)
    cov_mats_inv = torch.linalg.inv(cov_mats)

    # Step 3 & 4: Mahalanobis distance & label assignment
    pos_diff = query_set - positive_mean[:, None, :]
    neg_diff = query_set - negative_mean[:, None, :]

    pos_maha = torch.einsum("bik,bkx,bij->bi", pos_diff, cov_mats_inv, pos_diff)
    neg_maha = torch.einsum("bik,bkx,bij->bi", neg_diff, cov_mats_inv, neg_diff)

    pos_maha = torch.sqrt(pos_maha)
    neg_maha = torch.sqrt(neg_maha)

    return torch.stack([neg_maha, pos_maha], dim=-1)


def _torch_cov(tensor):
    # Shape: (batch_size, num_observations, num_features)
    batch_size, num_observations, num_features = tensor.shape

    # Center the data by subtracting the mean
    # Shape: (batch_size, 1, num_features)
    tensor_mean = torch.mean(tensor, dim=1, keepdim=True)
    # Shape: (batch_size, num_observations, num_features)
    tensor_centered = tensor - tensor_mean

    # Calculate covariance matrix
    cov_matrix = torch.matmul(tensor_centered.transpose(1, 2), tensor_centered) / (
        num_observations - 1
    )
    # Shape: (batch_size, num_features, num_features)

    return cov_matrix

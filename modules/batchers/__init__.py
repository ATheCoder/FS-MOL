###############################################################################
# Representation-specific batchers
###############################################################################
import torch
from torch_geometric.data import Batch


def fingerprint_batcher(batch):
    # For fingerprint-based tasks, each item is just a 1D vector.
    return torch.stack(batch, dim=0)

def pyg_batcher(batch):
    # For PyG data objects.
    return Batch.from_data_list(batch)

def descriptors_batcher(batch):
    return torch.stack(batch, dim=0)

def fingerprint_descriptors_batcher(batch):
    return torch.stack(batch, dim=0)

REPR_TO_BATCHER_MAP = {
    '2d': pyg_batcher,
    'fingerprint': fingerprint_batcher,
    # likewise for the new representation:
    'descriptors': descriptors_batcher,
    "fingerprint+descriptors": fingerprint_descriptors_batcher,
    '2d_gated': pyg_batcher,
    '3d_mxm': pyg_batcher,
    '3d_pam': pyg_batcher,
}
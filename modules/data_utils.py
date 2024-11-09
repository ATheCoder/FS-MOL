import numpy as np
import torch
from torch_geometric.data import Batch


def wide_collate_fn(batch):
    is_query_indices = torch.tensor(np.concatenate([b["is_query"] for b in batch]))
    input_sets = np.concatenate([b["mols"] for b in batch])
    input_features = [m.features for m in input_sets]
    batch_features = Batch.from_data_list(input_features)
    labels = torch.tensor([m.label for m in input_sets], dtype=torch.long)

    batch_index = torch.tensor([z for i, b in enumerate(batch) for z in [i] * len(b["mols"])])

    return {
        "graphs": batch_features,
        "labels": labels,
        "is_query": is_query_indices,
        "batch_index": batch_index,
        "batch_size": len(batch),
        "tasks": [b["task"] for b in batch if "task" in b],
    }

import numpy as np
import torch
from tqdm import tqdm
from fs_mol.data_modules.MXM_datamodule import MXMDataset
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


# val_ds = MXMValidationDataset("/FS-MOL/data/tarjan_11", support_size=16, split="test")

# for i in val_ds:
#     print(i)
#     pass


dataset = MXMDataset("/FS-MOL/data/mxm", "train", query_size=4, support_size=16, id="hei")

print(dataset[0])


def collate_fn(batch):
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


dl = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=False,
    num_workers=0,
    pin_memory=True,
)

# p = cProfile.Profile()
# p.enable()
for i in tqdm(iter(dl)):
    i
# p.disable()
# p.dump_stats("example_stats")
# stats = pstats.Stats("example_stats")
# stats.sort_stats("cumulative").print_stats(50)

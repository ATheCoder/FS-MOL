from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import pickle
import random
from typing import Literal
from joblib import Parallel, delayed
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

import numpy as np
import pandas as pd
from attr import dataclass
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from fewshot_utils.tqdm_joblib import tqdm_joblib
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from fs_mol.converters.smiles_to_mxm import preprocess_smile
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Since there are more samples inside a task than the number of tasks inside Tox21
# We will count a single epoch as seeing all the samples.
# It should be possible to set the number of support samples and query samples.
# Stratified sampling should be used for sampling for the support set


@dataclass()
class Tox21Sample:
    task_name: str
    label: bool
    features: Data


@dataclass
class Tox21Molecule:
    features: Data
    label: bool
    task_name: str


def generate_fewshot_inputs(arr, query_size, shuffle=False):
    arr = np.array(arr)
    if shuffle:
        np.random.shuffle(arr)

    query_set_buckets = [
        arr[i : i + query_size]
        for i in range(0, len(arr), query_size)
        if len(arr[i : i + query_size]) is query_size
    ]

    support_set_buckets = [np.setdiff1d(arr, bucket) for bucket in query_set_buckets]

    return list(zip(query_set_buckets, support_set_buckets))


class FewshotDataset(Dataset):
    def __init__(
        self,
        data,
        task_name_to_index_map,
        support_set_size=16,
        query_set_size=1,
        shuffle=False,
    ) -> None:
        super().__init__()

        self.data = np.array(data)
        self.should_shuffle = shuffle
        self.task_name_to_index = task_name_to_index_map
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size

        self.input_samples = self.generate_input_list()

    def generate_input_list(self):
        task_to_fewshot_input = {}

        for k, v in self.task_name_to_index.items():
            task_to_fewshot_input[k] = generate_fewshot_inputs(
                v, self.query_set_size, self.should_shuffle
            )

        ordered_task_to_fewshot_input = OrderedDict(
            sorted(task_to_fewshot_input.items(), key=lambda item: len(item[1]))
        )

        return [item for sublist in ordered_task_to_fewshot_input.values() for item in sublist]

    def sample_support_from_pool(self, support_set_candidates):
        support_set_candidate_labels = np.array([s.label for s in support_set_candidates])
        positive_indices = np.where(support_set_candidate_labels == 0)[0]
        negative_indices = np.where(support_set_candidate_labels == 1)[0]

        positive_samples = np.random.choice(positive_indices, 8)
        negative_samples = np.random.choice(negative_indices, 8)

        support_sample_indices = np.concatenate((positive_samples, negative_samples))

        if self.should_shuffle:
            np.random.shuffle(support_sample_indices)

        return (
            support_set_candidates[support_sample_indices],
            support_set_candidate_labels[support_sample_indices],
        )

        # sampler = StratifiedShuffleSplit(
        #     n_splits=1, test_size=min(len(support_set_candidates) - 2, self.support_set_size)
        # )

        # _, support_set_indices = list(
        #     sampler.split(support_set_candidates, support_set_candidate_labels)
        # )[0]
        # print(support_set_candidate_labels[support_set_indices])
        # return (
        #     support_set_candidates[support_set_indices],
        #     support_set_candidate_labels[support_set_indices],
        # )

    def __getitem__(self, index):
        query_set_indices, candidate_support_pool_indices = self.input_samples[index]

        query_set = self.data[query_set_indices]
        candidate_support_pool = self.data[candidate_support_pool_indices]

        support_set, support_labels = self.sample_support_from_pool(candidate_support_pool)

        mols = np.concatenate((query_set, support_set))

        return {
            "mols": mols,
            "is_query": [1] * len(query_set) + [0] * len(support_set),
            "task": "unknown",
        }

    def __len__(self):
        return len(self.input_samples)


class Tox21CSVProcessor:
    def __init__(self, csv_path) -> None:
        self.df = pd.read_csv(csv_path)

    def get_task_names(self):
        return self.df.columns[:-2]

    def get_smiles_for_task(self, task_name):
        assert task_name in self.get_task_names(), f"{task_name} is not a valid task_name"
        for index, row in self.df.iterrows():
            if pd.isna(row[task_name]):
                continue
            yield row["smiles"], row[task_name]


def undersample_balance(candidate_support_indices, support_set_candidate_labels):
    rus = RandomUnderSampler()
    X, Y = rus.fit_resample(candidate_support_indices.reshape(-1, 1), support_set_candidate_labels)  # type: ignore
    candidate_support_indices = X.reshape(-1)  # type: ignore
    support_set_candidate_labels = Y

    return candidate_support_indices, support_set_candidate_labels


def oversample_balance(candidate_support_indices, support_set_candidate_labels):
    rus = RandomOverSampler()
    X, Y = rus.fit_resample(candidate_support_indices.reshape(-1, 1), support_set_candidate_labels)  # type: ignore
    candidate_support_indices = X.reshape(-1)  # type: ignore
    support_set_candidate_labels = Y

    return candidate_support_indices, support_set_candidate_labels


def generate_balanced_sample(
    candidate_support_indices, support_set_candidate_labels, support_set_size
):
    assert support_set_size % 2 == 0
    positive_indices = candidate_support_indices[support_set_candidate_labels == 1]
    positive_sample = np.random.choice(positive_indices, support_set_size // 2)
    negative_indices = candidate_support_indices[support_set_candidate_labels == 0]
    negative_sample = np.random.choice(negative_indices, support_set_size // 2)

    support_set = np.concatenate((positive_sample, negative_sample))
    remaining = np.setdiff1d(candidate_support_indices, support_set)
    np.random.shuffle(support_set)

    return support_set, remaining


class Tox21Dataset(Dataset):
    def __init__(
        self,
        root_path,
        fold: str,
        query_size,
        support_size,
        shuffle=True,
        use_subgraphs=False,
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path) / fold
        self.query_size = query_size
        self.shuffle = shuffle
        self.support_size = support_size
        self.use_subgraphs = use_subgraphs

        self.task_names = np.array(os.listdir(self.root_path), dtype=np.string_)
        self.query_per_task = np.array(
            [
                int(str(task_name, encoding="utf-8").split("-")[-1]) // self.query_size
                for task_name in self.task_names
            ]
        )
        self.acc_query_per_task = np.cumsum(self.query_per_task)

    def sample_support_set_from_candidates(
        self, candidate_support_indices, support_set_candidate_labels, sample_size
    ):
        candidate_support_indices, support_set_candidate_labels = undersample_balance(
            candidate_support_indices, support_set_candidate_labels
        )
        sampler = StratifiedShuffleSplit(
            1, test_size=min(len(support_set_candidate_labels) - 2, sample_size)
        )

        remaining_indices, support_set_indices = list(
            sampler.split(candidate_support_indices, support_set_candidate_labels)
        )[0]

        del sampler

        return (
            candidate_support_indices[support_set_indices],
            candidate_support_indices[remaining_indices],
        )

    def get_task_name(self, index):
        indices = np.where(self.acc_query_per_task > index)
        idx_in_tasks = indices[0][0]

        prev_task_count = self.acc_query_per_task[idx_in_tasks - 1] if idx_in_tasks >= 1 else 0

        task_name = str(self.task_names[idx_in_tasks], encoding="utf-8")

        return task_name, index - prev_task_count

    def generate_q_s_indices(self, length, query_idx, labels):
        # assert self.query_size == 1
        arr = np.arange(length)
        query_indices = arr[query_idx * self.query_size : (query_idx + 1) * self.query_size]
        arr = np.setdiff1d(arr, query_indices)
        p_arr = np.random.permutation(arr)

        support_candidate_indices = p_arr

        support_indices, _ = self.sample_support_set_from_candidates(
            support_candidate_indices, labels[support_candidate_indices], self.support_size
        )

        return query_indices, support_indices

    def random_generate_q_s_indices(self, length, labels):
        arr = np.arange(length)
        support_indices, all_except_support = generate_balanced_sample(
            arr, labels, self.support_size
        )

        np.random.shuffle(all_except_support)

        query_indices = all_except_support[: self.query_size]

        return query_indices, support_indices

    def load_task_sample(self, task_name, idx, label):
        if self.use_subgraphs:
            random_division = random.choice(os.listdir(self.root_path / task_name / f"{idx}"))
            # print(f"Random Devision chosen: {random_division}")
            task_sample = torch.load(self.root_path / task_name / f"{idx}" / random_division)
            return task_sample

        task_sample = torch.load(self.root_path / task_name / f"{idx}.pt")
        return Tox21Molecule(task_sample, label, task_name)

    def __getitem__(self, index):
        task_name, i = self.get_task_name(index)
        task_count = int(task_name.split("-")[-1])

        labels = torch.load(self.root_path / task_name / "labels.pt")
        labels = np.array(labels)
        query_indices, raw_support_candidates = self.random_generate_q_s_indices(task_count, labels)

        query_samples = [self.load_task_sample(task_name, s, labels[s]) for s in query_indices]
        support_samples = [
            self.load_task_sample(task_name, s, labels[s]) for s in raw_support_candidates
        ]

        return {
            "mols": np.concatenate((query_samples, support_samples)),  # type: ignore
            "is_query": [1] * len(query_samples) + [0] * len(support_samples),
            "task": task_name,
        }

    def __len__(self):
        return self.acc_query_per_task[-1]


class Tox21ValidationDataset(Dataset):
    def __init__(
        self, root_path, support_size=16, split="valid", use_subgraph=False, query_set_size=32
    ) -> None:
        self.support_size = support_size
        self.query_set_size = query_set_size
        self.root_path = Path(root_path) / split

        self.task_names = np.array(os.listdir(self.root_path), dtype=np.string_)
        self.samples_in_task = np.array(
            [int(str(task_name, encoding="utf-8").split("-")[-1]) for task_name in self.task_names]
        )
        self.use_subgraph = use_subgraph

    def open_task_file(self, file_name):
        file_path = self.root_path / file_name
        # files = os.listdir(root_dir)
        # loaded = [torch.load(root_dir / f) for f in files]
        return np.array(torch.load(file_path / "0.pt"))

    def generate_support_query(self, task_name, sample_labels):
        task_size = int(task_name.split("-")[-1])

        sample_indices = np.arange(task_size)

        support_set, all_except_support = generate_balanced_sample(
            sample_indices, sample_labels, self.support_size
        )

        np.random.shuffle(all_except_support)

        return (
            support_set,
            all_except_support[: self.query_set_size],
        )

    def load_sample(self, task_name, idx, label):
        if self.use_subgraph:
            sample_path = self.root_path / task_name / f"{idx}" / f"0.pt"
            return torch.load(sample_path)
        sample_path = self.root_path / task_name / f"{idx}.pt"

        task_sample = torch.load(sample_path)

        return Tox21Molecule(task_sample, label, task_name)

    def __getitem__(self, index):
        index = index % len(self.task_names)
        task_name = str(self.task_names[index], encoding="utf-8")

        task_labels = np.array(torch.load(self.root_path / task_name / "labels.pt"))

        support_set_indices, query_set_indices = self.generate_support_query(task_name, task_labels)

        support_set = [self.load_sample(task_name, s, task_labels[s]) for s in support_set_indices]
        query_set = [self.load_sample(task_name, s, task_labels[s]) for s in query_set_indices]

        return {
            "mols": np.concatenate((query_set, support_set)),
            "is_query": [1] * len(query_set) + [0] * len(support_set),
            "task_name": task_name,
        }

    def __len__(self):
        return len(self.task_names) * 1000


class Tox21FewshotDataModule(LightningDataModule):
    def __init__(
        self,
        data_folder,
        csv_path,
        support_size=16,
        query_size=16,
        shuffle=True,
        valid_support_size=16,
        valid_query_size=16,
        batch_size=1,
        train_num_workers=6,
    ) -> None:
        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.valid_support_size = valid_support_size
        self.valid_query_size = valid_query_size
        self.train_support_size = support_size
        self.train_query_size = query_size
        self.root_path = Path(data_folder)
        self.train_num_workers = train_num_workers

        self.train_task_names = [
            "NR-AR",
            "NR-AR-LBD",
            "NR-AhR",
            "NR-Aromatase",
            "NR-ER",
            "NR-ER-LBD",
            "NR-PPAR-gamma",
            "SR-ARE",
            "SR-ATAD5",
        ]

        self.test_task_names = [
            "SR-HSE",
            "SR-MMP",
            "SR-p53",
        ]

        self.csv_processor = Tox21CSVProcessor(csv_path)

    def get_task_name_for_fold(self, fold: Literal["train", "test", "valid"]):
        if fold == "train":
            return self.train_task_names
        else:
            return self.test_task_names

    def prepare_mols(self, fold):
        for task_name in self.get_task_name_for_fold(fold):
            smiles_labels_for_task = list(self.csv_processor.get_smiles_for_task(task_name))
            smiles_for_task = [i[0] for i in smiles_labels_for_task]
            labels = [i[1] for i in smiles_labels_for_task]

            with tqdm_joblib(
                tqdm(f'Samples processed for "{task_name}"', total=len(smiles_for_task))
            ):
                samples = Parallel(n_jobs=8)(
                    delayed(self.process_sample)(smiles, labels[i])
                    for i, smiles in enumerate(smiles_for_task)
                )

                samples = [sample for sample in samples if sample != None]

                task_path = self.root_path / fold / f"{task_name}-{len(samples)}"

                os.makedirs(task_path)
                for i, sample in enumerate(samples):
                    torch.save(sample[0], task_path / f"{i}.pt")
                labels = [sample[1] for sample in samples]
                torch.save(labels, task_path / "labels.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def process_sample(self, smiles, label):
        try:
            mol = preprocess_smile(smiles)
            return mol, label
        except Exception as e:
            print(f"Could not process sample {smiles} because of {e}")
            return None

    def get_data_fold_path(self, datafold: Literal["train", "test", "valid"]):
        return self.root_path / datafold

    def prepare_data_for_fold(self, datafold: Literal["train", "test", "valid"]):
        datafold_path = self.get_data_fold_path(datafold)

        if datafold_path.exists():
            return

        os.makedirs(datafold_path, exist_ok=True)
        self.prepare_mols(datafold)

    def prepare_data(self) -> None:
        self.prepare_data_for_fold("train")
        self.prepare_data_for_fold("test")

    def setup(self, stage: str):
        self.train = Tox21Dataset(
            self.root_path,
            "train",
            self.train_support_size,
            self.train_query_size,
            self.shuffle,
        )

        self.valid = Tox21ValidationDataset(
            self.root_path,
            support_size=self.valid_support_size,
            split="test",
            query_set_size=self.valid_query_size,
        )

    def collate_fn(self, batch):
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
            "tasks": [b["task_name"] for b in batch if "task_name" in b],
        }

    def train_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            num_workers=self.train_num_workers,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()

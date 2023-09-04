import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import pickle
import random
from typing import List

import numpy as np
from lightning.pytorch.core import LightningDataModule
import torch
from joblib import Parallel, delayed
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm

from fewshot_utils.tqdm_joblib import tqdm_joblib
from fs_mol.converters.smiles_to_mxm import preprocess_smile
from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from preprocessing.geometric import get_all_divisions
from torch_geometric.data import Data


@dataclass
class MXMNetMolecule:
    features: Data
    label: bool
    task_name: str


@dataclass
class MoleculeWithCompleteDivisions(MXMNetMolecule):
    features: List[List[Data]]


@dataclass
class MoleculeWithOneDivision(MXMNetMolecule):
    features: List[Data]


@dataclass
class MXMNetMoleculeWithSmiles:
    smiles: str
    label: bool
    task_name: str


class MXMDataset(Dataset):
    def __init__(
        self,
        root_path,
        fold: str,
        query_size,
        support_size,
        shuffle=True,
        id=None,
        use_subgraphs=False,
    ) -> None:
        super().__init__()
        self.id = id
        self.root_path = Path(root_path) / fold
        self.query_size = query_size
        self.shuffle = shuffle
        self.support_size = support_size
        self.use_subgraphs = use_subgraphs

        self.task_names = np.array(os.listdir(self.root_path), dtype=np.string_)
        self.query_per_task = np.array(
            [
                int(str(task_name, encoding="utf-8").split("-")[1]) // self.query_size
                for task_name in self.task_names
            ]
        )
        self.acc_query_per_task = np.cumsum(self.query_per_task)

    def sample_support_set_from_candidates(
        self, candidate_support_indices, support_set_candidate_labels
    ):
        sampler = StratifiedShuffleSplit(
            1, test_size=min(len(support_set_candidate_labels) - 2, self.support_size)
        )

        _, support_set_indices = list(
            sampler.split(candidate_support_indices, support_set_candidate_labels)
        )[0]

        del sampler

        return candidate_support_indices[support_set_indices]

    def get_task_name(self, index):
        indices = np.where(self.acc_query_per_task > index)
        idx_in_tasks = indices[0][0]

        prev_task_count = self.acc_query_per_task[idx_in_tasks - 1] if idx_in_tasks >= 1 else 0

        task_name = str(self.task_names[idx_in_tasks], encoding="utf-8")

        return task_name, index - prev_task_count

    def generate_q_s_indices(self, length, query_idx, labels):
        # assert self.query_size == 1
        arr = np.arange(length)
        arr = arr[arr != query_idx]
        p_arr = np.random.permutation(arr)

        query_indices = np.array([query_idx])
        support_candidate_indices = p_arr

        support_indices = self.sample_support_set_from_candidates(
            support_candidate_indices, labels[support_candidate_indices]
        )

        return query_indices, support_indices

    def load_task_sample(self, task_name, idx, label):
        if self.use_subgraphs:
            random_division = random.choice(os.listdir(self.root_path / task_name / f"{idx}"))
            # print(f"Random Devision chosen: {random_division}")
            task_sample = torch.load(self.root_path / task_name / f"{idx}" / random_division)
            return task_sample

        task_sample = torch.load(self.root_path / task_name / f"{idx}.pt")
        return task_sample

    def __getitem__(self, index):
        task_name, i = self.get_task_name(index)
        task_count = int(task_name.split("-")[1])

        labels = torch.load(self.root_path / task_name / "labels.pt")
        labels = np.array(labels)
        query_indices, raw_support_candidates = self.generate_q_s_indices(task_count, i, labels)

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


class MXMValidationDataset(Dataset):
    def __init__(self, root_path, support_size=16, split="valid", use_subgraph=False) -> None:
        self.support_size = 16
        self.root_path = Path(root_path) / split

        self.task_names = np.array(os.listdir(self.root_path), dtype=np.string_)
        self.samples_in_task = np.array(
            [int(str(task_name, encoding="utf-8").split("-")[1]) for task_name in self.task_names]
        )
        self.use_subgraph = use_subgraph

    @lru_cache(2_400)
    def open_task_file(self, file_name):
        file_path = self.root_path / file_name
        # files = os.listdir(root_dir)
        # loaded = [torch.load(root_dir / f) for f in files]
        return np.array(torch.load(file_path / "0.pt"))

    def generate_support_query(self, task_name, sample_labels):
        task_size = int(task_name.split("-")[1])

        sample_indices = np.arange(task_size)
        sampler = StratifiedShuffleSplit(1, test_size=min(task_size - 2, self.support_size))
        query_set_indices, support_set_indices = list(sampler.split(sample_indices, sample_labels))[
            0
        ]

        np.random.shuffle(query_set_indices)

        return sample_indices[support_set_indices], sample_indices[query_set_indices[:16]]

    def load_sample(self, task_name, idx):
        if self.use_subgraph:
            sample_path = self.root_path / task_name / f"{idx}" / f"0.pt"
            return torch.load(sample_path)
        sample_path = self.root_path / task_name / f"{idx}.pt"

        return torch.load(sample_path)

    def __getitem__(self, index):
        task_name = str(self.task_names[index], encoding="utf-8")

        task_labels = torch.load(self.root_path / task_name / "labels.pt")

        support_set_indices, query_set_indices = self.generate_support_query(task_name, task_labels)

        support_set = [self.load_sample(task_name, s) for s in support_set_indices]
        query_set = [self.load_sample(task_name, s) for s in query_set_indices]

        return {
            "mols": np.concatenate((query_set, support_set)),
            "is_query": [1] * len(query_set) + [0] * len(support_set),
        }

    def __len__(self):
        return len(self.task_names)


def unique_based_on_smiles(samples):
    seen_mols = set()

    return [seen_mols.add(s.smiles) or s for s in samples if s.smiles not in seen_mols]


class MXMDataModule(LightningDataModule):
    def __init__(
        self,
        root_path,
        original_fsmol_path="/FS-MOL/datasets/fs-mol/",
        fsmol_task_json="/FS-MOL/datasets/mxm_tasks.json",
        query_size=16,
        batch_size=8,
        shuffle=True,
        support_size=16,
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.query_size = query_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.support_size = support_size

        self.original_fsmol_path = original_fsmol_path
        self.fsmol_task_json = fsmol_task_json

        self.raw_mol_path = self.root_path / "raw_mol.pt"
        self.index_map_path = self.root_path / "index_map.pt"

        os.makedirs(self.root_path, exist_ok=True)

    def get_data_fold_path(self, fold: DataFold):
        if fold == DataFold.TRAIN:
            return self.root_path / "train"
        elif fold == DataFold.TEST:
            return self.root_path / "test"
        elif fold == DataFold.VALIDATION:
            return self.root_path / "valid"

    def generate_raw_data(self, datafold: DataFold):
        datafold_path = self.get_data_fold_path(datafold)
        if not datafold_path.exists():
            os.makedirs(datafold_path, exist_ok=True)
            fsmol_dataset = FSMolDataset.from_directory(
                str(self.original_fsmol_path), task_list_file=str(self.fsmol_task_json)
            )

            iterator = iter(fsmol_dataset.get_task_reading_iterable(datafold))

            tasks = list(iterator)
            if True:
                with tqdm_joblib(tqdm("FSMOL Tasks processed", total=len(tasks))):
                    Parallel(n_jobs=16)(
                        delayed(self.process_task)(task, datafold_path) for task in tasks
                    )
            else:
                for task in tasks:
                    self.process_task(task, datafold_path)

    def process_smiles(self, mxm_net_smiles: MXMNetMoleculeWithSmiles):
        try:
            features = preprocess_smile(mxm_net_smiles.smiles, True)
            return MXMNetMolecule(
                features=features, label=mxm_net_smiles.label, task_name=mxm_net_smiles.task_name
            )
        except ValueError as e:
            print(f"Could not process: {mxm_net_smiles.smiles}")

    def process_task(self, task, datafold_path):
        samples = [
            MXMNetMoleculeWithSmiles(
                smiles=sample.smiles,
                label=sample.bool_label,
                task_name=task.name,
            )
            for sample in task.samples
        ]

        converted_samples = [self.process_smiles(sample) for sample in samples]
        converted_samples = [sample for sample in converted_samples if sample is not None]
        labels = [sample.label for sample in converted_samples]
        task_path = datafold_path / f"{task.name}-{len(converted_samples)}"
        os.makedirs(task_path, exist_ok=True)
        for i, s in enumerate(converted_samples):
            f_n = f"{i}.pt"
            torch.save(s, task_path / f_n, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        torch.save(labels, task_path / "labels.pt")

    def prepare_data(self) -> None:
        for fold in [DataFold.TRAIN, DataFold.TEST, DataFold.VALIDATION]:
            self.generate_raw_data(fold)
        # self.task_to_mols_map = self.generate_index_map()

    def setup(self, stage):
        print("Stage is: ", stage)
        if stage == "fit":
            self.train = MXMDataset(
                self.root_path,  # /FS-MOL/data/tarjan_5/
                "train",
                query_size=self.query_size,
                support_size=self.support_size,
                id="hei",
            )
            self.valid = MXMValidationDataset(
                self.root_path, support_size=self.support_size, split="test"
            )
        else:
            self.test = MXMValidationDataset(
                self.root_path,
                support_size=self.support_size,
                split="test",
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
            "tasks": [b["task"] for b in batch if "task" in b],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            drop_last=False,
            num_workers=6,
            pin_memory=True,
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
        return DataLoader(
            self.test,
            batch_size=1,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            drop_last=True,
        )


class TarjanDataModule(MXMDataModule):
    def processMXMNetMolecule(self, m: MXMNetMolecule):
        all_divisions_of_mol = MoleculeWithCompleteDivisions(
            features=get_all_divisions(m.features, 10), label=m.label, task_name=m.task_name
        )

        return [
            MoleculeWithOneDivision(features=feat, label=m.label, task_name=m.task_name)
            for feat in all_divisions_of_mol.features
        ]

    def process_task(self, task, datafold_path):
        samples = [
            MXMNetMoleculeWithSmiles(
                smiles=sample.smiles,
                label=sample.bool_label,
                task_name=task.name,
            )
            for sample in task.samples
        ]
        samples = unique_based_on_smiles(samples)

        converted_samples = [self.process_smiles(sample) for sample in samples]
        converted_samples = [sample for sample in converted_samples if sample is not None]
        converted_samples = [
            sample for sample in converted_samples if sample.features.x.shape[0] > 1
        ]

        converted_samples = [self.processMXMNetMolecule(sample) for sample in converted_samples]

        labels = [a[0].label for a in converted_samples]

        for i, sample in enumerate(converted_samples):
            sample_path = datafold_path / f"{task.name}-{len(converted_samples)}" / f"{i}"
            os.makedirs(sample_path, exist_ok=True)
            for j, div in enumerate(sample):
                torch.save(
                    div,
                    sample_path / f"{j}.pt",
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

        torch.save(
            labels,
            datafold_path / f"{task.name}-{len(converted_samples)}" / "labels.pt",
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

    def collate_fn(self, batch):
        task_names = [b["mols"][0].task_name for b in batch]
        molecules = np.concatenate([b["mols"] for b in batch])
        batch_index = torch.tensor(
            [z for i, b in enumerate(batch) for z in [i] * len(b["mols"])], dtype=torch.long
        )
        labels = torch.tensor([int(m.label) for m in molecules], dtype=torch.long)
        is_query_indices = torch.tensor(np.concatenate([b["is_query"] for b in batch]))

        sub_structure_to_mol_index = [
            z for i, mol in enumerate(molecules) for z in [i] * len(mol.features)
        ]

        sub_structure_graphs = [
            subgraph for b in batch for mol in b["mols"] for subgraph in mol.features
        ]

        sub_structure_graphs = Batch.from_data_list(sub_structure_graphs)

        return {
            "substructures": sub_structure_graphs,
            "substructure_mol_index": torch.tensor(sub_structure_to_mol_index, dtype=torch.long),
            "is_query": is_query_indices,
            "labels": labels,
            "batch_index": batch_index,
            "tasks": task_names,
        }

    def setup(self, stage):
        print("Stage is: ", stage)
        if stage == "fit":
            self.train = MXMDataset(
                self.root_path,  # /FS-MOL/data/tarjan_5/
                "train",
                query_size=self.query_size,
                support_size=self.support_size,
                id="hei",
                use_subgraphs=True,
            )
            self.valid = MXMValidationDataset(
                self.root_path, support_size=self.support_size, split="test", use_subgraph=True
            )
        else:
            self.test = MXMValidationDataset(
                self.root_path, support_size=self.support_size, split="test", use_subgraph=True
            )

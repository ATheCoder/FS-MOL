import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset


@dataclass()
class MHNFSMolecule:
    features: torch.Tensor
    label: bool
    assayId: str


class MHNFSTrainingDataset(Dataset):
    def __init__(self, root_file, fold_name="train", support_set_size=16) -> None:
        super().__init__()
        self.support_set_size = support_set_size

        self.root_path = Path(root_file)
        # The complete dataset. (The triplets)
        self.mols = torch.load(self.root_path / fold_name / "triplets.pt")
        # A map from targetId to indices in the main dataset that are for this molecule
        self.assayId_to_molIndices = torch.load(self.root_path / fold_name / "assay_to_mol_map.pt")

        self.padding_size = 12

    def add_padding(self, tensor):
        return F.pad(
            tensor,
            (0, 0, 0, self.padding_size - tensor.shape[0]),
        )

    def __getitem__(self, index):
        query_mol = self.mols[index]
        current_assay = query_mol.assayId

        mol_indices = self.assayId_to_molIndices[current_assay]

        # Remove the query mol:
        mol_indices = np.delete(mol_indices, np.where(mol_indices == index))

        support_set_candidates = self.mols[mol_indices]

        support_set_candidate_labels = [mol.label for mol in support_set_candidates]

        sampler = StratifiedShuffleSplit(
            1, test_size=min(len(support_set_candidates) - 2, self.support_set_size)
        )

        _, support_set_indices = list(
            sampler.split(support_set_candidates, support_set_candidate_labels)
        )[0]

        support_set = support_set_candidates[support_set_indices]

        positive_support_set = [mol for mol in support_set if mol.label == True]
        negative_support_set = [mol for mol in support_set if mol.label == False]

        # Question: Why do we need to pad when all batches will have 8 positive and 8 negative?
        return {
            "query": query_mol.features,
            "positive_support": self.add_padding(
                torch.vstack([mol.features for mol in positive_support_set])
            ),
            "negative_support": self.add_padding(
                torch.vstack([mol.features for mol in negative_support_set])
            ),
            "positive_support_count": torch.tensor(len(positive_support_set), dtype=torch.long),
            "negative_support_count": torch.tensor(len(negative_support_set), dtype=torch.long),
            "query_labels": torch.tensor(query_mol.label, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.mols)


class MHNFSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir,
        root_fsmol_dir="/FS-MOL/datasets/fs-mol/",
        support_set_size=16,
        task_list_file="/FS-MOL/datasets/mxm_tasks.json",
    ) -> None:
        super().__init__()
        self.support_set_size = support_set_size
        self.task_list_file = task_list_file

        self.root_fsmol_path = Path(root_fsmol_dir)
        self.root = Path(dataset_dir)
        self.descriptor_transformer = torch.load("/FS-MOL/example_transformer.pt")

    def prepare_data_from_fsmol(self, fold):
        fold_to_folder = {
            DataFold.TRAIN: "train",
            DataFold.VALIDATION: "valid",
            DataFold.TEST: "test",
        }

        fold_name = fold_to_folder[fold]

        if (
            not (self.root / fold_name / "triplets.pt").exists()
            or not (self.root / fold_name / "assay_to_mol_map.pt").exists()
        ):
            fsmol_dataset = FSMolDataset.from_directory(
                str(self.root_fsmol_path.absolute()), task_list_file=self.task_list_file
            )

            tasks = iter(fsmol_dataset.get_task_reading_iterable(fold))

            triplets = []

            for task in tqdm(tasks):
                for sample in task.samples:
                    descriptors = self.descriptor_transformer.transform(
                        sample.descriptors.reshape(1, -1)
                    )[0]
                    descriptors = torch.from_numpy(descriptors)
                    fingerprint = torch.from_numpy(sample.fingerprint)
                    MHNFS_mol = MHNFSMolecule(
                        features=torch.cat([descriptors, fingerprint], dim=-1),
                        label=sample.bool_label,
                        assayId=task.name,
                    )

                    triplets.append(MHNFS_mol)

            assay_to_mol_map = {}

            for i, m in enumerate(triplets):
                assay_id = m.assayId
                if not assay_id in assay_to_mol_map:
                    assay_to_mol_map[assay_id] = [i]
                else:
                    assay_to_mol_map[assay_id].append(i)

            for key in assay_to_mol_map:
                assay_to_mol_map[key] = np.array(assay_to_mol_map[key])

            triplet_save_path = self.root / fold_name / "triplets.pt"
            os.makedirs(triplet_save_path.parent, exist_ok=True)
            torch.save(np.array(triplets), triplet_save_path)

            assay_to_mol_path = self.root / fold_name / "assay_to_mol_map.pt"
            os.makedirs(assay_to_mol_path.parent, exist_ok=True)
            torch.save(assay_to_mol_map, assay_to_mol_path)

    def prepare_data(self) -> None:
        datafolds = [DataFold.TRAIN, DataFold.TEST, DataFold.VALIDATION]
        for datafold in datafolds:
            self.prepare_data_from_fsmol(datafold)

    def setup(self, stage: str):
        if stage == "fit":
            self.train = MHNFSTrainingDataset(
                self.root, fold_name="train", support_set_size=self.support_set_size
            )
            self.valid = MHNFSValidationDataset(self.root, support_set_size=self.support_set_size)

        if stage == "test":
            self.test = MHNFSTrainingDataset(
                self.root, fold_name="test", support_set_size=self.support_set_size
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=512, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=1)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=256)


class MHNFSValidationDataset(Dataset):
    def __init__(self, root_file, support_set_size=16) -> None:
        super().__init__()
        self.support_set_size = support_set_size

        self.root_path = Path(root_file)

        self.mols = torch.load(self.root_path / "valid" / "triplets.pt")

        self.assayId_to_molIndices = torch.load(self.root_path / "valid" / "assay_to_mol_map.pt")

        self.assayIds = list(self.assayId_to_molIndices.keys())

        self.padding_size = 12

    def sample_query_and_support(self, mols: List[MHNFSMolecule]):
        mol_labels = [mol.label for mol in mols]

        # Remaining is query_set
        support_set, remaining_set = train_test_split(
            mols, stratify=mol_labels, test_size=(len(mols) - self.support_set_size) / len(mols)
        )

        return support_set, remaining_set

    def add_padding(self, tensor):
        return F.pad(
            tensor,
            (0, 0, 0, self.padding_size - tensor.shape[0]),
        )

    def __getitem__(self, index):
        current_assay = self.assayIds[index]

        current_assay_molecule_indices = self.assayId_to_molIndices[current_assay]

        current_assay_mols = self.mols[current_assay_molecule_indices]

        support_set, query_set = self.sample_query_and_support(current_assay_mols)

        positive_support = [mol for mol in support_set if mol.label == True]
        negative_support = [mol for mol in support_set if mol.label == False]

        return {
            "query": torch.vstack([mol.features for mol in query_set]),
            "positive_support": self.add_padding(
                torch.vstack([mol.features for mol in positive_support]),
            ),
            "negative_support": self.add_padding(
                torch.vstack([mol.features for mol in negative_support])
            ),
            "positive_support_count": torch.tensor(len(positive_support), dtype=torch.long),
            "negative_support_count": torch.tensor(len(negative_support), dtype=torch.long),
            "query_labels": torch.tensor([mol.label for mol in query_set], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.assayIds)

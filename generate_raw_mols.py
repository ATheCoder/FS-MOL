from os import makedirs
from pathlib import Path
import pickle
from joblib import Parallel, delayed
import torch

from tqdm import tqdm
from fewshot_utils.tqdm_joblib import tqdm_joblib
from fs_mol.converters.smiles_to_mxm import preprocess_smile

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from fs_mol.data_modules.MXM_datamodule import (
    MXMNetMolecule,
    MXMNetMoleculeWithSmiles,
    MoleculeWithOneDivision,
)
from preprocessing.geometric import get_all_divisions


dest_path = Path("/FS-MOL/data/tarjan_11")


fsmol_dataset = FSMolDataset.from_directory(
    "/FS-MOL/datasets/fs-mol/", "/FS-MOL/datasets/mxm_tasks.json"
)

iterator = iter(fsmol_dataset.get_task_reading_iterable(DataFold.TEST))

tasks = list(iterator)


def process_smiles(mxm_net_smiles: MXMNetMoleculeWithSmiles):
    try:
        features = preprocess_smile(mxm_net_smiles.smiles)
        return MXMNetMolecule(
            features=features, label=mxm_net_smiles.label, task_name=mxm_net_smiles.task_name
        )
    except ValueError as e:
        print(f"Could not process: {mxm_net_smiles.smiles}")


makedirs(dest_path / "test", exist_ok=True)


def process_task(task):
    samples = [
        MXMNetMoleculeWithSmiles(
            smiles=sample.smiles,
            label=sample.bool_label,
            task_name=task.name,
        )
        for sample in task.samples
    ]

    converted_samples = [process_smiles(sample) for sample in samples]
    converted_samples = [sample for sample in converted_samples if sample is not None]
    converted_samples = [sample for sample in converted_samples if sample.features.x.shape[0] > 1]

    task_folder = dest_path / "test" / task.name

    # makedirs(task_folder, exist_ok=True)

    labels = [s.label for s in converted_samples]
    converted_samples = [
        MoleculeWithOneDivision(
            features=get_all_divisions(m.features, 10)[-1], label=m.label, task_name=m.task_name
        )
        for m in converted_samples
    ]

    torch.save(
        converted_samples, dest_path / f"{task_folder}.pt", pickle_protocol=pickle.HIGHEST_PROTOCOL
    )

    # torch.save(labels, task_folder / "labels.pt")


with tqdm_joblib(tqdm("FSMOL Tasks processed", total=len(tasks))):
    Parallel(n_jobs=16)(delayed(process_task)(task) for task in tasks)

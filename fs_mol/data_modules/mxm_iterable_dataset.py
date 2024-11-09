import os
from itertools import accumulate
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import IterableDataset
from tqdm import tqdm

from fewshot_utils.torch_cache import torch_cache
# from fs_mol.data_modules.MXM_datamodule import MXMNetMolecule
from fs_mol.data_modules.tox21_datamodule import generate_fewshot_inputs

class MXMIterableDataset(IterableDataset):
    def __init__(
        self, root_path, fold: str, query_size, support_size, shuffle=True, id=None
    ) -> None:
        super().__init__()
        print("INIT!")
        self.id = id
        self.root_path = Path(root_path) / fold
        self.query_size = query_size
        self.shuffle = shuffle
        self.support_size = support_size

        self.count_to_file_name, self.file_name_to_query_support_indices = self.generate_task_list()
        # self.task = self.load_tasks()
        # self.task = torch.save(self.task, self.root_path / 'all.pt', pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def get_task_sample_length(self, path):
        task = os.listdir(path)

        return len(task)
    
    
    def load_tasks(self):
        file_names = os.listdir(self.root_path)
        samples = [torch.load(self.root_path / file_name) for file_name in tqdm(file_names)]
        
        return dict(zip(file_names, samples))

    @torch_cache(lambda self, *arg, **kwargs: self.root_path / "../cached/task_name_length.pt")
    def get_task_names_and_lengths(self):
        file_names = os.listdir(self.root_path)
        length_of_samples = [
            self.get_task_sample_length(self.root_path / file_name)
            for file_name in tqdm(file_names)
        ]

        return file_names, length_of_samples

    def generate_task_list(self):
        file_names, length_of_samples = self.get_task_names_and_lengths()

        query_support_indices_for_all_tasks = [
            generate_fewshot_inputs(np.arange(size), self.query_size, False)
            for size in length_of_samples
        ]

        query_lengths = [
            len(query_indices) for query_indices in query_support_indices_for_all_tasks
        ]

        acc_length_of_samples = list(accumulate(query_lengths))

        count_to_file_name = OrderedDict(zip(acc_length_of_samples, file_names))
        file_name_to_query_support_indices = OrderedDict(
            zip(file_names, query_support_indices_for_all_tasks)
        )

        return count_to_file_name, file_name_to_query_support_indices

    def find_task_name_from_index(self, index):
        counts = self.count_to_file_name.keys()
        for i, k in enumerate(counts):
            if index < k:
                if i - 1 >= 0:
                    prev_key = list(counts)[i - 1]
                    return self.count_to_file_name[k], index - prev_key
                else:
                    return self.count_to_file_name[k], index

    # @lru_cache(1_200)
    def open_task_file(self, file_name):
        # return np.array(self.task[file_name])
        loaded = torch.load(self.root_path / file_name)
        # loaded = np.array([MXMNetMolecule(a["features"].to_data_list(), a["label"], a["task_name"]) for a in loaded])
        return np.array(loaded)

    def sample_support_set_from_candidates(self, support_set_candidates):
        support_candidate_labels = [s.label for s in support_set_candidates]
        sampler = StratifiedShuffleSplit(
            1, test_size=min(len(support_set_candidates) - 2, self.support_size)
        )

        _, support_set_indices = list(
            sampler.split(support_set_candidates, support_candidate_labels)
        )[0]

        support_set = support_set_candidates[support_set_indices]

        return support_set
    
    def open_task_indices(self, task_name, indices):
        pathz = [self.root_path / task_name / f"{indice}.pt" for indice in indices]
        data = []
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     for d in executor.map(self.open_task_in_indice, pathz):
        #         data.append(d)
        return np.array([self.open_task_in_indice(d) for d in pathz])

    def __iter__(self):
        for index in range(list(self.count_to_file_name.keys())[-1] - 1):
            file_name, index_in_task = self.find_task_name_from_index(index)
            # wandb.log({"task_name": file_name})

            query_indices, support_candidate_indices = self.file_name_to_query_support_indices[
                file_name
            ][index_in_task]
            file_name_p = f'{file_name}.pt'
            
            task = self.open_task_file(file_name_p)
        
            query_samples = task[query_indices]
            # candidate_samples = self.open_task_indices(file_name, support_candidate_indices)
            support_samples = self.sample_support_set_from_candidates(task[support_candidate_indices])

            yield {
                "mols": np.concatenate((query_samples, support_samples)),
                "is_query": [1] * len(query_samples) + [0] * len(support_samples),
                "task": file_name,
            }

    def __len__(self):
        return list(self.count_to_file_name.keys())[-1] - 1


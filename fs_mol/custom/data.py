from typing import List
from dpu_utils.utils.richpath import RichPath
from fs_mol.data.fsmol_dataset import DataFold
from fs_mol.data.fsmol_task import FSMolTask
from fs_mol.data.fsmol_task_sampler import StratifiedTaskSampler


def generate_episodic_iterable(dataset, repeat = False):
    if dataset._num_workers > 0:
        raise Exception("Number of Workers should be `0` on the dataset when using the episodic iterable.")
        
    task_sampler = StratifiedTaskSampler(
        train_size_or_ratio=64, test_size_or_ratio=256
    )

    def simple_task_reader(paths: List[RichPath], idx: int):
        task = FSMolTask.load_from_file(paths[0])

        try:
            sampled_task = task_sampler.sample(task)
        except Exception as e:
            return
        
        yield sampled_task

    
    return dataset.get_task_reading_iterable(
        data_fold=DataFold.TRAIN,
        task_reader_fn=simple_task_reader,
        repeat=repeat,
    )
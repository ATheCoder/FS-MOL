from fs_mol.data.fsmol_task_sampler import StratifiedTaskSampler
from fs_mol.data import FSMolDataset, DataFold, FSMolBatcher, FSMolTask
from dpu_utils.utils import RichPath
from torch_geometric.data import Dataset
from typing import Optional, Callable, List
from fs_mol.custom.utils import convert_to_pyg_graph

class FSMOL(Dataset):
    def __init__(self):
        super().__init__(root='/FS-MOL/GG')
        
        task_sampler = StratifiedTaskSampler(train_size_or_ratio=32, test_size_or_ratio=256)
        
        def task_reader(paths: List[RichPath], idx: int):
            task = FSMolTask.load_from_file(paths[0])
            
            task_sample = task_sampler.sample(task, seed=idx)
            
            # try:
            #     task_sample = task_sampler.sample(task, seed=idx + num_task_samples)
            #     num_task_samples += 1
            # except Exception as e:
            #     logger.debug(f"{task.name}: Sampling failed: {e}")
            #     continue
            
            train_samples = [convert_to_pyg_graph(mol) for mol in task_sample.train_samples]
            test_samples = [convert_to_pyg_graph(mol) for mol in task_sample.test_samples]
            
            
            yield train_samples, test_samples
            
        
        task_iterator = FSMolDataset.from_directory('/FS-MOL/datasets/fs-mol/', task_list_file=RichPath.create('/FS-MOL/datasets/fsmol-0.1.json'), num_workers=0).get_task_reading_iterable(DataFold.VALIDATION, task_reader_fn=task_reader, repeat=False)
        
        self.task_iterator = iter(task_iterator)
    
    def len(self):
        return 40
    
    def get(self, idx):
        res = next(self.task_iterator)
        return res
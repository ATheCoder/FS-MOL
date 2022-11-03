from more_itertools import partition
import torch
from fs_mol.custom.utils import convert_to_pyg_graph
from fs_mol.data.fsmol_task import FSMolTask, PyG_MoleculeDatapoint
from typing import List, Tuple
from dpu_utils.utils.richpath import RichPath


class PyGMolTask(FSMolTask):
    name: str
    samples: List[PyG_MoleculeDatapoint]
    
    def get_pos_neg_separated(self) -> Tuple[List[PyG_MoleculeDatapoint], List[PyG_MoleculeDatapoint]]:
        pos_samples, neg_samples = partition(pred=lambda s: s.y, iterable=self.samples)
        return list(pos_samples), list(neg_samples)
    
    def load_from_file(path: RichPath) -> "PyGMolTask":
        fsmol_task = FSMolTask.load_from_file(path)
        
        moleculeDatapoints = fsmol_task.samples
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        pyg_moleculardatapoints = list(map(lambda x: convert_to_pyg_graph(x, device=device), moleculeDatapoints))
        
        return PyGMolTask(fsmol_task.name, pyg_moleculardatapoints)